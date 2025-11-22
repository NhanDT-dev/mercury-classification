"""Attention mechanism visualization and interpretability for transformer models"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from app.utils.logger import logger
from app.utils.exceptions import PredictionError


class AttentionVisualizer:
    """
    Visualize and analyze attention patterns in transformer models.

    Features:
    - Extract attention weights from all layers
    - Compute attention flow across layers
    - Identify important tokens via attention aggregation
    - Generate attention heatmaps
    - Attention rollout for multi-layer analysis
    - Head importance scoring
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize attention visualizer.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

        logger.info(f"AttentionVisualizer initialized for {model_name}")

    def load_model(self, cache_dir: str = "./model_cache") -> None:
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading model for attention analysis: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )

            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                output_attentions=True  # Critical: enable attention outputs
            )

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True

            # Get model architecture info
            config = self.model.config
            self.num_layers = config.num_hidden_layers
            self.num_heads = config.num_attention_heads

            logger.info(
                f"Model loaded: {self.num_layers} layers, "
                f"{self.num_heads} attention heads per layer"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise PredictionError(f"Model loading failed: {str(e)}")

    @torch.no_grad()
    def extract_attention(
        self,
        text: str,
        layer: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract attention weights from model.

        Args:
            text: Input text
            layer: Specific layer to extract (None = all layers)

        Returns:
            Tuple of (attention_weights, tokens)
        """
        if not self.is_loaded:
            raise PredictionError("Model not loaded")

        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Forward pass with attention output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract attention weights
        # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions

        # Convert tokens to strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        if layer is not None:
            # Specific layer
            attention_weights = attentions[layer][0].cpu().numpy()
        else:
            # All layers stacked
            attention_weights = torch.stack(attentions).squeeze(1).cpu().numpy()

        return attention_weights, tokens

    def compute_attention_rollout(
        self,
        text: str,
        discard_ratio: float = 0.1
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute attention rollout across all layers.

        Attention rollout combines attention from all layers to show
        how information flows through the network.

        Args:
            text: Input text
            discard_ratio: Ratio of lowest attention to discard

        Returns:
            Tuple of (rollout_attention, tokens)
        """
        attention_weights, tokens = self.extract_attention(text)

        # attention_weights shape: (num_layers, num_heads, seq_len, seq_len)
        num_layers, num_heads, seq_len, _ = attention_weights.shape

        # Average across heads
        attention_weights = attention_weights.mean(axis=1)
        # Shape: (num_layers, seq_len, seq_len)

        # Add identity matrix (residual connections)
        residual_att = np.eye(seq_len)
        aug_att_mat = attention_weights + np.expand_dims(residual_att, 0)

        # Normalize rows
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1, keepdims=True)

        # Discard lowest attention values
        if discard_ratio > 0:
            flat = aug_att_mat.reshape(num_layers, -1)
            threshold = np.percentile(flat, discard_ratio * 100, axis=-1, keepdims=True)
            threshold = threshold.reshape(num_layers, 1, 1)
            aug_att_mat[aug_att_mat < threshold] = 0

            # Re-normalize
            aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1, keepdims=True)

        # Multiply attention matrices across layers (rollout)
        joint_attentions = np.zeros((num_layers + 1, seq_len, seq_len))
        joint_attentions[0] = residual_att

        for i in range(num_layers):
            joint_attentions[i + 1] = aug_att_mat[i] @ joint_attentions[i]

        # Final rollout attention
        rollout_attention = joint_attentions[-1]

        return rollout_attention, tokens

    def get_important_tokens(
        self,
        text: str,
        top_k: int = 5,
        method: str = "rollout"
    ) -> List[Dict[str, float]]:
        """
        Identify most important tokens based on attention.

        Args:
            text: Input text
            top_k: Number of top tokens to return
            method: "rollout" or "mean" aggregation

        Returns:
            List of dicts with token and importance score
        """
        if method == "rollout":
            attention, tokens = self.compute_attention_rollout(text)
            # Sum attention to [CLS] token from all other tokens
            importance_scores = attention[0, 1:]  # Skip [CLS] itself
        else:
            # Mean attention across all layers and heads
            attention, tokens = self.extract_attention(text)
            # Average across layers and heads
            attention_mean = attention.mean(axis=(0, 1))
            importance_scores = attention_mean[0, 1:]  # Attention to [CLS]

        # Create token-score pairs (skip special tokens)
        token_scores = []
        for i, (token, score) in enumerate(zip(tokens[1:], importance_scores)):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_scores.append({
                    "token": token,
                    "importance": float(score),
                    "position": i + 1
                })

        # Sort by importance
        token_scores.sort(key=lambda x: x["importance"], reverse=True)

        return token_scores[:top_k]

    def compute_head_importance(
        self,
        text: str,
        layer: int = -1
    ) -> np.ndarray:
        """
        Compute importance score for each attention head.

        Args:
            text: Input text
            layer: Layer to analyze (-1 for last layer)

        Returns:
            Array of importance scores per head
        """
        attention, _ = self.extract_attention(text, layer=layer)
        # Shape: (num_heads, seq_len, seq_len)

        # Importance: variance of attention distribution
        # Higher variance = head is more selective
        head_importance = attention.var(axis=(1, 2))

        return head_importance

    def generate_attention_heatmap(
        self,
        text: str,
        layer: int = -1,
        head: int = 0,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate attention heatmap for visualization.

        Args:
            text: Input text
            layer: Layer index
            head: Attention head index
            save_path: Optional path to save figure

        Returns:
            Attention matrix
        """
        attention, tokens = self.extract_attention(text, layer=layer)
        # Shape: (num_heads, seq_len, seq_len)

        # Get specific head
        head_attention = attention[head]

        if save_path:
            plt.figure(figsize=(10, 10))
            plt.imshow(head_attention, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(tokens)), tokens, rotation=90)
            plt.yticks(range(len(tokens)), tokens)
            plt.xlabel("Keys")
            plt.ylabel("Queries")
            plt.title(f"Attention Head {head}, Layer {layer}")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Attention heatmap saved to {save_path}")

        return head_attention

    def analyze_attention_patterns(
        self,
        text: str
    ) -> Dict:
        """
        Comprehensive attention analysis.

        Args:
            text: Input text

        Returns:
            Dict with various attention metrics
        """
        # Extract all attention
        attention, tokens = self.extract_attention(text)

        # Compute rollout
        rollout_attention, _ = self.compute_attention_rollout(text)

        # Important tokens
        important_tokens = self.get_important_tokens(text, top_k=5)

        # Head importance for last layer
        head_importance = self.compute_head_importance(text, layer=-1)

        # Attention statistics
        attention_stats = {
            "mean": float(attention.mean()),
            "std": float(attention.std()),
            "max": float(attention.max()),
            "min": float(attention.min())
        }

        # Attention entropy (measure of focus)
        # Lower entropy = more focused attention
        def compute_entropy(p):
            p = p + 1e-10  # Avoid log(0)
            return -np.sum(p * np.log(p), axis=-1)

        attention_entropy = compute_entropy(attention).mean()

        return {
            "num_tokens": len(tokens),
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "important_tokens": important_tokens,
            "head_importance": head_importance.tolist(),
            "attention_stats": attention_stats,
            "attention_entropy": float(attention_entropy),
            "rollout_to_cls": rollout_attention[0].tolist()
        }


class AttentionFlowAnalyzer:
    """
    Analyze information flow through transformer layers via attention.

    Tracks how information from each token propagates through the network.
    """

    def __init__(self, visualizer: AttentionVisualizer):
        self.visualizer = visualizer

    def compute_information_flow(
        self,
        text: str,
        source_token_idx: int
    ) -> np.ndarray:
        """
        Compute how information from a specific token flows through layers.

        Args:
            text: Input text
            source_token_idx: Index of source token to track

        Returns:
            Flow matrix (num_layers, seq_len)
        """
        attention, tokens = self.visualizer.extract_attention(text)
        num_layers, num_heads, seq_len, _ = attention.shape

        # Average across heads
        attention = attention.mean(axis=1)

        # Track flow from source token
        flow = np.zeros((num_layers, seq_len))
        flow[0, source_token_idx] = 1.0  # Start at source

        for layer in range(1, num_layers):
            # Flow from previous layer through current attention
            flow[layer] = attention[layer - 1] @ flow[layer - 1]

        return flow

    def identify_attention_sinks(
        self,
        text: str,
        threshold: float = 0.1
    ) -> List[int]:
        """
        Identify tokens that receive disproportionate attention (sinks).

        Args:
            text: Input text
            threshold: Relative threshold for sink identification

        Returns:
            List of token indices that are attention sinks
        """
        attention, tokens = self.visualizer.extract_attention(text)

        # Sum attention received by each token across all layers/heads
        attention_received = attention.sum(axis=(0, 1, 2))

        # Normalize
        attention_received = attention_received / attention_received.sum()

        # Find tokens receiving more than threshold
        mean_attention = 1.0 / len(tokens)
        sinks = np.where(attention_received > mean_attention * (1 + threshold))[0]

        return sinks.tolist()


# Factory function
def create_attention_visualizer(
    model_name: str = "bert-base-uncased",
    cache_dir: str = "./model_cache"
) -> AttentionVisualizer:
    """
    Create and load attention visualizer.

    Args:
        model_name: Model to analyze
        cache_dir: Cache directory

    Returns:
        Loaded AttentionVisualizer instance
    """
    visualizer = AttentionVisualizer(model_name)
    visualizer.load_model(cache_dir)
    return visualizer
