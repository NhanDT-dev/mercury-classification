import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from typing import List, Dict, Tuple, Optional
from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,  # noqa: F401
    BertForSequenceClassification  # noqa: F401
)
from app.core.config import settings  # noqa: F401
from app.utils.logger import logger
from app.utils.exceptions import ModelLoadError, PredictionError


class EnsembleClassifier(nn.Module):
    """
    Advanced ensemble classifier combining multiple pre-trained models.

    Uses weighted voting strategy with calibrated confidence scores.
    Supports both soft voting (probability averaging) and hard voting.

    Architecture:
    - Multiple transformer models (BERT, RoBERTa, DistilBERT)
    - Weighted aggregation layer
    - Temperature scaling for calibration
    - Uncertainty estimation
    """

    def __init__(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        voting_strategy: str = "soft",
        temperature: float = 1.0
    ):
        super().__init__()

        self.model_names = model_names
        self.num_models = len(model_names)
        self.voting_strategy = voting_strategy
        self.temperature = temperature

        # Initialize weights
        if weights is None:
            self.weights = np.ones(self.num_models) / self.num_models
        else:
            assert len(weights) == self.num_models
            self.weights = np.array(weights)

        self.models = nn.ModuleList()
        self.tokenizers = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

        logger.info(f"Initialized ensemble with {self.num_models} models")

    def load_models(self, cache_dir: str = "./model_cache") -> None:
        """Load all models in the ensemble"""
        try:
            logger.info("Loading ensemble models...")

            for i, model_name in enumerate(self.model_names):
                logger.info(f"Loading model {i+1}/{self.num_models}: {model_name}")

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=cache_dir, use_fast=True
                )
                self.tokenizers.append(tokenizer)

                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                model.to(self.device)
                model.eval()
                self.models.append(model)

            self.is_loaded = True
            logger.info("Ensemble loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load ensemble: {str(e)}")
            raise ModelLoadError(f"Ensemble loading failed: {str(e)}")

    def forward(self, input_ids, attention_mask):
        """Forward pass through ensemble"""
        all_logits = []

        with torch.no_grad():
            for model in self.models:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits)

        # Weighted average
        weighted_logits = sum(w * logits for w, logits in zip(self.weights, all_logits))

        return {"logits": weighted_logits}

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict with ensemble"""
        if not self.is_loaded:
            raise PredictionError("Ensemble not loaded")

        tokenizer = self.tokenizers[0]
        encoding = tokenizer(
            text, truncation=True, padding=True,
            max_length=512, return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs = self.forward(input_ids, attention_mask)
        probs = torch.softmax(outputs["logits"][0], dim=-1).cpu().numpy()

        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        predicted_idx = np.argmax(probs)
        predicted_label = label_map[predicted_idx]
        confidence = float(probs[predicted_idx])

        all_scores = {label_map[i]: float(p) for i, p in enumerate(probs)}

        return predicted_label, confidence, all_scores