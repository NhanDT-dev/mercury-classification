"""Advanced embedding analysis and semantic similarity computation"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine, euclidean
from app.utils.logger import logger


class EmbeddingAnalyzer:
    """
    Analyze and compute text embeddings for semantic similarity and clustering.

    Features:
    - Extract contextual embeddings (BERT, RoBERTa, etc.)
    - Compute semantic similarity (cosine, euclidean)
    - Dimensionality reduction (PCA, t-SNE)
    - Clustering analysis
    - Nearest neighbor search
    - Embedding visualization
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize embedding analyzer.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

        # Embedding cache
        self.embedding_cache = {}

        logger.info(f"EmbeddingAnalyzer initialized for {model_name}")

    def load_model(self, cache_dir: str = "./model_cache") -> None:
        """Load embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )

            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True

            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    @torch.no_grad()
    def get_embedding(
        self,
        text: str,
        pooling_strategy: str = "cls"
    ) -> np.ndarray:
        """
        Extract embedding for a single text.

        Args:
            text: Input text
            pooling_strategy: "cls", "mean", or "max"

        Returns:
            Embedding vector
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded")

        # Check cache
        cache_key = f"{text}_{pooling_strategy}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get hidden states
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Apply pooling
        if pooling_strategy == "cls":
            # Use [CLS] token embedding
            embedding = last_hidden_state[:, 0, :].squeeze()

        elif pooling_strategy == "mean":
            # Mean pooling over all tokens (excluding padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze()

        elif pooling_strategy == "max":
            # Max pooling
            embedding = torch.max(last_hidden_state, dim=1).values.squeeze()

        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        # Convert to numpy
        embedding_np = embedding.cpu().numpy()

        # Cache the result
        self.embedding_cache[cache_key] = embedding_np

        return embedding_np

    def get_batch_embeddings(
        self,
        texts: List[str],
        pooling_strategy: str = "cls",
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Extract embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts
            pooling_strategy: Pooling method
            batch_size: Batch size for processing

        Returns:
            Matrix of embeddings (n_texts, embedding_dim)
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = [
                self.get_embedding(text, pooling_strategy)
                for text in batch_texts
            ]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def compute_similarity(
        self,
        text1: str,
        text2: str,
        metric: str = "cosine",
        pooling: str = "cls"
    ) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            metric: "cosine" or "euclidean"
            pooling: Pooling strategy

        Returns:
            Similarity score
        """
        emb1 = self.get_embedding(text1, pooling)
        emb2 = self.get_embedding(text2, pooling)

        if metric == "cosine":
            # Cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(emb1, emb2)

        elif metric == "euclidean":
            # Negative euclidean distance (higher = more similar)
            similarity = -euclidean(emb1, emb2)

        else:
            raise ValueError(f"Unknown metric: {metric}")

        return float(similarity)

    def find_similar_texts(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """
        Find most similar texts to query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return
            metric: Similarity metric

        Returns:
            List of (text, similarity_score) tuples
        """
        query_emb = self.get_embedding(query)
        candidate_embs = self.get_batch_embeddings(candidates)

        # Compute similarities
        similarities = []
        for i, cand_emb in enumerate(candidate_embs):
            if metric == "cosine":
                sim = 1 - cosine(query_emb, cand_emb)
            else:
                sim = -euclidean(query_emb, cand_emb)

            similarities.append((candidates[i], float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def cluster_texts(
        self,
        texts: List[str],
        n_clusters: int = 3,
        pooling: str = "cls"
    ) -> Dict:
        """
        Cluster texts based on embeddings.

        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters
            pooling: Pooling strategy

        Returns:
            Dict with cluster assignments and statistics
        """
        # Get embeddings
        embeddings = self.get_batch_embeddings(texts, pooling)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Organize results
        clusters = {i: [] for i in range(n_clusters)}
        for text, label in zip(texts, cluster_labels):
            clusters[label].append(text)

        # Compute cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_embs = embeddings[cluster_labels == i]
            cluster_stats[i] = {
                "size": len(clusters[i]),
                "centroid": kmeans.cluster_centers_[i].tolist(),
                "inertia": np.sum(
                    np.linalg.norm(cluster_embs - kmeans.cluster_centers_[i], axis=1) ** 2
                )
            }

        return {
            "clusters": clusters,
            "labels": cluster_labels.tolist(),
            "statistics": cluster_stats,
            "total_inertia": float(kmeans.inertia_)
        }

    def reduce_dimensions(
        self,
        texts: List[str],
        method: str = "pca",
        n_components: int = 2,
        pooling: str = "cls"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce embedding dimensionality for visualization.

        Args:
            texts: List of texts
            method: "pca" or "tsne"
            n_components: Target dimensions
            pooling: Pooling strategy

        Returns:
            Tuple of (reduced_embeddings, original_embeddings)
        """
        # Get high-dimensional embeddings
        embeddings = self.get_batch_embeddings(texts, pooling)

        # Apply dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(texts)-1))
        else:
            raise ValueError(f"Unknown method: {method}")

        reduced_embeddings = reducer.fit_transform(embeddings)

        return reduced_embeddings, embeddings

    def analyze_semantic_space(
        self,
        texts: List[str],
        pooling: str = "cls"
    ) -> Dict:
        """
        Comprehensive analysis of semantic space.

        Args:
            texts: List of texts
            pooling: Pooling strategy

        Returns:
            Dict with various metrics
        """
        embeddings = self.get_batch_embeddings(texts, pooling)

        # Compute pairwise similarities
        n_texts = len(texts)
        similarity_matrix = np.zeros((n_texts, n_texts))

        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        # Statistics
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        max_similarity = np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        min_similarity = np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])

        # Find most/least similar pairs
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        flat_similarities = similarity_matrix[upper_tri_indices]

        most_similar_idx = np.argmax(flat_similarities)
        least_similar_idx = np.argmin(flat_similarities)

        i_most, j_most = upper_tri_indices[0][most_similar_idx], upper_tri_indices[1][most_similar_idx]
        i_least, j_least = upper_tri_indices[0][least_similar_idx], upper_tri_indices[1][least_similar_idx]

        return {
            "num_texts": n_texts,
            "embedding_dim": embeddings.shape[1],
            "avg_similarity": float(avg_similarity),
            "max_similarity": float(max_similarity),
            "min_similarity": float(min_similarity),
            "most_similar_pair": {
                "texts": (texts[i_most], texts[j_most]),
                "similarity": float(similarity_matrix[i_most, j_most])
            },
            "least_similar_pair": {
                "texts": (texts[i_least], texts[j_least]),
                "similarity": float(similarity_matrix[i_least, j_least])
            }
        }


class SemanticSearchEngine:
    """
    Semantic search engine using embeddings.

    Enables finding similar texts based on meaning rather than keywords.
    """

    def __init__(self, analyzer: EmbeddingAnalyzer):
        self.analyzer = analyzer
        self.index = {}  # Simple in-memory index

    def index_texts(self, texts: List[str], ids: Optional[List[str]] = None) -> None:
        """
        Index texts for searching.

        Args:
            texts: List of texts to index
            ids: Optional IDs for texts
        """
        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        embeddings = self.analyzer.get_batch_embeddings(texts)

        for text, text_id, embedding in zip(texts, ids, embeddings):
            self.index[text_id] = {
                "text": text,
                "embedding": embedding
            }

        logger.info(f"Indexed {len(texts)} texts")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar texts.

        Args:
            query: Search query
            top_k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of search results with scores
        """
        query_emb = self.analyzer.get_embedding(query)

        # Compute similarities
        results = []
        for text_id, data in self.index.items():
            similarity = 1 - cosine(query_emb, data["embedding"])

            if similarity >= min_similarity:
                results.append({
                    "id": text_id,
                    "text": data["text"],
                    "similarity": float(similarity)
                })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]
