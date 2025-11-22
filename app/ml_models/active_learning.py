"""Active learning strategies for efficient model training with minimal labeled data"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from scipy.stats import entropy
from sklearn.cluster import KMeans
from app.utils.logger import logger


class ActiveLearningSelector:
    """
    Select most informative samples for labeling using active learning strategies.

    Strategies:
    - Uncertainty sampling (least confident, margin, entropy)
    - Query-by-committee (model disagreement)
    - Diversity sampling (representative selection)
    - Expected model change
    - Hybrid strategies
    """

    def __init__(self, strategy: str = "uncertainty"):
        """
        Initialize active learning selector.

        Args:
            strategy: Selection strategy name
        """
        self.strategy = strategy
        self.labeled_indices = set()
        self.selection_history = []

        logger.info(f"ActiveLearningSelector initialized with strategy: {strategy}")

    def uncertainty_sampling(
        self,
        predictions: np.ndarray,
        method: str = "least_confident"
    ) -> np.ndarray:
        """
        Select samples based on prediction uncertainty.

        Args:
            predictions: Model predictions (n_samples, n_classes) - probabilities
            method: "least_confident", "margin", or "entropy"

        Returns:
            Uncertainty scores (higher = more uncertain)
        """
        n_samples = predictions.shape[0]

        if method == "least_confident":
            # Samples where max probability is lowest
            max_probs = np.max(predictions, axis=1)
            uncertainty = 1 - max_probs

        elif method == "margin":
            # Difference between top two probabilities
            sorted_probs = np.sort(predictions, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            uncertainty = 1 - margin

        elif method == "entropy":
            # Prediction entropy
            uncertainty = entropy(predictions.T)  # Compute along class dimension

        else:
            raise ValueError(f"Unknown uncertainty method: {method}")

        return uncertainty

    def query_by_committee(
        self,
        committee_predictions: List[np.ndarray],
        method: str = "vote_entropy"
    ) -> np.ndarray:
        """
        Select samples where committee members disagree most.

        Args:
            committee_predictions: List of predictions from different models
                Each array has shape (n_samples, n_classes)
            method: "vote_entropy" or "kl_divergence"

        Returns:
            Disagreement scores
        """
        n_samples = committee_predictions[0].shape[0]
        n_models = len(committee_predictions)

        if method == "vote_entropy":
            # Compute vote entropy
            # Get hard predictions (class with max probability)
            hard_preds = np.array([
                np.argmax(preds, axis=1)
                for preds in committee_predictions
            ])  # (n_models, n_samples)

            # Count votes for each class
            vote_counts = np.array([
                np.bincount(hard_preds[:, i], minlength=committee_predictions[0].shape[1])
                for i in range(n_samples)
            ])  # (n_samples, n_classes)

            # Normalize to get vote distribution
            vote_dist = vote_counts / n_models

            # Compute entropy of vote distribution
            disagreement = entropy(vote_dist.T)

        elif method == "kl_divergence":
            # Average KL divergence between models
            # Compute consensus (mean prediction)
            consensus = np.mean(committee_predictions, axis=0)

            # KL divergence from each model to consensus
            kl_divs = []
            for preds in committee_predictions:
                kl = np.sum(preds * np.log((preds + 1e-10) / (consensus + 1e-10)), axis=1)
                kl_divs.append(kl)

            # Average KL divergence
            disagreement = np.mean(kl_divs, axis=0)

        else:
            raise ValueError(f"Unknown QBC method: {method}")

        return disagreement

    def diversity_sampling(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None,
        method: str = "kmeans"
    ) -> np.ndarray:
        """
        Select diverse samples using clustering.

        Args:
            embeddings: Sample embeddings (n_samples, embedding_dim)
            n_clusters: Number of clusters (None = auto)
            method: "kmeans" or "density"

        Returns:
            Diversity scores
        """
        n_samples = embeddings.shape[0]

        if n_clusters is None:
            n_clusters = min(10, n_samples // 10)

        if method == "kmeans":
            # Cluster and select samples closest to centroids
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Distance to nearest centroid
            distances = np.min(
                np.linalg.norm(
                    embeddings[:, np.newaxis, :] - kmeans.cluster_centers_[np.newaxis, :, :],
                    axis=2
                ),
                axis=1
            )

            # Invert so higher = more representative
            diversity = 1 / (distances + 1e-10)

        elif method == "density":
            # Select samples from low-density regions
            # Compute local density (distance to k nearest neighbors)
            k = min(10, n_samples - 1)

            densities = []
            for i in range(n_samples):
                # Distances to all other points
                dists = np.linalg.norm(embeddings - embeddings[i], axis=1)

                # Distance to k-th nearest neighbor
                knn_dist = np.partition(dists, k)[k]
                densities.append(knn_dist)

            diversity = np.array(densities)

        else:
            raise ValueError(f"Unknown diversity method: {method}")

        return diversity

    def expected_model_change(
        self,
        predictions: np.ndarray,
        gradient_norms: np.ndarray
    ) -> np.ndarray:
        """
        Select samples that would change model most if labeled.

        Args:
            predictions: Model predictions (n_samples, n_classes)
            gradient_norms: Gradient norms for each sample

        Returns:
            Expected model change scores
        """
        # Samples with high uncertainty and large gradients
        uncertainty = entropy(predictions.T)

        # Normalize both to [0, 1]
        uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-10)
        gradient_norm = (gradient_norms - gradient_norms.min()) / (gradient_norms.max() - gradient_norms.min() + 1e-10)

        # Combined score
        emc_score = uncertainty_norm * gradient_norm

        return emc_score

    def hybrid_sampling(
        self,
        predictions: np.ndarray,
        embeddings: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Combine uncertainty and diversity.

        Args:
            predictions: Model predictions
            embeddings: Sample embeddings
            alpha: Weight for uncertainty (1-alpha for diversity)

        Returns:
            Combined scores
        """
        # Uncertainty component
        uncertainty = self.uncertainty_sampling(predictions, method="entropy")

        # Diversity component
        diversity = self.diversity_sampling(embeddings, method="density")

        # Normalize to [0, 1]
        uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-10)
        diversity_norm = (diversity - diversity.min()) / (diversity.max() - diversity.min() + 1e-10)

        # Weighted combination
        combined = alpha * uncertainty_norm + (1 - alpha) * diversity_norm

        return combined

    def select_samples(
        self,
        predictions: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        committee_predictions: Optional[List[np.ndarray]] = None,
        gradient_norms: Optional[np.ndarray] = None,
        n_samples: int = 10,
        exclude_labeled: bool = True
    ) -> List[int]:
        """
        Select samples for labeling.

        Args:
            predictions: Single model predictions
            embeddings: Sample embeddings
            committee_predictions: Predictions from multiple models
            gradient_norms: Gradient norms
            n_samples: Number of samples to select
            exclude_labeled: Exclude already labeled samples

        Returns:
            List of selected sample indices
        """
        # Compute scores based on strategy
        if self.strategy == "uncertainty":
            if predictions is None:
                raise ValueError("Predictions required for uncertainty sampling")
            scores = self.uncertainty_sampling(predictions)

        elif self.strategy == "qbc":
            if committee_predictions is None:
                raise ValueError("Committee predictions required for QBC")
            scores = self.query_by_committee(committee_predictions)

        elif self.strategy == "diversity":
            if embeddings is None:
                raise ValueError("Embeddings required for diversity sampling")
            scores = self.diversity_sampling(embeddings)

        elif self.strategy == "emc":
            if predictions is None or gradient_norms is None:
                raise ValueError("Predictions and gradients required for EMC")
            scores = self.expected_model_change(predictions, gradient_norms)

        elif self.strategy == "hybrid":
            if predictions is None or embeddings is None:
                raise ValueError("Predictions and embeddings required for hybrid")
            scores = self.hybrid_sampling(predictions, embeddings)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Mask already labeled samples if requested
        if exclude_labeled and self.labeled_indices:
            scores = scores.copy()
            for idx in self.labeled_indices:
                scores[idx] = -np.inf

        # Select top-k samples
        selected_indices = np.argsort(scores)[-n_samples:][::-1].tolist()

        # Update labeled set
        self.labeled_indices.update(selected_indices)
        self.selection_history.append(selected_indices)

        logger.info(f"Selected {len(selected_indices)} samples using {self.strategy}")

        return selected_indices

    def get_selection_statistics(self) -> Dict:
        """Get statistics about selection history"""
        return {
            "total_labeled": len(self.labeled_indices),
            "selection_rounds": len(self.selection_history),
            "strategy": self.strategy,
            "labeled_indices": list(self.labeled_indices)
        }


class ActiveLearningPipeline:
    """
    Complete active learning pipeline.

    Manages the iterative process of:
    1. Training initial model
    2. Selecting informative samples
    3. Getting labels
    4. Retraining model
    5. Repeating until budget exhausted
    """

    def __init__(
        self,
        selector: ActiveLearningSelector,
        model_trainer: Callable,
        initial_samples: int = 10,
        budget: int = 100
    ):
        """
        Initialize active learning pipeline.

        Args:
            selector: ActiveLearningSelector instance
            model_trainer: Function to train model
            initial_samples: Number of initial random samples
            budget: Total labeling budget
        """
        self.selector = selector
        self.model_trainer = model_trainer
        self.initial_samples = initial_samples
        self.budget = budget

        self.labeled_data = []
        self.unlabeled_data = []
        self.performance_history = []

        logger.info(f"ActiveLearningPipeline initialized (budget={budget})")

    def initialize(
        self,
        all_data: List,
        initial_labels: Optional[List] = None
    ) -> None:
        """
        Initialize with random samples or provided labels.

        Args:
            all_data: All available data
            initial_labels: Optional initial labels
        """
        if initial_labels is not None:
            self.labeled_data = initial_labels
        else:
            # Random initial selection
            indices = np.random.choice(
                len(all_data),
                self.initial_samples,
                replace=False
            )

            self.labeled_data = [all_data[i] for i in indices]
            self.selector.labeled_indices.update(indices.tolist())

        # Remaining data is unlabeled
        unlabeled_indices = [
            i for i in range(len(all_data))
            if i not in self.selector.labeled_indices
        ]
        self.unlabeled_data = [all_data[i] for i in unlabeled_indices]

        logger.info(
            f"Initialized with {len(self.labeled_data)} labeled, "
            f"{len(self.unlabeled_data)} unlabeled samples"
        )

    def run_iteration(
        self,
        n_select: int,
        labeling_function: Callable
    ) -> Dict:
        """
        Run one iteration of active learning.

        Args:
            n_select: Number of samples to select
            labeling_function: Function to get labels for selected samples

        Returns:
            Iteration statistics
        """
        # Train model on current labeled data
        model, predictions, embeddings = self.model_trainer(
            self.labeled_data,
            self.unlabeled_data
        )

        # Select samples
        selected_indices = self.selector.select_samples(
            predictions=predictions,
            embeddings=embeddings,
            n_samples=n_select
        )

        # Get labels
        newly_labeled = []
        for idx in selected_indices:
            sample = self.unlabeled_data[idx]
            label = labeling_function(sample)
            newly_labeled.append((sample, label))

        # Update datasets
        self.labeled_data.extend(newly_labeled)

        # Remove from unlabeled
        self.unlabeled_data = [
            sample for i, sample in enumerate(self.unlabeled_data)
            if i not in selected_indices
        ]

        # Evaluate
        performance = self.evaluate_model(model)
        self.performance_history.append(performance)

        iteration_stats = {
            "labeled_count": len(self.labeled_data),
            "unlabeled_count": len(self.unlabeled_data),
            "selected_indices": selected_indices,
            "performance": performance
        }

        logger.info(
            f"Iteration complete: {len(self.labeled_data)} labeled, "
            f"performance: {performance:.4f}"
        )

        return iteration_stats

    def evaluate_model(self, model) -> float:
        """
        Evaluate model performance.

        Args:
            model: Trained model

        Returns:
            Performance metric (e.g., accuracy)
        """
        # Placeholder - implement actual evaluation
        return 0.85

    def run_full_pipeline(
        self,
        iterations: int,
        samples_per_iteration: int,
        labeling_function: Callable
    ) -> List[Dict]:
        """
        Run complete active learning pipeline.

        Args:
            iterations: Number of iterations
            samples_per_iteration: Samples to label per iteration
            labeling_function: Function to get labels

        Returns:
            List of iteration statistics
        """
        results = []

        for i in range(iterations):
            logger.info(f"Starting iteration {i+1}/{iterations}")

            stats = self.run_iteration(samples_per_iteration, labeling_function)
            results.append(stats)

            # Check budget
            if len(self.labeled_data) >= self.budget:
                logger.info("Budget exhausted, stopping")
                break

        return results
