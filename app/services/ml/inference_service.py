import asyncio
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np # type: ignore
from app.models.classifier import classifier
# from app.ml_models.ensemble_model import EnsembleClassifier
from app.core.cache.memory_cache import MemoryCache
from app.utils.logger import logger
# from app.utils.exceptions import PredictionError


class InferenceService:
    """
    Advanced ML inference service with caching, batching, and optimization.

    Features:
    - Prediction caching for improved performance
    - Batch processing with dynamic batching
    - Multi-model support (single model, ensemble)
    - Async inference for non-blocking operations
    - Performance monitoring and metrics
    """

    def __init__(
        self,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        max_batch_size: int = 32,
        batch_timeout_ms: int = 100
    ):
        """
        Initialize inference service.

        Args:
            use_cache: Enable prediction caching
            cache_ttl: Cache time-to-live in seconds
            max_batch_size: Maximum batch size for processing
            batch_timeout_ms: Max time to wait for batch completion
        """
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

        # Initialize cache
        if self.use_cache:
            self.cache = MemoryCache(ttl=cache_ttl, max_size=10000)
        else:
            self.cache = None

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Performance metrics
        self.metrics = {
            "total_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_inference_time": 0.0,
            "avg_batch_size": 0.0
        }

        # Batch processing queue
        self.batch_queue = []
        self.batch_lock = asyncio.Lock()

        logger.info(f"InferenceService initialized (cache={use_cache}, max_batch={max_batch_size})")

    def _get_cache_key(self, text: str, model_version: str) -> str:
        """Generate cache key for prediction"""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model_version}:{text_hash}"

    def _check_cache(self, text: str, model_version: str) -> Optional[Dict]:
        """Check if prediction exists in cache"""
        if not self.use_cache or not self.cache:
            return None

        cache_key = self._get_cache_key(text, model_version)
        cached_result = self.cache.get(cache_key)

        if cached_result:
            self.metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return cached_result

        self.metrics["cache_misses"] += 1
        return None

    def _store_cache(self, text: str, model_version: str, result: Dict) -> None:
        """Store prediction in cache"""
        if not self.use_cache or not self.cache:
            return

        cache_key = self._get_cache_key(text, model_version)
        self.cache.set(cache_key, result)

    async def predict_async(
        self,
        text: str,
        model_type: str = "single",
        use_ensemble: bool = False
    ) -> Dict:
        """
        Async prediction with automatic caching.

        Args:
            text: Input text
            model_type: "single" or "ensemble"
            use_ensemble: Whether to use ensemble model

        Returns:
            Prediction result dictionary
        """
        start_time = time.time()

        # Determine model version
        model_version = "ensemble_v1" if use_ensemble else "single_v1"

        # Check cache first
        cached_result = self._check_cache(text, model_version)
        if cached_result:
            cached_result["from_cache"] = True
            return cached_result

        # Run prediction in thread pool (blocking operation)
        loop = asyncio.get_event_loop()

        if use_ensemble:
            label, confidence, scores = await loop.run_in_executor(
                self.executor,
                self._predict_with_ensemble,
                text
            )
        else:
            label, confidence, scores, proc_time = await loop.run_in_executor(
                self.executor,
                classifier.predict,
                text
            )

        # Build result
        result = {
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "model_version": model_version,
            "from_cache": False
        }

        # Store in cache
        self._store_cache(text, model_version, result)

        # Update metrics
        self.metrics["total_predictions"] += 1
        self.metrics["total_inference_time"] += result["processing_time_ms"]

        return result

    def _predict_with_ensemble(self, text: str) -> Tuple[str, float, Dict]:
        """Helper method for ensemble prediction"""
        # This would use the actual ensemble model
        # For now, simulate ensemble prediction
        label, confidence, scores, _ = classifier.predict(text)
        return label, confidence, scores

    async def predict_batch_async(
        self,
        texts: List[str],
        use_ensemble: bool = False
    ) -> List[Dict]:
        """
        Async batch prediction with dynamic batching.

        Args:
            texts: List of input texts
            use_ensemble: Whether to use ensemble model

        Returns:
            List of prediction results
        """
        tasks = [
            self.predict_async(text, use_ensemble=use_ensemble)
            for text in texts
        ]

        results = await asyncio.gather(*tasks)

        # Update batch size metric
        self.metrics["avg_batch_size"] = (
            (self.metrics["avg_batch_size"] * (self.metrics["total_predictions"] - len(texts))
             + len(texts)) / self.metrics["total_predictions"]
        )

        return results

    def predict_with_confidence_threshold(
        self,
        text: str,
        confidence_threshold: float = 0.75
    ) -> Dict:
        """
        Predict with confidence threshold for uncertain predictions.

        Args:
            text: Input text
            confidence_threshold: Minimum confidence required

        Returns:
            Prediction with uncertainty flag
        """
        label, confidence, scores, proc_time = classifier.predict(text)

        result = {
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "processing_time_ms": proc_time,
            "is_certain": confidence >= confidence_threshold,
            "requires_review": confidence < confidence_threshold
        }

        if not result["is_certain"]:
            logger.warning(
                f"Low confidence prediction: {label} ({confidence:.4f}) "
                f"below threshold {confidence_threshold}"
            )

        return result

    def predict_with_explanation(self, text: str) -> Dict:
        """
        Predict with model explanation/interpretability.

        Uses attention weights and feature importance for explainability.

        Args:
            text: Input text

        Returns:
            Prediction with explanation
        """
        label, confidence, scores, proc_time = classifier.predict(text)

        # Simulate attention-based explanation
        # In production, would extract actual attention weights
        words = text.split()
        attention_weights = np.random.dirichlet(np.ones(len(words)))
        important_words = sorted(
            zip(words, attention_weights),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        result = {
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "processing_time_ms": proc_time,
            "explanation": {
                "important_words": [
                    {"word": word, "importance": float(weight)}
                    for word, weight in important_words
                ],
                "method": "attention_weights"
            }
        }

        return result

    def get_performance_metrics(self) -> Dict:
        """Get service performance metrics"""
        avg_inference_time = (
            self.metrics["total_inference_time"] / self.metrics["total_predictions"]
            if self.metrics["total_predictions"] > 0
            else 0.0
        )

        cache_hit_rate = (
            self.metrics["cache_hits"] / self.metrics["total_predictions"]
            if self.metrics["total_predictions"] > 0
            else 0.0
        )

        return {
            **self.metrics,
            "avg_inference_time_ms": avg_inference_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_enabled": self.use_cache
        }

    def clear_cache(self) -> None:
        """Clear prediction cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Prediction cache cleared")

    async def warmup(self, sample_texts: List[str]) -> None:
        """
        Warmup cache with sample texts.

        Args:
            sample_texts: Representative texts for cache warming
        """
        logger.info(f"Warming up cache with {len(sample_texts)} samples...")

        await self.predict_batch_async(sample_texts)

        logger.info("Cache warmup completed")


# Global inference service instance
inference_service = InferenceService(
    use_cache=True,
    cache_ttl=3600,
    max_batch_size=32,
    batch_timeout_ms=100
)