import time
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from app.core.config import settings
from app.utils.logger import logger
from app.utils.exceptions import ModelLoadError, PredictionError

class MedicalTextClassifier:
    """
    Medical text sentiment classifier using pre-trained transformer model.

    This classifier uses DistilBERT fine-tuned on SST-2 dataset for sentiment analysis.
    While not specifically trained on medical texts, it provides good general sentiment
    classification that can be applied to medical reviews and patient feedback.
    """

    def __init__(self):
        """Initialize the classifier"""
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = settings.MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

    def load_model(self) -> None:
        """
        Load the pre-trained model and tokenizer.

        Raises:
            ModelLoadError: If model fails to load
        """
        try:
            logger.info(f"Loading model: {self.model_name} on device: {self.device}")
            start_time = time.time()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )

            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else - 1,
                top_k=None  # Return scores for all labels
            )

            load_time = time.time() - start_time
            self.is_loaded = True

            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelLoadError(f"Could not load model {self.model_name}: {str(e)}")

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float], float]:
        """
        Predict sentiment for a single text.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (label, confidence, all_scores, processing_time_ms)

        Raises:
            PredictionError: If prediction fails
        """
        if not self.is_loaded:
            raise PredictionError("Model is not loaded. Please initialize the model first.")

        try:
            start_time = time.time()

            # Truncate if too long
            if len(text) > settings.MAX_TEXT_LENGTH:
                logger.warning(f"Text truncated from {len(text)} to {settings.MAX_TEXT_LENGTH} characters")
                text = text[:settings.MAX_TEXT_LENGTH]

            # Run prediction
            result = self.pipeline(text)[0]

            # Extract results
            all_scores = {item['label']: item['score'] for item in result}
            top_prediction = max(result, key=lambda x: x['score'])
            label = top_prediction['label']
            confidence = top_prediction['score']

            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            logger.debug(f"Prediction: {label} ({confidence:.4f}) - Time: {processing_time:.2f}ms")

            return label, confidence, all_scores, processing_time

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Failed to classify text: {str(e)}")

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float, Dict[str, float], float]]:
        """
        Predict sentiment for multiple texts.

        Args:
            texts: List of input texts to classify

        Returns:
            List of tuples (label, confidence, all_scores, processing_time_ms)

        Raises:
            PredictionError: If prediction fails
        """
        if not self.is_loaded:
            raise PredictionError("Model is not loaded. Please initialize the model first.")

        try:
            results = []

            # Process each text individually to track individual timing
            for text in texts:
                result = self.predict(text)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise PredictionError(f"Failed to classify texts: {str(e)}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "name": self.model_name,
            "version": "1.0.0",
            "type": "DistilBERT Sentiment Classifier",
            "max_length": settings.MAX_TEXT_LENGTH,
            "labels": ["POSITIVE", "NEGATIVE"],
            "device": self.device,
            "is_loaded": self.is_loaded,
            "description": "Pre-trained DistilBERT model fine-tuned on SST-2 for sentiment analysis"
        }

    def unload_model(self) -> None:
        """Unload the model to free memory"""
        if self.is_loaded:
            logger.info("Unloading model...")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            self.is_loaded = False

            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Model unloaded successfully")


# Global classifier instance
classifier = MedicalTextClassifier()
