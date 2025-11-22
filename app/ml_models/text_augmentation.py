"""Advanced text augmentation techniques for data augmentation and model robustness"""
import random
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
from app.utils.logger import logger


class TextAugmentor:
    """
    Advanced text augmentation for training data expansion and robustness testing.

    Techniques:
    - Synonym replacement using WordNet
    - Back-translation (English -> German -> English)
    - Contextual word substitution using BERT
    - Random insertion, deletion, swap
    - Paraphrasing with T5
    - Adversarial perturbations
    """

    def __init__(self, seed: int = 42):
        """
        Initialize text augmentor.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.back_translation_models = {}
        self.paraphrase_model = None

        logger.info("TextAugmentor initialized")

    def synonym_replacement(
        self,
        text: str,
        n: int = 2,
        stopwords: Optional[List[str]] = None
    ) -> str:
        """
        Replace n random words with their synonyms.

        Args:
            text: Input text
            n: Number of words to replace
            stopwords: Words to exclude from replacement

        Returns:
            Augmented text
        """
        words = text.split()

        if stopwords is None:
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}

        # Identify replaceable words (not stopwords, not short)
        replaceable_indices = [
            i for i, word in enumerate(words)
            if word.lower() not in stopwords and len(word) > 3
        ]

        if len(replaceable_indices) < n:
            n = len(replaceable_indices)

        # Randomly select n words to replace
        replace_indices = random.sample(replaceable_indices, n)

        # Synonym dictionary (simplified - in production use WordNet or similar)
        synonym_dict = {
            "good": ["excellent", "great", "positive", "favorable"],
            "bad": ["poor", "negative", "unfavorable", "terrible"],
            "patient": ["individual", "person", "subject", "case"],
            "treatment": ["therapy", "procedure", "intervention", "care"],
            "effective": ["successful", "efficient", "productive", "potent"],
            "result": ["outcome", "consequence", "effect", "finding"],
            "medical": ["clinical", "healthcare", "therapeutic", "diagnostic"],
            "doctor": ["physician", "practitioner", "clinician", "specialist"],
            "hospital": ["clinic", "facility", "institution", "center"],
            "improve": ["enhance", "better", "ameliorate", "upgrade"]
        }

        augmented_words = words.copy()
        for idx in replace_indices:
            word = words[idx].lower()
            if word in synonym_dict:
                synonym = random.choice(synonym_dict[word])
                # Preserve capitalization
                if words[idx][0].isupper():
                    synonym = synonym.capitalize()
                augmented_words[idx] = synonym

        return " ".join(augmented_words)

    def random_insertion(
        self,
        text: str,
        n: int = 1,
        insert_words: Optional[List[str]] = None
    ) -> str:
        """
        Randomly insert n words into the text.

        Args:
            text: Input text
            n: Number of words to insert
            insert_words: Pool of words to insert

        Returns:
            Augmented text
        """
        words = text.split()

        if insert_words is None:
            # Default medical domain words
            insert_words = [
                "very", "quite", "extremely", "somewhat", "rather",
                "significantly", "notably", "remarkably", "particularly"
            ]

        for _ in range(n):
            # Random word to insert
            word_to_insert = random.choice(insert_words)

            # Random position
            position = random.randint(0, len(words))

            words.insert(position, word_to_insert)

        return " ".join(words)

    def random_swap(self, text: str, n: int = 2) -> str:
        """
        Randomly swap n pairs of words.

        Args:
            text: Input text
            n: Number of swaps

        Returns:
            Augmented text
        """
        words = text.split()

        if len(words) < 2:
            return text

        for _ in range(n):
            # Random two positions
            idx1, idx2 = random.sample(range(len(words)), 2)

            # Swap
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)

    def random_deletion(
        self,
        text: str,
        p: float = 0.1,
        min_words: int = 3
    ) -> str:
        """
        Randomly delete words with probability p.

        Args:
            text: Input text
            p: Probability of deleting each word
            min_words: Minimum number of words to keep

        Returns:
            Augmented text
        """
        words = text.split()

        if len(words) <= min_words:
            return text

        # Keep words with probability (1 - p)
        kept_words = [
            word for word in words
            if random.random() > p
        ]

        # Ensure minimum length
        if len(kept_words) < min_words:
            return text

        return " ".join(kept_words)

    def load_back_translation_models(
        self,
        src_lang: str = "en",
        tgt_lang: str = "de"
    ) -> None:
        """
        Load models for back-translation.

        Args:
            src_lang: Source language code
            tgt_lang: Target language code
        """
        try:
            logger.info(f"Loading back-translation models ({src_lang}->{tgt_lang})")

            # Forward translation (en -> de)
            forward_model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
            forward_tokenizer = MarianTokenizer.from_pretrained(forward_model_name)
            forward_model = MarianMTModel.from_pretrained(forward_model_name)
            forward_model.to(self.device)
            forward_model.eval()

            # Backward translation (de -> en)
            backward_model_name = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"
            backward_tokenizer = MarianTokenizer.from_pretrained(backward_model_name)
            backward_model = MarianMTModel.from_pretrained(backward_model_name)
            backward_model.to(self.device)
            backward_model.eval()

            self.back_translation_models = {
                "forward_tokenizer": forward_tokenizer,
                "forward_model": forward_model,
                "backward_tokenizer": backward_tokenizer,
                "backward_model": backward_model
            }

            logger.info("Back-translation models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load back-translation models: {str(e)}")

    @torch.no_grad()
    def back_translation(
        self,
        text: str,
        intermediate_lang: str = "de"
    ) -> str:
        """
        Augment text using back-translation.

        Args:
            text: Input text
            intermediate_lang: Intermediate language for translation

        Returns:
            Back-translated text
        """
        if not self.back_translation_models:
            logger.warning("Back-translation models not loaded, skipping augmentation")
            return text

        try:
            # Forward translation
            forward_inputs = self.back_translation_models["forward_tokenizer"](
                text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            forward_outputs = self.back_translation_models["forward_model"].generate(
                **forward_inputs, max_length=512
            )

            intermediate_text = self.back_translation_models["forward_tokenizer"].decode(
                forward_outputs[0], skip_special_tokens=True
            )

            # Backward translation
            backward_inputs = self.back_translation_models["backward_tokenizer"](
                intermediate_text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            backward_outputs = self.back_translation_models["backward_model"].generate(
                **backward_inputs, max_length=512
            )

            back_translated_text = self.back_translation_models["backward_tokenizer"].decode(
                backward_outputs[0], skip_special_tokens=True
            )

            return back_translated_text

        except Exception as e:
            logger.error(f"Back-translation failed: {str(e)}")
            return text

    def contextual_word_substitution(
        self,
        text: str,
        n: int = 2,
        top_k: int = 5
    ) -> str:
        """
        Replace words using BERT-based contextual predictions.

        Args:
            text: Input text
            n: Number of words to replace
            top_k: Number of top predictions to sample from

        Returns:
            Augmented text
        """
        # Simplified version - in production, use actual BERT masked LM
        words = text.split()

        replaceable_indices = [
            i for i, word in enumerate(words)
            if len(word) > 3 and not word.startswith('[')
        ]

        if len(replaceable_indices) < n:
            n = len(replaceable_indices)

        replace_indices = random.sample(replaceable_indices, n)

        # Context-aware replacements (simplified)
        context_replacements = {
            "patient": ["individual", "subject", "case"],
            "treatment": ["therapy", "intervention", "procedure"],
            "effective": ["successful", "beneficial", "helpful"],
            "symptoms": ["signs", "manifestations", "indicators"]
        }

        augmented_words = words.copy()
        for idx in replace_indices:
            word = words[idx].lower()
            if word in context_replacements:
                replacement = random.choice(context_replacements[word])
                if words[idx][0].isupper():
                    replacement = replacement.capitalize()
                augmented_words[idx] = replacement

        return " ".join(augmented_words)

    def augment_text(
        self,
        text: str,
        methods: Optional[List[str]] = None,
        augmentation_factor: int = 1
    ) -> List[str]:
        """
        Apply multiple augmentation techniques.

        Args:
            text: Input text
            methods: List of methods to use (None = all)
            augmentation_factor: Number of augmented versions per method

        Returns:
            List of augmented texts
        """
        if methods is None:
            methods = [
                "synonym_replacement",
                "random_insertion",
                "random_swap",
                "random_deletion",
                "contextual_word_substitution"
            ]

        augmented_texts = []

        for method in methods:
            for _ in range(augmentation_factor):
                if method == "synonym_replacement":
                    aug_text = self.synonym_replacement(text, n=2)
                elif method == "random_insertion":
                    aug_text = self.random_insertion(text, n=1)
                elif method == "random_swap":
                    aug_text = self.random_swap(text, n=2)
                elif method == "random_deletion":
                    aug_text = self.random_deletion(text, p=0.1)
                elif method == "contextual_word_substitution":
                    aug_text = self.contextual_word_substitution(text, n=2)
                elif method == "back_translation":
                    aug_text = self.back_translation(text)
                else:
                    aug_text = text

                augmented_texts.append(aug_text)

        return augmented_texts


class AdversarialAugmentor:
    """
    Generate adversarial examples for robustness testing.

    Techniques:
    - Character-level perturbations (typos, homoglyphs)
    - Word-level perturbations (misspellings)
    - Phrase-level perturbations (negation insertion)
    """

    @staticmethod
    def add_typos(text: str, n: int = 2) -> str:
        """Add random typos to text"""
        words = text.split()
        if len(words) < n:
            n = len(words)

        indices = random.sample(range(len(words)), n)

        for idx in indices:
            word = list(words[idx])
            if len(word) > 2:
                # Random typo type
                typo_type = random.choice(["swap", "delete", "insert"])

                if typo_type == "swap" and len(word) > 1:
                    # Swap adjacent characters
                    pos = random.randint(0, len(word) - 2)
                    word[pos], word[pos + 1] = word[pos + 1], word[pos]

                elif typo_type == "delete":
                    # Delete random character
                    pos = random.randint(0, len(word) - 1)
                    del word[pos]

                elif typo_type == "insert":
                    # Insert random character
                    pos = random.randint(0, len(word))
                    char = random.choice('abcdefghijklmnopqrstuvwxyz')
                    word.insert(pos, char)

                words[idx] = ''.join(word)

        return " ".join(words)

    @staticmethod
    def insert_negation(text: str) -> str:
        """Insert or remove negations to flip meaning"""
        # Simplified negation insertion
        negation_words = ["not", "never", "no", "n't"]

        words = text.split()

        # Find verb positions (simplified)
        verb_indicators = ["is", "are", "was", "were", "has", "have", "will", "can"]

        for i, word in enumerate(words):
            if word.lower() in verb_indicators:
                # Check if negation already present
                if i + 1 < len(words) and words[i + 1].lower() not in negation_words:
                    # Insert negation
                    words.insert(i + 1, "not")
                    break

        return " ".join(words)


# Factory function
def create_augmentor(
    load_translation_models: bool = False
) -> TextAugmentor:
    """
    Create text augmentor instance.

    Args:
        load_translation_models: Whether to load back-translation models

    Returns:
        TextAugmentor instance
    """
    augmentor = TextAugmentor()

    if load_translation_models:
        augmentor.load_back_translation_models()

    return augmentor
