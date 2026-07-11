import logging
import os
from typing import List

from deepsearcher.embedding.base import BaseEmbedding

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import Embeddings
except ImportError:
    Credentials = None
    Embeddings = None

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class WatsonXEmbedding(BaseEmbedding):
    """
    IBM watsonx.ai embedding model implementation.

    This class provides an interface to the IBM watsonx.ai embedding API, which offers
    various embedding models for text processing. It automatically handles text truncation
    to fit within the model's token limits.

    For more information, see:
    https://www.ibm.com/products/watsonx-ai/foundation-models#ibmembedding
    """

    def __init__(self, model: str = "ibm/slate-125m-english-rtrvr-v2", **kwargs):
        """
        Initialize the watsonx.ai embedding model.

        Args:
            model (str): The model identifier to use for embeddings.
                        Default is "ibm/slate-125m-english-rtrvr-v2".
            **kwargs: Additional keyword arguments for WatsonX configuration.
                - url (str): WatsonX endpoint URL
                - project_id (str): WatsonX project ID
                - space_id (str): WatsonX deployment space ID (alternative to project_id)
                - max_tokens (int): Maximum number of tokens per text (default: 480)
                - use_tokenizer (bool): Whether to use HuggingFace tokenizer for precise counting (default: True)

        Environment Variables:
            WATSONX_APIKEY: IBM Cloud API key for authentication
            WATSONX_URL: WatsonX service endpoint URL
            WATSONX_PROJECT_ID: WatsonX project ID (if not provided in kwargs)
        """
        if Credentials is None or Embeddings is None:
            raise ImportError(
                "WatsonX embedding requires ibm-watsonx-ai. "
                "Install it with: pip install ibm-watsonx-ai"
            )

        self.model = model
        # Set max tokens to 480 to leave more room for start/end tokens and safety margin (model limit is 512)
        self.max_tokens = kwargs.pop("max_tokens", 480)
        self.use_tokenizer = kwargs.pop("use_tokenizer", True) and TRANSFORMERS_AVAILABLE

        # Initialize tokenizer if available and requested
        self.tokenizer = None
        if self.use_tokenizer:
            try:
                # Use a general-purpose tokenizer for token counting
                # BERT tokenizer is a good default for most models
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                logger.info("Using HuggingFace tokenizer for precise token counting")
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer, falling back to character-based estimation: {e}"
                )
                self.use_tokenizer = False

        # Get credentials from environment or kwargs
        api_key = kwargs.pop("api_key", os.getenv("WATSONX_APIKEY"))
        url = kwargs.pop("url", os.getenv("WATSONX_URL"))
        project_id = kwargs.pop("project_id", None)
        space_id = kwargs.pop("space_id", None)

        # Only get project_id from environment if neither project_id nor space_id were provided
        if project_id is None and space_id is None:
            project_id = os.getenv("WATSONX_PROJECT_ID")

        if not api_key:
            raise ValueError("WATSONX_APIKEY environment variable or api_key parameter is required")
        if not url:
            raise ValueError("WATSONX_URL environment variable or url parameter is required")
        if not project_id and not space_id:
            raise ValueError(
                "WATSONX_PROJECT_ID environment variable, project_id or space_id parameter is required"
            )

        # Set up credentials
        credentials = Credentials(url=url, api_key=api_key)

        # Initialize the embeddings client - prioritize space_id if provided
        if space_id:
            self.client = Embeddings(
                model_id=self.model, credentials=credentials, space_id=space_id
            )
        else:
            self.client = Embeddings(
                model_id=self.model, credentials=credentials, project_id=project_id
            )

        # Get dimension for this model
        self.dim = self._get_dim()

    def _get_dim(self):
        """
        Get the dimensionality of embeddings for the current model.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        # Common WatsonX embedding model dimensions
        model_dimensions = {
            "ibm/granite-embedding-278m-multilingual": 768,
            "ibm/granite-embedding-107m-multilingual": 384,
            "ibm/slate-125m-english-rtrvr-v2": 768,
            "ibm/slate-125m-english-rtrvr": 768,
            "ibm/slate-30m-english-rtrvr-v2": 384,
            "ibm/slate-30m-english-rtrvr": 384,
            "sentence-transformers/all-minilm-l6-v2": 384,
            "sentence-transformers/all-minilm-l12-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }

        return model_dimensions.get(self.model, 768)  # Default to 768

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.

        Args:
            text (str): The input text.

        Returns:
            int: The number of tokens.
        """
        if self.use_tokenizer and self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Fallback: rough estimation using word count
            # For most languages, token count is roughly word count * 1.3
            words = len(text.split())
            return int(words * 1.3)

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to fit within the model's token limit.

        Args:
            text (str): The input text to truncate.

        Returns:
            str: The truncated text.
        """
        if not text:
            return text

        # First check if truncation is needed
        token_count = self._count_tokens(text)
        if token_count <= self.max_tokens:
            return text

        logger.debug(
            f"Text exceeds {self.max_tokens} tokens (actual: {token_count}), truncating..."
        )

        if self.use_tokenizer and self.tokenizer:
            # Use tokenizer for precise truncation
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > self.max_tokens:
                truncated_tokens = tokens[: self.max_tokens]
                truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                # Double-check the token count after truncation
                final_count = self._count_tokens(truncated_text)
                if final_count > self.max_tokens:
                    # If still too long, be more aggressive
                    logger.warning(
                        f"Aggressive truncation needed: {final_count} -> {self.max_tokens}"
                    )
                    truncated_tokens = tokens[: self.max_tokens - 10]  # Extra safety margin
                    truncated_text = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                return truncated_text
                return text
        else:
            # Fallback: character-based truncation with word boundary respect
            # Be more conservative: 3 characters per token for safety
            max_chars = self.max_tokens * 3

            if len(text) <= max_chars:
                return text

            # Truncate and try to end at a word boundary
            truncated = text[:max_chars]

            # Find the last space to avoid cutting words in half
            last_space = truncated.rfind(" ")
            if last_space > max_chars * 0.7:  # More conservative threshold
                truncated = truncated[:last_space]

            # Double-check with token counting
            final_count = self._count_tokens(truncated)
            if final_count > self.max_tokens:
                # If still too long, be more aggressive
                logger.warning(
                    f"Fallback aggressive truncation needed: {final_count} -> {self.max_tokens}"
                )
                # Reduce by 20% and try again
                max_chars = int(max_chars * 0.8)
                truncated = text[:max_chars]
                last_space = truncated.rfind(" ")
                if last_space > max_chars * 0.7:
                    truncated = truncated[:last_space]

            return truncated

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        try:
            # Truncate text if it's too long
            truncated_text = self._truncate_text(text)
            if len(truncated_text) != len(text):
                logger.warning(
                    f"Query text was truncated from {len(text)} to {len(truncated_text)} characters"
                )

            # The embed_query method expects a single string and returns a list of floats directly
            response = self.client.embed_query(text=truncated_text)
            return response
        except Exception as e:
            raise RuntimeError(f"Error embedding query with WatsonX: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.
        """
        try:
            # Truncate all texts if they're too long. Current limit from watsonx.ai API is 512 context tokens
            truncated_texts = []
            truncation_count = 0

            for i, text in enumerate(texts):
                truncated_text = self._truncate_text(text)
                if len(truncated_text) != len(text):
                    truncation_count += 1
                    logger.debug(
                        f"Document {i} was truncated from {len(text)} to {len(truncated_text)} characters"
                    )
                truncated_texts.append(truncated_text)

            if truncation_count > 0:
                logger.warning(
                    f"Truncated {truncation_count} out of {len(texts)} documents to fit token limits"
                )

            # The embed_documents method expects a list of strings and returns a list of embedding vectors directly
            response = self.client.embed_documents(texts=truncated_texts)
            return response
        except Exception as e:
            raise RuntimeError(f"Error embedding documents with WatsonX: {str(e)}")

    def _embed_documents_individually(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents one by one, skipping problematic ones.
        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.
        """
        embeddings = []
        failed_count = 0

        for i, text in enumerate(texts):
            try:
                # Use very conservative truncation
                original_max = self.max_tokens
                self.max_tokens = 350  # Very conservative limit
                truncated_text = self._truncate_text(text)
                self.max_tokens = original_max

                # Try to embed single text using embed_query API - it returns the embedding directly
                embedding = self.client.embed_query(text=truncated_text)
                embeddings.append(embedding)
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to embed document {i}: {str(e)}")
                # Use zero vector as fallback
                zero_embedding = [0.0] * self.dimension
                embeddings.append(zero_embedding)

        if failed_count > 0:
            logger.warning(
                f"Failed to embed {failed_count} out of {len(texts)} documents. Using zero vectors as fallback."
            )
        return embeddings

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings for the current model.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        return self.dim
