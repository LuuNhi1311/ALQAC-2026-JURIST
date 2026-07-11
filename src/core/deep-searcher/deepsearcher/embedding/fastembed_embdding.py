from functools import cached_property
from typing import List

from deepsearcher.embedding.base import BaseEmbedding


class FastEmbedEmbedding(BaseEmbedding):
    """
    [FastEmbed](https://github.com/qdrant/fastembed) based vector embeddings.

    """

    def __init__(self, model="BAAI/bge-small-en-v1.5", **kwargs):
        """
        Initialize the Fastembed embedding model.

        Args:
            model (str): The name of the model to use.
                         Defaults is "BAAI/bge-small-en-v1.5".
            **kwargs: Additional keyword arguments for TextEmbedding.
        """
        try:
            import fastembed  # noqa: F401
        except ImportError as original_error:
            raise ImportError(
                "Fastembed is not installed. Install it using: pip install fastembed\n"
            ) from original_error

        self.model = model
        self._kwargs = kwargs

        self._embedding_model = None

    def _ensure_model_loaded(self):
        """
        Lazily load the embedding model when first needed.
        """
        from fastembed import TextEmbedding

        if self._embedding_model is None:
            self._embedding_model = TextEmbedding(model_name=self.model, **self._kwargs)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        self._ensure_model_loaded()

        embeddings = next(self._embedding_model.query_embed([text]))
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.
        """
        self._ensure_model_loaded()

        embeddings = [embedding.tolist() for embedding in self._embedding_model.embed(texts)]
        return embeddings

    @cached_property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        self._ensure_model_loaded()

        sample_embedding = self.embed_query("SAMPLE TEXT")

        return len(sample_embedding)
