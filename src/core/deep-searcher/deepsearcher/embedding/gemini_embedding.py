import os
from typing import List

from deepsearcher.embedding.base import BaseEmbedding

GEMINI_MODEL_DIM_MAP = {
    "text-embedding-004": 768,
    "gemini-embedding-exp-03-07": 3072,
}


class GeminiEmbedding(BaseEmbedding):
    """
    Gemini AI embedding model implementation.

    This class provides an interface to the Gemini AI embedding API, which offers
    various embedding models for text processing.

    For more information, see:
    https://ai.google.dev/api/embeddings
    """

    def __init__(self, model: str = "text-embedding-004", **kwargs):
        """
        Initialize the Gemini embedding model.

        Args:
            model (str): The model identifier to use for embeddings. Default is "text-embedding-004".
            **kwargs: Additional keyword arguments.
                - api_key (str, optional): The Gemini API key. If not provided,
                  it will be read from the GEMINI_API_KEY environment variable.
                - dimension (int, optional): The dimension of the embedding vectors.
                  If not provided, the default dimension for the model will be used.
        """
        from google import genai

        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("GEMINI_API_KEY")
        if "dimension" in kwargs:
            dimension = kwargs.pop("dimension")
        else:
            dimension = GEMINI_MODEL_DIM_MAP[model]
        self.dim = dimension
        self.model = model
        self.client = genai.Client(api_key=api_key, **kwargs)

    def embed_chunks(self, chunks, batch_size: int = 100):
        # For Gemini free level, the maximum rqeusts in one batch is 100, so set the default batch size to 100
        return super().embed_chunks(chunks, batch_size)

    def _get_dim(self):
        """
        Get the dimension of the embedding model.

        Returns:
            int: The dimension of the embedding model.
        """
        return self.dim

    def _embed_content(self, texts: List[str]):
        """
        Embed a list of content texts.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            List: A list of embeddings corresponding to the input texts.
        """
        from google.genai import types

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self._get_dim()),
        )
        return response.embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        # make sure the text is one string
        if len(text) != 1:
            text = " ".join(text)
        result = self._embed_content(text)
        # embedding = [r.values for r in result]
        embedding = result[0].values
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.
        """
        result = self._embed_content(texts)
        embeddings = [r.values for r in result]
        return embeddings

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings for the current model.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        return self.dim
