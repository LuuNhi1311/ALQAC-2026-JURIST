import os
from typing import List

from deepsearcher.embedding.base import BaseEmbedding

VOYAGE_MODEL_DIM_MAP = {
    "voyage-3-large": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
}


class VoyageEmbedding(BaseEmbedding):
    """
    Voyage AI embedding model implementation.

    This class provides an interface to the Voyage AI embedding API, which offers
    various embedding models for text processing.

    For more information, see:
    https://docs.voyageai.com/embeddings/
    """

    def __init__(self, model="voyage-3", **kwargs):
        """
        Initialize the Voyage AI embedding model.

        Args:
            model (str): The model identifier to use for embeddings. Default is "voyage-3".
            **kwargs: Additional keyword arguments.
                - api_key (str, optional): The Voyage AI API key. If not provided,
                  it will be read from the VOYAGE_API_KEY environment variable.
                - model_name (str, optional): Alternative way to specify the model.
                - Additional parameters passed to the voyageai.Client.
        """
        if "model_name" in kwargs and (not model or model == "voyage-3"):
            model = kwargs.pop("model_name")
        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.voyageai_api_key = api_key

        import voyageai

        voyageai.api_key = self.voyageai_api_key
        self.vo = voyageai.Client(**kwargs)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.

        Note:
            For retrieval cases, this method uses "query" as the input type.
        """
        embeddings = self.vo.embed([text], model=self.model, input_type="query")
        return embeddings.embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.

        Note:
            This method uses "document" as the input type for retrieval optimization.
        """
        embeddings = self.vo.embed(texts, model=self.model, input_type="document")
        return embeddings.embeddings

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings for the current model.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        return VOYAGE_MODEL_DIM_MAP[self.model]
