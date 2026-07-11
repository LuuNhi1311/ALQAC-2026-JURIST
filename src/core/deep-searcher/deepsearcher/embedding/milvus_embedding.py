from typing import List

import numpy as np

from deepsearcher.embedding.base import BaseEmbedding

MILVUS_MODEL_DIM_MAP = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-zh-v1.5": 1024,
    "BAAI/bge-base-zh-v1.5": 768,
    "BAAI/bge-small-zh-v1.5": 384,
    "GPTCache/paraphrase-albert-onnx": 768,
    "default": 768,  # 'GPTCache/paraphrase-albert-onnx',
    # see https://github.com/milvus-io/milvus-model/blob/4974e2d190169618a06359bcda040eaed73c4d0f/src/pymilvus/model/dense/onnx.py#L12
    "jina-embeddings-v3": 1024,  # required jina api key
}


class MilvusEmbedding(BaseEmbedding):
    """
    Milvus embedding model implementation.
    https://milvus.io/docs/embeddings.md

    This class provides an interface to the Milvus embedding models, which offers
    various embedding models for text processing, including BGE and Jina models.
    """

    def __init__(self, model: str = None, **kwargs) -> None:
        """
        Initialize the Milvus embedding model.

        Args:
            model (str, optional): The model identifier to use for embeddings.
                                  If None, the default model will be used.
            **kwargs: Additional keyword arguments passed to the underlying embedding function.
                - model_name (str, optional): Alternative way to specify the model.

        Raises:
            ValueError: If an unsupported model name is provided.

        Notes:
            Supported models include:
            - Default model: "GPTCache/paraphrase-albert-onnx" (768 dimensions)
            - BGE models: "BAAI/bge-*" series (various dimensions)
            - Jina models: "jina-*" series (requires Jina API key)
        """
        model_name = model
        from pymilvus import model

        if "model_name" in kwargs and (not model_name or model_name == "default"):
            model_name = kwargs.pop("model_name")

        if not model_name or model_name in [
            "default",
            "GPTCache/paraphrase-albert-onnx",
        ]:
            self.model = model.DefaultEmbeddingFunction(**kwargs)
        else:
            if model_name.startswith("jina-"):
                self.model = model.dense.JinaEmbeddingFunction(model_name, **kwargs)
            elif model_name.startswith("BAAI/"):
                self.model = model.dense.SentenceTransformerEmbeddingFunction(model_name, **kwargs)
            else:
                # Only support default model and BGE series model
                raise ValueError(f"Currently unsupported model name: {model_name}")

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        return self.model.encode_queries([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.

        Note:
            This method handles conversion from numpy arrays to lists if needed.
        """
        embeddings = self.model.encode_documents(texts)
        if isinstance(embeddings[0], np.ndarray):
            return [embedding.tolist() for embedding in embeddings]
        else:
            return embeddings

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings for the current model.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        return self.model.dim  # or MILVUS_MODEL_DIM_MAP[self.model_name]
