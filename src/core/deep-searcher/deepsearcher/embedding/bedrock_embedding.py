import json
import os
from typing import List

from deepsearcher.embedding.base import BaseEmbedding

MODEL_ID_TITAN_TEXT_G1 = "amazon.titan-embed-text-v1"
MODEL_ID_TITAN_TEXT_V2 = "amazon.titan-embed-text-v2:0"
MODEL_ID_TITAN_MULTIMODAL_G1 = "amazon.titan-embed-image-v1"
MODEL_ID_COHERE_ENGLISH_V3 = "cohere.embed-english-v3"
MODEL_ID_COHERE_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"

BEDROCK_MODEL_DIM_MAP = {
    MODEL_ID_TITAN_TEXT_G1: 1536,
    MODEL_ID_TITAN_TEXT_V2: 1024,
    MODEL_ID_TITAN_MULTIMODAL_G1: 1024,
    MODEL_ID_COHERE_ENGLISH_V3: 1024,
    MODEL_ID_COHERE_MULTILINGUAL_V3: 1024,
}

DEFAULT_MODEL_ID = MODEL_ID_TITAN_TEXT_V2


class BedrockEmbedding(BaseEmbedding):
    """
    Amazon Bedrock embedding model implementation.

    This class provides an interface to the Amazon Bedrock embedding API, which offers
    various embedding models for text processing, including Amazon Titan and Cohere models.
    """

    def __init__(self, model: str = DEFAULT_MODEL_ID, region_name: str = "us-east-1", **kwargs):
        """
        Initialize the Amazon Bedrock embedding model.

        Args:
            model (str): The model identifier to use for embeddings.
                         Default is "amazon.titan-embed-text-v2:0".
            **kwargs: Additional keyword arguments.
                - aws_access_key_id (str, optional): AWS access key ID. If not provided,
                  it will be read from the AWS_ACCESS_KEY_ID environment variable.
                - aws_secret_access_key (str, optional): AWS secret access key. If not provided,
                  it will be read from the AWS_SECRET_ACCESS_KEY environment variable.
                - model_name (str, optional): Alternative way to specify the model.

        Notes:
            Available models:
                - 'amazon.titan-embed-text-v2:0': dimensions include 256, 512, 1024, default is 1024
                - 'amazon.titan-embed-text-v1': 1536 dimensions
                - 'amazon.titan-embed-image-v1': 1024 dimensions
                - 'cohere.embed-english-v3': 1024 dimensions
                - 'cohere.embed-multilingual-v3': 1024 dimensions
        """
        import boto3

        aws_access_key_id = kwargs.pop("aws_access_key_id", os.getenv("AWS_ACCESS_KEY_ID"))
        aws_secret_access_key = kwargs.pop(
            "aws_secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        if model in {None, DEFAULT_MODEL_ID} and "model_name" in kwargs:
            model = kwargs.pop("model_name")  # overwrites `model` with `model_name`

        self.model = model

        # TODO: initiate boto3 client
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text using Amazon Bedrock.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        response = self.client.invoke_model(
            modelId=self.model, body=json.dumps({"inputText": text})
        )
        model_response = json.loads(response["body"].read())
        embedding = model_response["embedding"]
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        This implementation calls embed_query for each text individually.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.
        """
        return [self.embed_query(text) for text in texts]

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings for the current model.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        return BEDROCK_MODEL_DIM_MAP[self.model]
