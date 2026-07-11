import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class Bedrock(BaseLLM):
    """
    AWS Bedrock language model implementation.

    This class provides an interface to interact with foundation models available through
    the AWS Bedrock service via the AWS SDK.

    Attributes:
        model_id (str): The Bedrock model identifier to use.
        max_tokens (int): The maximum number of tokens to generate in the response.
        client: The Bedrock runtime client instance.
    """

    def __init__(
        self,
        model: str = "us.deepseek.r1-v1:0",
        max_tokens: int = 20000,
        region_name: str = "us-west-2",
        **kwargs,
    ):
        """
        Initialize an AWS Bedrock language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "us.deepseek.r1-v1:0".
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 2000.
            region_name (str, optional): AWS region for the Bedrock service. Defaults to "us-west-2".
            **kwargs: Additional keyword arguments to pass to the boto3 client.
                - aws_access_key_id: AWS access key. If not provided, uses AWS credentials from environment.
                - aws_secret_access_key: AWS secret key. If not provided, uses AWS credentials from environment.
                - aws_session_token: AWS session token if using temporary credentials.
        """
        import boto3

        self.model = model
        self.max_tokens = max_tokens

        # Extract AWS credentials if provided
        client_kwargs = {"region_name": region_name}

        for key in ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"]:
            if key in kwargs:
                client_kwargs[key] = kwargs.pop(key)
            else:
                client_kwargs[key] = os.getenv(f"{key.upper()}")

        # Create the Bedrock runtime client
        self.client = boto3.client("bedrock-runtime", **client_kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the Bedrock model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        # Convert messages format if needed (in case content is a string instead of a list of objects)
        formatted_messages = []
        for message in messages:
            if isinstance(message.get("content"), str):
                formatted_message = {
                    "role": message["role"],
                    "content": [{"text": message["content"]}],
                }
                formatted_messages.append(formatted_message)
            else:
                formatted_messages.append(message)

        response = self.client.converse(
            modelId=self.model,
            messages=formatted_messages,
            inferenceConfig={
                "maxTokens": self.max_tokens,
            },
        )

        text = response["output"]["message"]["content"][0]["text"]
        cleaned_text = text.replace("\n", "")

        return ChatResponse(
            content=cleaned_text,
            total_tokens=response["usage"]["totalTokens"],
        )
