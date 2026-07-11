import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class Anthropic(BaseLLM):
    """
    Anthropic language model implementation.

    This class provides an interface to interact with Anthropic's Claude language models
    through their API.

    Attributes:
        model (str): The Anthropic model identifier to use.
        max_tokens (int): The maximum number of tokens to generate in the response.
        client: The Anthropic client instance.
    """

    def __init__(self, model: str = "claude-sonnet-4-0", max_tokens: int = 8192, **kwargs):
        """
        Initialize an Anthropic language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "claude-sonnet-4-0".
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 8192.
            **kwargs: Additional keyword arguments to pass to the Anthropic client.
                - api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY environment variable.
                - base_url: Anthropic API base URL. If not provided, uses the default Anthropic API endpoint.
        """
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = None
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the Anthropic model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return ChatResponse(
            content=message.content[0].text,
            total_tokens=message.usage.input_tokens + message.usage.output_tokens,
        )
