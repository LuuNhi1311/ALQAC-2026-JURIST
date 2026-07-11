import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class Gemini(BaseLLM):
    """
    Google Gemini language model implementation.

    This class provides an interface to interact with Google's Gemini language models
    through their API.

    API Documentation: https://ai.google.dev/gemini-api/docs/sdks

    Attributes:
        model (str): The Gemini model identifier to use.
        client: The Google Generative AI client instance.
    """

    def __init__(self, model: str = "gemini-2.0-flash", **kwargs):
        """
        Initialize a Gemini language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "gemini-2.0-flash".
            **kwargs: Additional keyword arguments to pass to the Gemini client.
                - api_key: Gemini API key. If not provided, uses GEMINI_API_KEY environment variable.
        """
        from google import genai

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the Gemini model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents="\n".join([m["content"] for m in messages]),
        )
        return ChatResponse(
            content=response.text,
            total_tokens=response.usage_metadata.total_token_count,
        )
