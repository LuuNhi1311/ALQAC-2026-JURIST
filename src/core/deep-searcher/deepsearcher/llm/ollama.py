from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class Ollama(BaseLLM):
    """
    Ollama language model implementation.

    This class provides an interface to interact with locally hosted language models
    through the Ollama API.

    Attributes:
        model (str): The Ollama model identifier to use.
        client: The Ollama client instance.
    """

    def __init__(self, model: str = "qwq", **kwargs):
        """
        Initialize an Ollama language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "qwq".
            **kwargs: Additional keyword arguments to pass to the Ollama client.
                - base_url: Ollama API base URL. If not provided, defaults to "http://localhost:11434".
        """
        from ollama import Client

        self.model = model
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = "http://localhost:11434"
        self.client = Client(host=base_url)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the Ollama model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        completion = self.client.chat(model=self.model, messages=messages)

        return ChatResponse(
            content=completion.message.content,
            total_tokens=completion.prompt_eval_count + completion.eval_count,
        )
