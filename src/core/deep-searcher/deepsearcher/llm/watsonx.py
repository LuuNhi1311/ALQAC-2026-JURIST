import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
except ImportError:
    Credentials = None
    ModelInference = None
    GenTextParamsMetaNames = None


class WatsonX(BaseLLM):
    """
    IBM watsonx.ai large language model implementation.

    This class provides an interface to the IBM watsonx.ai language models,
    supporting various foundation models available on the watsonx.ai platform.

    For more information, see:
    https://www.ibm.com/products/watsonx-ai/foundation-models#ibmfm
    """

    def __init__(self, model: str = "ibm/granite-3-3-8b-instruct", **kwargs):
        """
        Initialize the WatsonX language model.
        Args:
            model (str): The model identifier to use.
                        Default is "ibm/granite-3-3-8b-instruct".
            **kwargs: Additional keyword arguments for WatsonX configuration.
                - url (str): WatsonX endpoint URL
                - project_id (str): WatsonX project ID
                - space_id (str): WatsonX deployment space ID (alternative to project_id)
                - max_new_tokens (int): Maximum number of tokens to generate
                - temperature (float): Sampling temperature (0.0 to 1.0)
                - top_p (float): Top-p sampling parameter
                - top_k (int): Top-k sampling parameter

        Environment Variables:
            WATSONX_APIKEY: IBM Cloud API key for authentication
            WATSONX_URL: WatsonX service endpoint URL
            WATSONX_PROJECT_ID: WatsonX project ID (if not provided in kwargs)
        """
        super().__init__()

        if Credentials is None or ModelInference is None or GenTextParamsMetaNames is None:
            raise ImportError(
                "WatsonX LLM requires ibm-watsonx-ai. Install it with: pip install ibm-watsonx-ai"
            )

        self.model = model

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

        # Initialize the model inference client - prioritize space_id if provided
        if space_id:
            self.client = ModelInference(
                model_id=self.model, credentials=credentials, space_id=space_id
            )
        else:
            self.client = ModelInference(
                model_id=self.model, credentials=credentials, project_id=project_id
            )

        # Set up generation parameters
        self.generation_params = {
            GenTextParamsMetaNames.MAX_NEW_TOKENS: kwargs.get("max_new_tokens", 1000),
            GenTextParamsMetaNames.TEMPERATURE: kwargs.get("temperature", 0.1),
            GenTextParamsMetaNames.TOP_P: kwargs.get("top_p", 1.0),
            GenTextParamsMetaNames.TOP_K: kwargs.get("top_k", 50),
        }

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the WatsonX model and get a response.

        Args:
            messages: A list of message dictionaries in the format
                     [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: A ChatResponse object containing the model's response.
        """
        try:
            # Convert messages to a single prompt text
            prompt = self._messages_to_prompt(messages)

            # Generate response
            response = self.client.generate_text(prompt=prompt, params=self.generation_params)

            # Extract content and token usage
            content = response

            # WatsonX doesn't always provide token counts, so we estimate
            # This is a rough estimation - actual implementation may vary
            total_tokens = len(prompt.split()) + len(content.split())

            return ChatResponse(content=content, total_tokens=total_tokens)

        except Exception as e:
            raise RuntimeError(f"Error generating response with WatsonX: {str(e)}")

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """
        Convert a list of chat messages to a single prompt string.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.

        Returns:
            str: Formatted prompt string.
        """
        prompt_parts = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)

        # Add final prompt for assistant response
        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)
