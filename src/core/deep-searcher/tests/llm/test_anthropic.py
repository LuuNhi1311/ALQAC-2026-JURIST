import unittest
from unittest.mock import patch, MagicMock
import os
import logging
from typing import List

# Disable logging for tests
logging.disable(logging.CRITICAL)

from deepsearcher.llm import Anthropic
from deepsearcher.llm.base import ChatResponse


class ContentItem:
    """Mock content item for Anthropic response."""
    def __init__(self, text: str):
        self.text = text


class TestAnthropic(unittest.TestCase):
    """Tests for the Anthropic LLM provider."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_anthropic = MagicMock()
        self.mock_client = MagicMock()
        
        # Set up mock response
        self.mock_response = MagicMock()
        
        # Set up response content with proper structure
        content_item = ContentItem("Test response")
        self.mock_response.content = [content_item]
        self.mock_response.usage.input_tokens = 50
        self.mock_response.usage.output_tokens = 50
        
        # Set up the mock module structure and response
        self.mock_client.messages.create.return_value = self.mock_response
        self.mock_anthropic.Anthropic.return_value = self.mock_client

        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'anthropic': self.mock_anthropic})
        self.module_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    def test_init_default(self):
        """Test initialization with default parameters."""
        # Clear environment variables temporarily
        with patch.dict('os.environ', {}, clear=True):
            llm = Anthropic()
            # Check that Anthropic client was initialized correctly
            self.mock_anthropic.Anthropic.assert_called_once_with(
                api_key=None,
                base_url=None
            )
            
            # Check default attributes
            self.assertEqual(llm.model, "claude-sonnet-4-0")
            self.assertEqual(llm.max_tokens, 8192)

    def test_init_with_api_key_from_env(self):
        """Test initialization with API key from environment variable."""
        api_key = "test_api_key_from_env"
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": api_key}):
            llm = Anthropic()
            self.mock_anthropic.Anthropic.assert_called_with(
                api_key=api_key,
                base_url=None
            )

    def test_init_with_api_key_parameter(self):
        """Test initialization with API key as parameter."""
        with patch.dict('os.environ', {}, clear=True):
            api_key = "test_api_key_param"
            llm = Anthropic(api_key=api_key)
            self.mock_anthropic.Anthropic.assert_called_with(
                api_key=api_key,
                base_url=None
            )

    def test_init_with_custom_model_and_tokens(self):
        """Test initialization with custom model and max tokens."""
        with patch.dict('os.environ', {}, clear=True):
            model = "claude-3-opus-20240229"
            max_tokens = 4096
            llm = Anthropic(model=model, max_tokens=max_tokens)
            self.assertEqual(llm.model, model)
            self.assertEqual(llm.max_tokens, max_tokens)

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        # Clear environment variables temporarily
        with patch.dict('os.environ', {}, clear=True):
            base_url = "https://custom.anthropic.api"
            llm = Anthropic(base_url=base_url)
            self.mock_anthropic.Anthropic.assert_called_with(
                api_key=None,
                base_url=base_url
            )

    def test_chat_single_message(self):
        """Test chat with a single message."""
        # Create Anthropic instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Anthropic()
            
        messages = [{"role": "user", "content": "Hello"}]
        response = llm.chat(messages)

        # Check that messages.create was called correctly
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args
        self.assertEqual(call_args[1]["model"], "claude-sonnet-4-0")
        self.assertEqual(call_args[1]["messages"], messages)
        self.assertEqual(call_args[1]["max_tokens"], 8192)

        # Check response
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)  # 50 input + 50 output

    def test_chat_multiple_messages(self):
        """Test chat with multiple messages."""
        # Create Anthropic instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Anthropic()
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        response = llm.chat(messages)

        # Check that messages.create was called correctly
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args
        self.assertEqual(call_args[1]["model"], "claude-sonnet-4-0")
        self.assertEqual(call_args[1]["messages"], messages)
        self.assertEqual(call_args[1]["max_tokens"], 8192)

        # Check response
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)  # 50 input + 50 output

    def test_chat_with_error(self):
        """Test chat when an error occurs."""
        # Create Anthropic instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Anthropic()
            
        # Mock an error response
        self.mock_client.messages.create.side_effect = Exception("Anthropic API Error")

        messages = [{"role": "user", "content": "Hello"}]
        with self.assertRaises(Exception) as context:
            llm.chat(messages)

        self.assertEqual(str(context.exception), "Anthropic API Error")


if __name__ == "__main__":
    unittest.main() 