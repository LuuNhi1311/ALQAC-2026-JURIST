import unittest
from unittest.mock import patch, MagicMock
import os
import logging

# Disable logging for tests
logging.disable(logging.CRITICAL)

from deepsearcher.llm import Gemini
from deepsearcher.llm.base import ChatResponse


class TestGemini(unittest.TestCase):
    """Tests for the Gemini LLM provider."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_genai = MagicMock()
        self.mock_client = MagicMock()
        self.mock_response = MagicMock()
        self.mock_metadata = MagicMock()

        # Set up the mock module structure
        self.mock_genai.Client = MagicMock(return_value=self.mock_client)
        
        # Set up mock response
        self.mock_response.text = "Test response"
        self.mock_metadata.total_token_count = 100
        self.mock_response.usage_metadata = self.mock_metadata
        self.mock_client.models.generate_content.return_value = self.mock_response

        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'google.genai': self.mock_genai})
        self.module_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    def test_init_default(self):
        """Test initialization with default parameters."""
        # Clear environment variables temporarily
        with patch.dict('os.environ', {}, clear=True):
            llm = Gemini()
            # Check that Client was initialized correctly
            self.mock_genai.Client.assert_called_once_with(api_key=None)
            
            # Check default model
            self.assertEqual(llm.model, "gemini-2.0-flash")

    def test_init_with_api_key_from_env(self):
        """Test initialization with API key from environment variable."""
        api_key = "test_api_key_from_env"
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            llm = Gemini()
            self.mock_genai.Client.assert_called_with(api_key=api_key)

    def test_init_with_api_key_parameter(self):
        """Test initialization with API key as parameter."""
        with patch.dict('os.environ', {}, clear=True):
            api_key = "test_api_key_param"
            llm = Gemini(api_key=api_key)
            self.mock_genai.Client.assert_called_with(api_key=api_key)

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict('os.environ', {}, clear=True):
            model = "gemini-pro"
            llm = Gemini(model=model)
            self.assertEqual(llm.model, model)

    def test_chat_single_message(self):
        """Test chat with a single message."""
        # Create Gemini instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Gemini()
            
        messages = [{"role": "user", "content": "Hello"}]
        response = llm.chat(messages)

        # Check that generate_content was called correctly
        self.mock_client.models.generate_content.assert_called_once()
        call_args = self.mock_client.models.generate_content.call_args
        self.assertEqual(call_args[1]["model"], "gemini-2.0-flash")
        self.assertEqual(call_args[1]["contents"], "Hello")

        # Check response
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)

    def test_chat_multiple_messages(self):
        """Test chat with multiple messages."""
        # Create Gemini instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Gemini()
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        response = llm.chat(messages)

        # Check that generate_content was called correctly
        self.mock_client.models.generate_content.assert_called_once()
        call_args = self.mock_client.models.generate_content.call_args
        self.assertEqual(call_args[1]["model"], "gemini-2.0-flash")
        expected_content = "You are a helpful assistant\nHello\nHi there!\nHow are you?"
        self.assertEqual(call_args[1]["contents"], expected_content)

        # Check response
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)

    def test_chat_with_error(self):
        """Test chat when an error occurs."""
        # Create Gemini instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Gemini()
            
        # Mock an error response
        self.mock_client.models.generate_content.side_effect = Exception("API Error")

        messages = [{"role": "user", "content": "Hello"}]
        with self.assertRaises(Exception) as context:
            llm.chat(messages)

        self.assertEqual(str(context.exception), "API Error")


if __name__ == "__main__":
    unittest.main() 