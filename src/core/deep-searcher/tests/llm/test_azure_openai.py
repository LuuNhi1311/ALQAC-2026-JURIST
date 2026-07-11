import unittest
from unittest.mock import patch, MagicMock
import os
import logging

# Disable logging for tests
logging.disable(logging.CRITICAL)

from deepsearcher.llm import AzureOpenAI
from deepsearcher.llm.base import ChatResponse


class TestAzureOpenAI(unittest.TestCase):
    """Tests for the Azure OpenAI LLM provider."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_openai = MagicMock()
        self.mock_client = MagicMock()
        self.mock_chat = MagicMock()
        self.mock_completions = MagicMock()
        
        # Set up the mock module structure
        self.mock_openai.AzureOpenAI = MagicMock(return_value=self.mock_client)
        self.mock_client.chat = self.mock_chat
        self.mock_chat.completions = self.mock_completions
        
        # Set up mock response
        self.mock_response = MagicMock()
        self.mock_choice = MagicMock()
        self.mock_message = MagicMock()
        self.mock_usage = MagicMock()
        
        self.mock_message.content = "Test response"
        self.mock_choice.message = self.mock_message
        self.mock_usage.total_tokens = 100
        
        self.mock_response.choices = [self.mock_choice]
        self.mock_response.usage = self.mock_usage
        self.mock_completions.create.return_value = self.mock_response

        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'openai': self.mock_openai})
        self.module_patcher.start()

        # Test parameters
        self.test_model = "gpt-4"
        self.test_endpoint = "https://test.openai.azure.com"
        self.test_api_key = "test_api_key"
        self.test_api_version = "2024-02-15"

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    def test_init_with_parameters(self):
        """Test initialization with explicit parameters."""
        # Clear environment variables temporarily
        with patch.dict('os.environ', {}, clear=True):
            llm = AzureOpenAI(
                model=self.test_model,
                azure_endpoint=self.test_endpoint,
                api_key=self.test_api_key,
                api_version=self.test_api_version
            )
            # Check that Azure OpenAI client was initialized correctly
            self.mock_openai.AzureOpenAI.assert_called_once_with(
                azure_endpoint=self.test_endpoint,
                api_key=self.test_api_key,
                api_version=self.test_api_version
            )
            
            # Check model attribute
            self.assertEqual(llm.model, self.test_model)

    def test_init_with_env_variables(self):
        """Test initialization with environment variables."""
        env_endpoint = "https://env.openai.azure.com"
        env_api_key = "env_api_key"
        
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": env_endpoint,
            "AZURE_OPENAI_KEY": env_api_key
        }):
            llm = AzureOpenAI(model=self.test_model)
            self.mock_openai.AzureOpenAI.assert_called_with(
                azure_endpoint=env_endpoint,
                api_key=env_api_key,
                api_version=None
            )

    def test_chat_single_message(self):
        """Test chat with a single message."""
        # Create Azure OpenAI instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = AzureOpenAI(
                model=self.test_model,
                azure_endpoint=self.test_endpoint,
                api_key=self.test_api_key,
                api_version=self.test_api_version
            )
            
        messages = [{"role": "user", "content": "Hello"}]
        response = llm.chat(messages)

        # Check that completions.create was called correctly
        self.mock_completions.create.assert_called_once()
        call_args = self.mock_completions.create.call_args
        self.assertEqual(call_args[1]["model"], self.test_model)
        self.assertEqual(call_args[1]["messages"], messages)

        # Check response
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)

    def test_chat_multiple_messages(self):
        """Test chat with multiple messages."""
        # Create Azure OpenAI instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = AzureOpenAI(
                model=self.test_model,
                azure_endpoint=self.test_endpoint,
                api_key=self.test_api_key,
                api_version=self.test_api_version
            )
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        response = llm.chat(messages)

        # Check that completions.create was called correctly
        self.mock_completions.create.assert_called_once()
        call_args = self.mock_completions.create.call_args
        self.assertEqual(call_args[1]["model"], self.test_model)
        self.assertEqual(call_args[1]["messages"], messages)

        # Check response
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)

    def test_chat_with_error(self):
        """Test chat when an error occurs."""
        # Create Azure OpenAI instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = AzureOpenAI(
                model=self.test_model,
                azure_endpoint=self.test_endpoint,
                api_key=self.test_api_key,
                api_version=self.test_api_version
            )
            
        # Mock an error response
        self.mock_completions.create.side_effect = Exception("Azure OpenAI API Error")

        messages = [{"role": "user", "content": "Hello"}]
        with self.assertRaises(Exception) as context:
            llm.chat(messages)

        self.assertEqual(str(context.exception), "Azure OpenAI API Error")


if __name__ == "__main__":
    unittest.main() 