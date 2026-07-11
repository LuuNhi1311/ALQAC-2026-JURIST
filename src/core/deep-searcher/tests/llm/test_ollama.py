import unittest
from unittest.mock import patch, MagicMock
import logging
import os

# Disable logging for tests
logging.disable(logging.CRITICAL)

from deepsearcher.llm import Ollama
from deepsearcher.llm.base import ChatResponse


class TestOllama(unittest.TestCase):
    """Tests for the Ollama LLM provider."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_ollama = MagicMock()
        self.mock_client = MagicMock()
        
        # Set up the mock module structure
        self.mock_ollama.Client = MagicMock(return_value=self.mock_client)
        
        # Set up mock response
        self.mock_response = MagicMock()
        self.mock_message = MagicMock()
        
        self.mock_message.content = "Test response"
        self.mock_response.message = self.mock_message
        self.mock_response.prompt_eval_count = 50
        self.mock_response.eval_count = 50
        
        self.mock_client.chat.return_value = self.mock_response

        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'ollama': self.mock_ollama})
        self.module_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    def test_init_default(self):
        """Test initialization with default parameters."""
        # Clear environment variables temporarily
        with patch.dict('os.environ', {}, clear=True):
            llm = Ollama()
            # Check that Ollama client was initialized correctly
            self.mock_ollama.Client.assert_called_once_with(
                host="http://localhost:11434"
            )
            
            # Check default model
            self.assertEqual(llm.model, "qwq")

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict('os.environ', {}, clear=True):
            model = "llama2"
            llm = Ollama(model=model)
            self.assertEqual(llm.model, model)

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        # Clear environment variables temporarily
        with patch.dict('os.environ', {}, clear=True):
            base_url = "http://custom.ollama:11434"
            llm = Ollama(base_url=base_url)
            self.mock_ollama.Client.assert_called_with(
                host=base_url
            )

    def test_chat_single_message(self):
        """Test chat with a single message."""
        # Create Ollama instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Ollama()
            
        messages = [{"role": "user", "content": "Hello"}]
        response = llm.chat(messages)

        # Check that chat was called correctly
        self.mock_client.chat.assert_called_once_with(
            model="qwq",
            messages=messages
        )

        # Check response
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)

    def test_chat_multiple_messages(self):
        """Test chat with multiple messages."""
        # Create Ollama instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Ollama()
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        response = llm.chat(messages)

        # Check that chat was called correctly
        self.mock_client.chat.assert_called_once_with(
            model="qwq",
            messages=messages
        )

        # Check response
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)

    def test_chat_with_error(self):
        """Test chat when an error occurs."""
        # Create Ollama instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Ollama()
            
        # Mock an error response
        self.mock_client.chat.side_effect = Exception("Ollama API Error")

        messages = [{"role": "user", "content": "Hello"}]
        with self.assertRaises(Exception) as context:
            llm.chat(messages)

        self.assertEqual(str(context.exception), "Ollama API Error")


if __name__ == "__main__":
    unittest.main() 