import unittest
from unittest.mock import patch, MagicMock
import os
import logging

# Disable logging for tests
logging.disable(logging.CRITICAL)

from deepsearcher.llm import Bedrock
from deepsearcher.llm.base import ChatResponse


class TestBedrock(unittest.TestCase):
    """Tests for the AWS Bedrock LLM provider."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_boto3 = MagicMock()
        self.mock_client = MagicMock()
        
        # Set up the mock module structure
        self.mock_boto3.client = MagicMock(return_value=self.mock_client)
        
        # Set up mock response
        self.mock_response = {
            "output": {
                "message": {
                    "content": [{"text": "Test response\nwith newline"}]
                }
            },
            "usage": {
                "totalTokens": 100
            }
        }
        self.mock_client.converse.return_value = self.mock_response

        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'boto3': self.mock_boto3})
        self.module_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    def test_init_default(self):
        """Test initialization with default parameters."""
        # Clear environment variables temporarily
        with patch.dict('os.environ', {}, clear=True):
            llm = Bedrock()
            # Check that client was initialized correctly
            self.mock_boto3.client.assert_called_once_with(
                "bedrock-runtime",
                region_name="us-west-2",
                aws_access_key_id=None,
                aws_secret_access_key=None,
                aws_session_token=None
            )
            
            # Check default attributes
            self.assertEqual(llm.model, "us.deepseek.r1-v1:0")
            self.assertEqual(llm.max_tokens, 20000)

    def test_init_with_aws_credentials_from_env(self):
        """Test initialization with AWS credentials from environment variables."""
        credentials = {
            "AWS_ACCESS_KEY_ID": "test_access_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret_key",
            "AWS_SESSION_TOKEN": "test_session_token"
        }
        with patch.dict(os.environ, credentials):
            llm = Bedrock()
            self.mock_boto3.client.assert_called_with(
                "bedrock-runtime",
                region_name="us-west-2",
                aws_access_key_id="test_access_key",
                aws_secret_access_key="test_secret_key",
                aws_session_token="test_session_token"
            )

    def test_init_with_aws_credentials_parameters(self):
        """Test initialization with AWS credentials as parameters."""
        with patch.dict('os.environ', {}, clear=True):
            llm = Bedrock(
                aws_access_key_id="param_access_key",
                aws_secret_access_key="param_secret_key",
                aws_session_token="param_session_token"
            )
            self.mock_boto3.client.assert_called_with(
                "bedrock-runtime",
                region_name="us-west-2",
                aws_access_key_id="param_access_key",
                aws_secret_access_key="param_secret_key",
                aws_session_token="param_session_token"
            )

    def test_init_with_custom_model_and_tokens(self):
        """Test initialization with custom model and max tokens."""
        with patch.dict('os.environ', {}, clear=True):
            llm = Bedrock(model="custom.model", max_tokens=1000)
            self.assertEqual(llm.model, "custom.model")
            self.assertEqual(llm.max_tokens, 1000)

    def test_init_with_custom_region(self):
        """Test initialization with custom region."""
        with patch.dict('os.environ', {}, clear=True):
            llm = Bedrock(region_name="us-east-1")
            self.mock_boto3.client.assert_called_with(
                "bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id=None,
                aws_secret_access_key=None,
                aws_session_token=None
            )

    def test_chat_single_message(self):
        """Test chat with a single message."""
        # Create Bedrock instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Bedrock()
            messages = [{"role": "user", "content": "Hello"}]
            response = llm.chat(messages)

            # Check that converse was called correctly
            self.mock_client.converse.assert_called_once()
            call_args = self.mock_client.converse.call_args
            self.assertEqual(call_args[1]["modelId"], "us.deepseek.r1-v1:0")
            self.assertEqual(call_args[1]["messages"], [
                {"role": "user", "content": [{"text": "Hello"}]}
            ])
            self.assertEqual(call_args[1]["inferenceConfig"], {"maxTokens": 20000})

            # Check response
            self.assertIsInstance(response, ChatResponse)
            self.assertEqual(response.content, "Test responsewith newline")
            self.assertEqual(response.total_tokens, 100)

    def test_chat_multiple_messages(self):
        """Test chat with multiple messages."""
        # Create Bedrock instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Bedrock()
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
            response = llm.chat(messages)

            # Check that converse was called correctly
            self.mock_client.converse.assert_called_once()
            call_args = self.mock_client.converse.call_args
            
            expected_messages = [
                {"role": "system", "content": [{"text": "You are a helpful assistant"}]},
                {"role": "user", "content": [{"text": "Hello"}]},
                {"role": "assistant", "content": [{"text": "Hi there!"}]},
                {"role": "user", "content": [{"text": "How are you?"}]}
            ]
            self.assertEqual(call_args[1]["messages"], expected_messages)

    def test_chat_with_error(self):
        """Test chat when an error occurs."""
        # Create Bedrock instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Bedrock()
            # Mock an error response
            self.mock_client.converse.side_effect = Exception("AWS Bedrock Error")

            messages = [{"role": "user", "content": "Hello"}]
            with self.assertRaises(Exception) as context:
                llm.chat(messages)

            self.assertEqual(str(context.exception), "AWS Bedrock Error")

    def test_chat_with_preformatted_messages(self):
        """Test chat with messages that are already in the correct format."""
        # Create Bedrock instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = Bedrock()
            messages = [
                {
                    "role": "user",
                    "content": [{"text": "Hello"}]
                }
            ]
            response = llm.chat(messages)

            # Check that the message format was preserved
            call_args = self.mock_client.converse.call_args
            self.assertEqual(call_args[1]["messages"], messages)


if __name__ == "__main__":
    unittest.main() 