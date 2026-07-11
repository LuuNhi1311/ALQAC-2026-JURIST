import unittest
from unittest.mock import MagicMock, patch
import os

class TestWatsonX(unittest.TestCase):
    """Test cases for WatsonX class."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_init_with_env_vars(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test initialization with environment variables."""
        from deepsearcher.llm.watsonx import WatsonX

        mock_credentials_instance = MagicMock()
        mock_model_inference_instance = MagicMock()

        mock_credentials_class.return_value = mock_credentials_instance
        mock_model_inference_class.return_value = mock_model_inference_instance

        # Mock the GenTextParamsMetaNames attributes
        mock_gen_text_params_class.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_text_params_class.TEMPERATURE = 'temperature'
        mock_gen_text_params_class.TOP_P = 'top_p'
        mock_gen_text_params_class.TOP_K = 'top_k'

        llm = WatsonX()

        # Check that Credentials was called with correct parameters
        mock_credentials_class.assert_called_once_with(
            url='https://test.watsonx.com',
            api_key='test-api-key'
        )

        # Check that ModelInference was called with correct parameters
        mock_model_inference_class.assert_called_once_with(
            model_id='ibm/granite-3-3-8b-instruct',
            credentials=mock_credentials_instance,
            project_id='test-project-id'
        )

        # Check default model
        self.assertEqual(llm.model, 'ibm/granite-3-3-8b-instruct')

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com'
    })
    def test_init_with_space_id(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test initialization with space_id instead of project_id."""
        from deepsearcher.llm.watsonx import WatsonX

        mock_credentials_instance = MagicMock()
        mock_model_inference_instance = MagicMock()

        mock_credentials_class.return_value = mock_credentials_instance
        mock_model_inference_class.return_value = mock_model_inference_instance

        # Mock the GenTextParamsMetaNames attributes
        mock_gen_text_params_class.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_text_params_class.TEMPERATURE = 'temperature'
        mock_gen_text_params_class.TOP_P = 'top_p'
        mock_gen_text_params_class.TOP_K = 'top_k'

        llm = WatsonX(space_id='test-space-id')

        # Check that ModelInference was called with space_id
        mock_model_inference_class.assert_called_once_with(
            model_id='ibm/granite-3-3-8b-instruct',
            credentials=mock_credentials_instance,
            space_id='test-space-id'
        )

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_init_with_custom_model(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test initialization with custom model."""
        from deepsearcher.llm.watsonx import WatsonX

        mock_credentials_instance = MagicMock()
        mock_model_inference_instance = MagicMock()

        mock_credentials_class.return_value = mock_credentials_instance
        mock_model_inference_class.return_value = mock_model_inference_instance

        # Mock the GenTextParamsMetaNames attributes
        mock_gen_text_params_class.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_text_params_class.TEMPERATURE = 'temperature'
        mock_gen_text_params_class.TOP_P = 'top_p'
        mock_gen_text_params_class.TOP_K = 'top_k'

        llm = WatsonX(model='ibm/granite-13b-chat-v2')

        # Check that ModelInference was called with custom model
        mock_model_inference_class.assert_called_once_with(
            model_id='ibm/granite-13b-chat-v2',
            credentials=mock_credentials_instance,
            project_id='test-project-id'
        )

        self.assertEqual(llm.model, 'ibm/granite-13b-chat-v2')

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_init_with_custom_params(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test initialization with custom generation parameters."""
        from deepsearcher.llm.watsonx import WatsonX

        mock_credentials_instance = MagicMock()
        mock_model_inference_instance = MagicMock()

        mock_credentials_class.return_value = mock_credentials_instance
        mock_model_inference_class.return_value = mock_model_inference_instance

        # Mock the GenTextParamsMetaNames attributes
        mock_gen_text_params_class.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_text_params_class.TEMPERATURE = 'temperature'
        mock_gen_text_params_class.TOP_P = 'top_p'
        mock_gen_text_params_class.TOP_K = 'top_k'

        llm = WatsonX(
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            top_k=40
        )

        # Check that generation parameters were set correctly
        expected_params = {
            'max_new_tokens': 500,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40
        }
        self.assertEqual(llm.generation_params, expected_params)

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    def test_init_missing_api_key(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test initialization with missing API key."""
        from deepsearcher.llm.watsonx import WatsonX

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                WatsonX()

            self.assertIn("WATSONX_APIKEY", str(context.exception))

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key'
    })
    def test_init_missing_url(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test initialization with missing URL."""
        from deepsearcher.llm.watsonx import WatsonX

        with self.assertRaises(ValueError) as context:
            WatsonX()

        self.assertIn("WATSONX_URL", str(context.exception))

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com'
    })
    def test_init_missing_project_and_space_id(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test initialization with missing both project_id and space_id."""
        from deepsearcher.llm.watsonx import WatsonX

        with self.assertRaises(ValueError) as context:
            WatsonX()

        self.assertIn("WATSONX_PROJECT_ID", str(context.exception))

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_chat_simple_message(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test chat with a simple message."""
        from deepsearcher.llm.watsonx import WatsonX

        mock_credentials_instance = MagicMock()
        mock_model_inference_instance = MagicMock()
        mock_model_inference_instance.generate_text.return_value = "This is a test response from WatsonX."

        mock_credentials_class.return_value = mock_credentials_instance
        mock_model_inference_class.return_value = mock_model_inference_instance

        # Mock the GenTextParamsMetaNames attributes
        mock_gen_text_params_class.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_text_params_class.TEMPERATURE = 'temperature'
        mock_gen_text_params_class.TOP_P = 'top_p'
        mock_gen_text_params_class.TOP_K = 'top_k'

        llm = WatsonX()

        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]

        response = llm.chat(messages)

        # Check that generate_text was called
        mock_model_inference_instance.generate_text.assert_called_once()
        call_args = mock_model_inference_instance.generate_text.call_args

        # Check the prompt format
        expected_prompt = "Human: Hello, how are you?\n\nAssistant:"
        self.assertEqual(call_args[1]['prompt'], expected_prompt)

        # Check response
        self.assertEqual(response.content, "This is a test response from WatsonX.")
        self.assertIsInstance(response.total_tokens, int)
        self.assertGreater(response.total_tokens, 0)

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_chat_with_system_message(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test chat with system and user messages."""
        from deepsearcher.llm.watsonx import WatsonX

        mock_credentials_instance = MagicMock()
        mock_model_inference_instance = MagicMock()
        mock_model_inference_instance.generate_text.return_value = "4"

        mock_credentials_class.return_value = mock_credentials_instance
        mock_model_inference_class.return_value = mock_model_inference_instance

        # Mock the GenTextParamsMetaNames attributes
        mock_gen_text_params_class.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_text_params_class.TEMPERATURE = 'temperature'
        mock_gen_text_params_class.TOP_P = 'top_p'
        mock_gen_text_params_class.TOP_K = 'top_k'

        llm = WatsonX()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]

        response = llm.chat(messages)

        # Check that generate_text was called
        mock_model_inference_instance.generate_text.assert_called_once()
        call_args = mock_model_inference_instance.generate_text.call_args

        # Check the prompt format
        expected_prompt = "System: You are a helpful assistant.\n\nHuman: What is 2+2?\n\nAssistant:"
        self.assertEqual(call_args[1]['prompt'], expected_prompt)

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_chat_conversation_history(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test chat with conversation history."""
        from deepsearcher.llm.watsonx import WatsonX

        mock_credentials_instance = MagicMock()
        mock_model_inference_instance = MagicMock()
        mock_model_inference_instance.generate_text.return_value = "6"

        mock_credentials_class.return_value = mock_credentials_instance
        mock_model_inference_class.return_value = mock_model_inference_instance

        # Mock the GenTextParamsMetaNames attributes
        mock_gen_text_params_class.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_text_params_class.TEMPERATURE = 'temperature'
        mock_gen_text_params_class.TOP_P = 'top_p'
        mock_gen_text_params_class.TOP_K = 'top_k'

        llm = WatsonX()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"}
        ]

        response = llm.chat(messages)

        # Check that generate_text was called
        mock_model_inference_instance.generate_text.assert_called_once()
        call_args = mock_model_inference_instance.generate_text.call_args

        # Check the prompt format includes conversation history
        expected_prompt = ("System: You are a helpful assistant.\n\n"
                          "Human: What is 2+2?\n\n"
                          "Assistant: 2+2 equals 4.\n\n"
                          "Human: What about 3+3?\n\n"
                          "Assistant:")
        self.assertEqual(call_args[1]['prompt'], expected_prompt)

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_chat_error_handling(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test error handling in chat method."""
        from deepsearcher.llm.watsonx import WatsonX

        mock_credentials_instance = MagicMock()
        mock_model_inference_instance = MagicMock()
        mock_model_inference_instance.generate_text.side_effect = Exception("API Error")

        mock_credentials_class.return_value = mock_credentials_instance
        mock_model_inference_class.return_value = mock_model_inference_instance

        # Mock the GenTextParamsMetaNames attributes
        mock_gen_text_params_class.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_text_params_class.TEMPERATURE = 'temperature'
        mock_gen_text_params_class.TOP_P = 'top_p'
        mock_gen_text_params_class.TOP_K = 'top_k'

        llm = WatsonX()

        messages = [{"role": "user", "content": "Hello"}]

        # Test that the exception is properly wrapped
        with self.assertRaises(RuntimeError) as context:
            llm.chat(messages)

        self.assertIn("Error generating response with WatsonX", str(context.exception))

    @patch('deepsearcher.llm.watsonx.ModelInference')
    @patch('deepsearcher.llm.watsonx.Credentials')
    @patch('deepsearcher.llm.watsonx.GenTextParamsMetaNames')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_messages_to_prompt(self, mock_gen_text_params_class, mock_credentials_class, mock_model_inference_class):
        """Test the _messages_to_prompt method."""
        from deepsearcher.llm.watsonx import WatsonX

        mock_credentials_instance = MagicMock()
        mock_model_inference_instance = MagicMock()

        mock_credentials_class.return_value = mock_credentials_instance
        mock_model_inference_class.return_value = mock_model_inference_instance

        # Mock the GenTextParamsMetaNames attributes
        mock_gen_text_params_class.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_text_params_class.TEMPERATURE = 'temperature'
        mock_gen_text_params_class.TOP_P = 'top_p'
        mock_gen_text_params_class.TOP_K = 'top_k'

        llm = WatsonX()

        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "user", "content": "Another user message"}
        ]

        prompt = llm._messages_to_prompt(messages)

        expected_prompt = ("System: System message\n\n"
                          "Human: User message\n\n"
                          "Assistant: Assistant message\n\n"
                          "Human: Another user message\n\n"
                          "Assistant:")

        self.assertEqual(prompt, expected_prompt)


if __name__ == '__main__':
    unittest.main()
