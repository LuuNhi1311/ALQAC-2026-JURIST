import unittest
from unittest.mock import MagicMock, patch, ANY
import os

class TestWatsonXEmbedding(unittest.TestCase):
    """Test cases for WatsonXEmbedding class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the ibm_watsonx_ai imports
        self.mock_credentials = MagicMock()
        self.mock_embeddings = MagicMock()

        # Create a mock client
        self.mock_client = MagicMock()

        # Set up mock response for embed_query
        self.mock_client.embed_query.return_value = {
            'results': [
                {'embedding': [0.1] * 768}
            ]
        }

    @patch('deepsearcher.embedding.watsonx_embedding.Embeddings')
    @patch('deepsearcher.embedding.watsonx_embedding.Credentials')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_init_with_env_vars(self, mock_credentials_class, mock_embeddings_class):
        """Test initialization with environment variables."""
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        mock_credentials_instance = MagicMock()
        mock_embeddings_instance = MagicMock()

        mock_credentials_class.return_value = mock_credentials_instance
        mock_embeddings_class.return_value = mock_embeddings_instance

        embedding = WatsonXEmbedding()

        # Check that Credentials was called with correct parameters
        mock_credentials_class.assert_called_once_with(
            url='https://test.watsonx.com',
            api_key='test-api-key'
        )

        # Check that Embeddings was called with correct parameters
        mock_embeddings_class.assert_called_once_with(
            model_id='ibm/slate-125m-english-rtrvr-v2',
            credentials=mock_credentials_instance,
            project_id='test-project-id'
        )

        # Check default model and dimension
        self.assertEqual(embedding.model, 'ibm/slate-125m-english-rtrvr-v2')
        self.assertEqual(embedding.dimension, 768)

    @patch('deepsearcher.embedding.watsonx_embedding.Embeddings')
    @patch('deepsearcher.embedding.watsonx_embedding.Credentials')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com'
    })
    def test_init_with_space_id(self, mock_credentials_class, mock_embeddings_class):
        """Test initialization with space_id instead of project_id."""
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        mock_credentials_instance = MagicMock()
        mock_embeddings_instance = MagicMock()

        mock_credentials_class.return_value = mock_credentials_instance
        mock_embeddings_class.return_value = mock_embeddings_instance

        embedding = WatsonXEmbedding(space_id='test-space-id')

        # Check that Embeddings was called with space_id
        mock_embeddings_class.assert_called_once_with(
            model_id='ibm/slate-125m-english-rtrvr-v2',
            credentials=mock_credentials_instance,
            space_id='test-space-id'
        )

    @patch('deepsearcher.embedding.watsonx_embedding.Embeddings')
    @patch('deepsearcher.embedding.watsonx_embedding.Credentials')
    def test_init_missing_api_key(self, mock_credentials_class, mock_embeddings_class):
        """Test initialization with missing API key."""
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                WatsonXEmbedding()

            self.assertIn("WATSONX_APIKEY", str(context.exception))

    @patch('deepsearcher.embedding.watsonx_embedding.Embeddings')
    @patch('deepsearcher.embedding.watsonx_embedding.Credentials')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key'
    })
    def test_init_missing_url(self, mock_credentials_class, mock_embeddings_class):
        """Test initialization with missing URL."""
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        with self.assertRaises(ValueError) as context:
            WatsonXEmbedding()

        self.assertIn("WATSONX_URL", str(context.exception))

    @patch('deepsearcher.embedding.watsonx_embedding.Embeddings')
    @patch('deepsearcher.embedding.watsonx_embedding.Credentials')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com'
    })
    def test_init_missing_project_and_space_id(self, mock_credentials_class, mock_embeddings_class):
        """Test initialization with missing both project_id and space_id."""
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        with self.assertRaises(ValueError) as context:
            WatsonXEmbedding()

        self.assertIn("WATSONX_PROJECT_ID", str(context.exception))

    @patch('deepsearcher.embedding.watsonx_embedding.Embeddings')
    @patch('deepsearcher.embedding.watsonx_embedding.Credentials')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_embed_query(self, mock_credentials_class, mock_embeddings_class):
        """Test embedding a single query."""
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        mock_credentials_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        # WatsonX embed_query returns the embedding vector directly, not wrapped in a dict
        mock_embeddings_instance.embed_query.return_value = [0.1] * 768
        mock_credentials_class.return_value = mock_credentials_instance
        mock_embeddings_class.return_value = mock_embeddings_instance

        # Create the embedder
        embedding = WatsonXEmbedding()

        # Create a test query
        query = "This is a test query"

        # Call the method
        result = embedding.embed_query(query)

        # Verify that embed_query was called correctly
        mock_embeddings_instance.embed_query.assert_called_once_with(text=query)

        # Check the result
        self.assertEqual(result, [0.1] * 768)

    @patch('deepsearcher.embedding.watsonx_embedding.Embeddings')
    @patch('deepsearcher.embedding.watsonx_embedding.Credentials')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_embed_documents(self, mock_credentials_class, mock_embeddings_class):
        """Test embedding multiple documents."""
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        mock_credentials_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        # WatsonX embed_documents returns a list of embedding vectors directly
        mock_embeddings_instance.embed_documents.return_value = [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768
        ]
        mock_credentials_class.return_value = mock_credentials_instance
        mock_embeddings_class.return_value = mock_embeddings_instance

        # Create the embedder
        embedding = WatsonXEmbedding()

        # Create test documents
        documents = ["Document 1", "Document 2", "Document 3"]

        # Call the method
        results = embedding.embed_documents(documents)

        # Verify that embed_documents was called correctly
        mock_embeddings_instance.embed_documents.assert_called_once_with(texts=documents)

        # Check the results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], [0.1] * 768)
        self.assertEqual(results[1], [0.2] * 768)
        self.assertEqual(results[2], [0.3] * 768)

    @patch('deepsearcher.embedding.watsonx_embedding.Embeddings')
    @patch('deepsearcher.embedding.watsonx_embedding.Credentials')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_dimension_property(self, mock_credentials_class, mock_embeddings_class):
        """Test the dimension property for different models."""
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        mock_credentials_instance = MagicMock()
        mock_embeddings_instance = MagicMock()

        mock_credentials_class.return_value = mock_credentials_instance
        mock_embeddings_class.return_value = mock_embeddings_instance

        # Test default model
        embedding = WatsonXEmbedding()
        self.assertEqual(embedding.dimension, 768)

        # Test different model
        embedding = WatsonXEmbedding(model='ibm/slate-30m-english-rtrvr')
        self.assertEqual(embedding.dimension, 384)

        # Test unknown model (should default to 768)
        embedding = WatsonXEmbedding(model='unknown-model')
        self.assertEqual(embedding.dimension, 768)

    @patch('deepsearcher.embedding.watsonx_embedding.Embeddings')
    @patch('deepsearcher.embedding.watsonx_embedding.Credentials')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_embed_query_error_handling(self, mock_credentials_class, mock_embeddings_class):
        """Test error handling in embed_query."""
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        mock_credentials_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_query.side_effect = Exception("API Error")

        mock_credentials_class.return_value = mock_credentials_instance
        mock_embeddings_class.return_value = mock_embeddings_instance

        # Create the embedder
        embedding = WatsonXEmbedding()

        # Test that the exception is properly wrapped
        with self.assertRaises(RuntimeError) as context:
            embedding.embed_query("test")

        self.assertIn("Error embedding query with WatsonX", str(context.exception))

    @patch('deepsearcher.embedding.watsonx_embedding.Embeddings')
    @patch('deepsearcher.embedding.watsonx_embedding.Credentials')
    @patch.dict('os.environ', {
        'WATSONX_APIKEY': 'test-api-key',
        'WATSONX_URL': 'https://test.watsonx.com',
        'WATSONX_PROJECT_ID': 'test-project-id'
    })
    def test_embed_documents_error_handling(self, mock_credentials_class, mock_embeddings_class):
        """Test error handling in embed_documents."""
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        mock_credentials_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.side_effect = Exception("API Error")

        mock_credentials_class.return_value = mock_credentials_instance
        mock_embeddings_class.return_value = mock_embeddings_instance

        # Create the embedder
        embedding = WatsonXEmbedding()

        # Test that the exception is properly wrapped
        with self.assertRaises(RuntimeError) as context:
            embedding.embed_documents(["test"])

        self.assertIn("Error embedding documents with WatsonX", str(context.exception))


if __name__ == '__main__':
    unittest.main()
