import unittest
import os
from unittest.mock import patch, MagicMock

from deepsearcher.embedding import VoyageEmbedding


class TestVoyageEmbedding(unittest.TestCase):
    """Tests for the VoyageEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock module
        self.mock_voyageai = MagicMock()
        self.mock_client = MagicMock()
        
        # Set up mock response for embed
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]  # voyage-3 has 1024 dimensions
        self.mock_client.embed.return_value = mock_response
        
        # Set up the mock module
        self.mock_voyageai.Client.return_value = self.mock_client
        
        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'voyageai': self.mock_voyageai})
        self.module_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()
    
    @patch.dict('os.environ', {'VOYAGE_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_default(self):
        """Test initialization with default parameters."""
        # Create the embedder
        embedding = VoyageEmbedding()
        
        # Check that voyageai was initialized correctly
        self.mock_voyageai.Client.assert_called_once()
        
        # Check attributes
        self.assertEqual(embedding.model, 'voyage-3')
        self.assertEqual(embedding.voyageai_api_key, 'fake-api-key')
    
    @patch.dict('os.environ', {'VOYAGE_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_model(self):
        """Test initialization with specified model."""
        # Initialize with a different model
        embedding = VoyageEmbedding(model='voyage-3-lite')
        
        # Check attributes
        self.assertEqual(embedding.model, 'voyage-3-lite')
        self.assertEqual(embedding.dimension, 512)  # voyage-3-lite has 512 dimensions
    
    @patch.dict('os.environ', {'VOYAGE_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_model_name(self):
        """Test initialization with model_name parameter."""
        # Initialize with model_name
        embedding = VoyageEmbedding(model_name='voyage-3-large')
        
        # Check attributes
        self.assertEqual(embedding.model, 'voyage-3-large')
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_api_key(self):
        """Test initialization with API key parameter."""
        # Initialize with API key
        embedding = VoyageEmbedding(api_key='test-api-key')
        
        # Check that the API key was set correctly
        self.assertEqual(embedding.voyageai_api_key, 'test-api-key')
    
    @patch.dict('os.environ', {'VOYAGE_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_query(self):
        """Test embedding a single query."""
        # Create the embedder
        embedding = VoyageEmbedding()
        
        # Create a test query
        query = "This is a test query"
        
        # Call the method
        result = embedding.embed_query(query)
        
        # Verify that embed was called correctly
        self.mock_client.embed.assert_called_once_with(
            [query],
            model='voyage-3',
            input_type='query'
        )
        
        # Check the result
        self.assertEqual(result, [0.1] * 1024)
    
    @patch.dict('os.environ', {'VOYAGE_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        # Create the embedder
        embedding = VoyageEmbedding()
        
        # Create test documents
        texts = ["text 1", "text 2", "text 3"]
        
        # Set up mock response for multiple documents
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1 * (i + 1)] * 1024 for i in range(3)]
        self.mock_client.embed.return_value = mock_response
        
        # Call the method
        results = embedding.embed_documents(texts)
        
        # Verify that embed was called correctly
        self.mock_client.embed.assert_called_once_with(
            texts,
            model='voyage-3',
            input_type='document'
        )
        
        # Check the results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result, [0.1 * (i + 1)] * 1024)
    
    @patch.dict('os.environ', {'VOYAGE_API_KEY': 'fake-api-key'}, clear=True)
    def test_dimension_property(self):
        """Test the dimension property."""
        # Create the embedder
        embedding = VoyageEmbedding()
        
        # For voyage-3
        self.assertEqual(embedding.dimension, 1024)
        
        # For voyage-3-lite
        embedding = VoyageEmbedding(model='voyage-3-lite')
        self.assertEqual(embedding.dimension, 512)
        
        # For voyage-3-large
        embedding = VoyageEmbedding(model='voyage-3-large')
        self.assertEqual(embedding.dimension, 1024)


if __name__ == "__main__":
    unittest.main() 