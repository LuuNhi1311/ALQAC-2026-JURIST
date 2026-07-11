import unittest
import os
from unittest.mock import patch, MagicMock

from deepsearcher.embedding import GLMEmbedding


class TestGLMEmbedding(unittest.TestCase):
    """Tests for the GLMEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_zhipuai = MagicMock()
        self.mock_client = MagicMock()
        self.mock_embeddings = MagicMock()
        
        # Set up mock response
        mock_data_item = MagicMock()
        mock_data_item.embedding = [0.1] * 2048  # embedding-3 has 2048 dimensions
        mock_response = MagicMock()
        mock_response.data = [mock_data_item]
        self.mock_embeddings.create.return_value = mock_response
        
        # Set up the mock module structure
        self.mock_zhipuai.ZhipuAI.return_value = self.mock_client
        self.mock_client.embeddings = self.mock_embeddings
        
        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'zhipuai': self.mock_zhipuai})
        self.module_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()
    
    @patch.dict('os.environ', {'GLM_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_default(self):
        """Test initialization with default parameters."""
        # Create the embedder
        embedding = GLMEmbedding()
        
        # Check that ZhipuAI was initialized correctly
        self.mock_zhipuai.ZhipuAI.assert_called_once_with(
            api_key='fake-api-key',
            base_url='https://open.bigmodel.cn/api/paas/v4/'
        )
        
        # Check attributes
        self.assertEqual(embedding.model, 'embedding-3')
        self.assertEqual(embedding.client, self.mock_client)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_api_key(self):
        """Test initialization with API key parameter."""
        # Initialize with API key
        embedding = GLMEmbedding(api_key='test-api-key')
        
        # Check that ZhipuAI was initialized with the provided API key
        self.mock_zhipuai.ZhipuAI.assert_called_with(
            api_key='test-api-key',
            base_url='https://open.bigmodel.cn/api/paas/v4/'
        )
    
    @patch.dict('os.environ', {'GLM_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_base_url(self):
        """Test initialization with base URL parameter."""
        # Initialize with base URL
        embedding = GLMEmbedding(base_url='https://custom-api.example.com')
        
        # Check that ZhipuAI was initialized with the provided base URL
        self.mock_zhipuai.ZhipuAI.assert_called_with(
            api_key='fake-api-key',
            base_url='https://custom-api.example.com'
        )
    
    @patch.dict('os.environ', {'GLM_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_query(self):
        """Test embedding a single query."""
        # Create the embedder
        embedding = GLMEmbedding()
        
        # Create a test query
        query = "This is a test query"
        
        # Call the method
        result = embedding.embed_query(query)
        
        # Verify that create was called correctly
        self.mock_embeddings.create.assert_called_once_with(
            input=[query],
            model='embedding-3'
        )
        
        # Check the result
        self.assertEqual(result, [0.1] * 2048)
    
    @patch.dict('os.environ', {'GLM_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        # Create the embedder
        embedding = GLMEmbedding()
        
        # Create test documents
        texts = ["text 1", "text 2", "text 3"]
        
        # Set up mock response for multiple documents
        mock_data_items = []
        for i in range(3):
            mock_data_item = MagicMock()
            mock_data_item.embedding = [0.1 * (i + 1)] * 2048
            mock_data_items.append(mock_data_item)
        
        mock_response = MagicMock()
        mock_response.data = mock_data_items
        self.mock_embeddings.create.return_value = mock_response
        
        # Call the method
        results = embedding.embed_documents(texts)
        
        # Verify that create was called correctly
        self.mock_embeddings.create.assert_called_once_with(
            input=texts,
            model='embedding-3'
        )
        
        # Check the results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result, [0.1 * (i + 1)] * 2048)
    
    @patch.dict('os.environ', {'GLM_API_KEY': 'fake-api-key'}, clear=True)
    def test_dimension_property(self):
        """Test the dimension property."""
        # Create the embedder
        embedding = GLMEmbedding()
        
        # For embedding-3
        self.assertEqual(embedding.dimension, 2048)


if __name__ == "__main__":
    unittest.main() 