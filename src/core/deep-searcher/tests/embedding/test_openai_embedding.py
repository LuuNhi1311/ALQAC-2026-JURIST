import unittest
import os
from unittest.mock import patch, MagicMock, ANY

from openai._types import NOT_GIVEN
from deepsearcher.embedding import OpenAIEmbedding


class TestOpenAIEmbedding(unittest.TestCase):
    """Tests for the OpenAIEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patches for OpenAI classes
        self.openai_patcher = patch('openai.OpenAI')
        self.mock_openai = self.openai_patcher.start()
        
        # Set up mock client
        self.mock_client = MagicMock()
        self.mock_openai.return_value = self.mock_client
        
        # Set up mock embeddings
        self.mock_embeddings = MagicMock()
        self.mock_client.embeddings = self.mock_embeddings
        
        # Set up mock response for embed_query
        mock_data_item = MagicMock()
        mock_data_item.embedding = [0.1] * 1536
        self.mock_response = MagicMock()
        self.mock_response.data = [mock_data_item]
        self.mock_embeddings.create.return_value = self.mock_response
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.openai_patcher.stop()
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_default(self):
        """Test initialization with default parameters."""
        # Create the embedder
        embedding = OpenAIEmbedding()
        
        # Check that OpenAI was initialized correctly
        self.mock_openai.assert_called_once_with(api_key='fake-api-key', base_url=None)
        
        # Check attributes
        self.assertEqual(embedding.model, 'text-embedding-ada-002')
        self.assertEqual(embedding.dim, 1536)
        self.assertFalse(embedding.is_azure)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_model(self):
        """Test initialization with specified model."""
        # Initialize with a different model
        embedding = OpenAIEmbedding(model='text-embedding-3-large')
        
        # Check attributes
        self.assertEqual(embedding.model, 'text-embedding-3-large')
        self.assertEqual(embedding.dim, 3072)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_model_name(self):
        """Test initialization with model_name parameter."""
        # Initialize with model_name
        embedding = OpenAIEmbedding(model_name='text-embedding-3-small')
        
        # Check attributes
        self.assertEqual(embedding.model, 'text-embedding-3-small')
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_dimension(self):
        """Test initialization with specified dimension."""
        # Initialize with custom dimension
        embedding = OpenAIEmbedding(model='text-embedding-3-small', dimension=512)
        
        # Check attributes
        self.assertEqual(embedding.dim, 512)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_api_key(self):
        """Test initialization with API key parameter."""
        # Initialize with API key
        embedding = OpenAIEmbedding(api_key='test-api-key')
        
        # Check that OpenAI was initialized with the provided API key
        self.mock_openai.assert_called_with(api_key='test-api-key', base_url=None)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_base_url(self):
        """Test initialization with base URL parameter."""
        # Initialize with base URL
        embedding = OpenAIEmbedding(base_url='https://test-openai-api.com')
        
        # Check that OpenAI was initialized with the provided base URL
        self.mock_openai.assert_called_with(api_key='fake-api-key', base_url='https://test-openai-api.com')
    
    @patch('openai.AzureOpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_azure(self, mock_azure_openai):
        """Test initialization with Azure OpenAI."""
        # Set up mock Azure client
        mock_azure_client = MagicMock()
        mock_azure_openai.return_value = mock_azure_client
        
        # Initialize with Azure endpoint
        embedding = OpenAIEmbedding(
            azure_endpoint='https://test-azure.openai.azure.com',
            api_key='test-azure-key',
            api_version='2023-05-15'
        )
        
        # Check that AzureOpenAI was initialized correctly
        mock_azure_openai.assert_called_once_with(
            api_key='test-azure-key',
            api_version='2023-05-15',
            azure_endpoint='https://test-azure.openai.azure.com'
        )
        
        # Check attributes
        self.assertEqual(embedding.model, 'text-embedding-ada-002')
        self.assertEqual(embedding.client, mock_azure_client)
        self.assertTrue(embedding.is_azure)
        self.assertEqual(embedding.deployment, 'text-embedding-ada-002')
    
    @patch('openai.AzureOpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_azure_deployment(self, mock_azure_openai):
        """Test initialization with Azure OpenAI and custom deployment."""
        # Set up mock Azure client
        mock_azure_client = MagicMock()
        mock_azure_openai.return_value = mock_azure_client
        
        # Initialize with Azure endpoint and deployment
        embedding = OpenAIEmbedding(
            azure_endpoint='https://test-azure.openai.azure.com',
            azure_deployment='test-deployment'
        )
        
        # Check attributes
        self.assertEqual(embedding.deployment, 'test-deployment')
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_get_dim(self):
        """Test the _get_dim method."""
        # Create the embedder
        embedding = OpenAIEmbedding()
        
        # For text-embedding-ada-002
        self.assertIs(embedding._get_dim(), NOT_GIVEN)
        
        # For text-embedding-3-small
        embedding = OpenAIEmbedding(model='text-embedding-3-small', dimension=512)
        self.assertEqual(embedding._get_dim(), 512)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_query(self):
        """Test embedding a single query."""
        # Create the embedder
        embedding = OpenAIEmbedding()
        
        # Create a test query
        query = "This is a test query"
        
        # Call the method
        result = embedding.embed_query(query)
        
        # Verify that create was called correctly
        self.mock_embeddings.create.assert_called_once_with(
            input=[query],
            model='text-embedding-ada-002',
            dimensions=ANY
        )
        
        # Check the result
        self.assertEqual(result, [0.1] * 1536)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_query_azure(self):
        """Test embedding a single query with Azure."""
        # Set up Azure embedding
        with patch('openai.AzureOpenAI') as mock_azure_openai:
            # Set up mock Azure client
            mock_azure_client = MagicMock()
            mock_azure_openai.return_value = mock_azure_client
            
            # Set up mock embeddings
            mock_azure_embeddings = MagicMock()
            mock_azure_client.embeddings = mock_azure_embeddings
            
            # Set up mock response
            mock_data_item = MagicMock()
            mock_data_item.embedding = [0.2] * 1536
            mock_response = MagicMock()
            mock_response.data = [mock_data_item]
            mock_azure_embeddings.create.return_value = mock_response
            
            # Initialize with Azure endpoint
            embedding = OpenAIEmbedding(
                azure_endpoint='https://test-azure.openai.azure.com',
                azure_deployment='test-deployment'
            )
            
            # Create a test query
            query = "This is a test query"
            
            # Call the method
            result = embedding.embed_query(query)
            
            # Verify that create was called correctly
            mock_azure_embeddings.create.assert_called_once_with(
                input=[query],
                model='text-embedding-ada-002'  # For Azure, this is the deployment name
            )
            
            # Check the result
            self.assertEqual(result, [0.2] * 1536)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        # Create the embedder
        embedding = OpenAIEmbedding()
        
        # Create test documents
        texts = ["text 1", "text 2", "text 3"]
        
        # Set up mock response for multiple documents
        mock_data_items = []
        for i in range(3):
            mock_data_item = MagicMock()
            mock_data_item.embedding = [0.1 * (i + 1)] * 1536
            mock_data_items.append(mock_data_item)
        
        mock_response = MagicMock()
        mock_response.data = mock_data_items
        self.mock_embeddings.create.return_value = mock_response
        
        # Call the method
        results = embedding.embed_documents(texts)
        
        # Verify that create was called correctly
        self.mock_embeddings.create.assert_called_once_with(
            input=texts,
            model='text-embedding-ada-002',
            dimensions=ANY
        )
        
        # Check the results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result, [0.1 * (i + 1)] * 1536)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'}, clear=True)
    def test_dimension_property(self):
        """Test the dimension property."""
        # Create the embedder
        embedding = OpenAIEmbedding()
        
        # For text-embedding-ada-002
        self.assertEqual(embedding.dimension, 1536)
        
        # For text-embedding-3-small
        embedding = OpenAIEmbedding(model='text-embedding-3-small', dimension=512)
        self.assertEqual(embedding.dimension, 512)
        
        # For text-embedding-3-large
        embedding = OpenAIEmbedding(model='text-embedding-3-large')
        self.assertEqual(embedding.dimension, 3072)


if __name__ == "__main__":
    unittest.main() 