import unittest
import sys
from unittest.mock import patch, MagicMock

from deepsearcher.embedding import OllamaEmbedding


class TestOllamaEmbedding(unittest.TestCase):
    """Tests for the OllamaEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock module for ollama
        mock_ollama_module = MagicMock()
        
        # Create mock Client class
        self.mock_ollama_client = MagicMock()
        mock_ollama_module.Client = self.mock_ollama_client
        
        # Add the mock module to sys.modules
        self.module_patcher = patch.dict('sys.modules', {'ollama': mock_ollama_module})
        self.module_patcher.start()
        
        # Set up mock client instance
        self.mock_client = MagicMock()
        self.mock_ollama_client.return_value = self.mock_client
        
        # Configure mock embed method
        self.mock_client.embed.return_value = {"embeddings": [[0.1] * 1024]}
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_default(self):
        """Test initialization with default parameters."""
        # Create instance to test
        embedding = OllamaEmbedding(model="bge-m3")
        
        # Check that Client was initialized correctly
        self.mock_ollama_client.assert_called_once_with(host="http://localhost:11434/")
        
        # Check instance attributes
        self.assertEqual(embedding.model, "bge-m3")
        self.assertEqual(embedding.dim, 1024)
        self.assertEqual(embedding.batch_size, 32)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_base_url(self):
        """Test initialization with custom base URL."""
        # Reset mock
        self.mock_ollama_client.reset_mock()
        
        # Create embedding with custom base URL
        embedding = OllamaEmbedding(base_url="http://custom-ollama-server:11434/")
        
        # Check that Client was initialized with custom base URL
        self.mock_ollama_client.assert_called_with(host="http://custom-ollama-server:11434/")
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_model_name(self):
        """Test initialization with model_name parameter."""
        # Reset mock
        self.mock_ollama_client.reset_mock()
        
        # Create embedding with model_name
        embedding = OllamaEmbedding(model_name="mxbai-embed-large")
        
        # Check model attribute
        self.assertEqual(embedding.model, "mxbai-embed-large")
        # Check dimension is set correctly based on model
        self.assertEqual(embedding.dim, 768)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_dimension(self):
        """Test initialization with custom dimension."""
        # Reset mock
        self.mock_ollama_client.reset_mock()
        
        # Create embedding with custom dimension
        embedding = OllamaEmbedding(dimension=512)
        
        # Check dimension attribute
        self.assertEqual(embedding.dim, 512)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_query(self):
        """Test embedding a single query."""
        # Create instance to test
        embedding = OllamaEmbedding(model="bge-m3")
        
        # Set up mock response
        self.mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3] * 341 + [0.4]]}  # 1024 dimensions
        
        # Call the method
        result = embedding.embed_query("test query")
        
        # Verify embed was called correctly
        self.mock_client.embed.assert_called_once_with(model="bge-m3", input="test query")
        
        # Check the result
        self.assertEqual(len(result), 1024)
        self.assertEqual(result, [0.1, 0.2, 0.3] * 341 + [0.4])
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents_small_batch(self):
        """Test embedding documents with a small batch (less than batch size)."""
        # Create instance to test
        embedding = OllamaEmbedding(model="bge-m3")
        
        # Set up mock response for multiple documents
        mock_embeddings = [
            [0.1, 0.2, 0.3] * 341 + [0.4],  # 1024 dimensions
            [0.4, 0.5, 0.6] * 341 + [0.7],
            [0.7, 0.8, 0.9] * 341 + [0.1]
        ]
        self.mock_client.embed.return_value = {"embeddings": mock_embeddings}
        
        # Create test texts
        texts = ["text 1", "text 2", "text 3"]
        
        # Call the method
        results = embedding.embed_documents(texts)
        
        # Verify embed was called correctly
        self.mock_client.embed.assert_called_once_with(model="bge-m3", input=texts)
        
        # Check the results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(len(result), 1024)
            self.assertEqual(result, mock_embeddings[i])
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents_large_batch(self):
        """Test embedding documents with a large batch (more than batch size)."""
        # Create instance to test
        embedding = OllamaEmbedding(model="bge-m3")
        
        # Set a smaller batch size for testing
        embedding.batch_size = 2
        
        # Set up mock responses for batches
        batch1_embeddings = [
            [0.1, 0.2, 0.3] * 341 + [0.4],  # 1024 dimensions
            [0.4, 0.5, 0.6] * 341 + [0.7]
        ]
        batch2_embeddings = [
            [0.7, 0.8, 0.9] * 341 + [0.1]
        ]
        
        # Configure mock to return different responses for each call
        self.mock_client.embed.side_effect = [
            {"embeddings": batch1_embeddings},
            {"embeddings": batch2_embeddings}
        ]
        
        # Create test texts
        texts = ["text 1", "text 2", "text 3"]
        
        # Call the method
        results = embedding.embed_documents(texts)
        
        # Verify embed was called twice with the right batches
        self.assertEqual(self.mock_client.embed.call_count, 2)
        self.mock_client.embed.assert_any_call(model="bge-m3", input=["text 1", "text 2"])
        self.mock_client.embed.assert_any_call(model="bge-m3", input=["text 3"])
        
        # Check the results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], batch1_embeddings[0])
        self.assertEqual(results[1], batch1_embeddings[1])
        self.assertEqual(results[2], batch2_embeddings[0])
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents_no_batching(self):
        """Test embedding documents with batching disabled."""
        # Create instance to test
        embedding = OllamaEmbedding(model="bge-m3")
        
        # Disable batching
        embedding.batch_size = 0
        
        # Mock the embed_query method
        original_embed_query = embedding.embed_query
        embed_query_calls = []
        
        def mock_embed_query(text):
            embed_query_calls.append(text)
            return [0.1] * 1024  # Return a simple mock embedding
        
        embedding.embed_query = mock_embed_query
        
        # Create test texts
        texts = ["text 1", "text 2", "text 3"]
        
        # Call the method
        results = embedding.embed_documents(texts)
        
        # Check that embed_query was called for each text
        self.assertEqual(len(embed_query_calls), 3)
        self.assertEqual(embed_query_calls, texts)
        
        # Check the results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(len(result), 1024)
            self.assertEqual(result, [0.1] * 1024)
        
        # Restore original method
        embedding.embed_query = original_embed_query
    
    @patch.dict('os.environ', {}, clear=True)
    def test_dimension_property(self):
        """Test the dimension property."""
        # Create instance to test
        embedding = OllamaEmbedding(model="bge-m3")
        
        # Check dimension for bge-m3
        self.assertEqual(embedding.dimension, 1024)
        
        # Test with different models
        self.mock_ollama_client.reset_mock()
        embedding = OllamaEmbedding(model="mxbai-embed-large")
        self.assertEqual(embedding.dimension, 768)
        
        self.mock_ollama_client.reset_mock()
        embedding = OllamaEmbedding(model="nomic-embed-text")
        self.assertEqual(embedding.dimension, 768)
        
        # Test with custom dimension
        self.mock_ollama_client.reset_mock()
        embedding = OllamaEmbedding(dimension=512)
        self.assertEqual(embedding.dimension, 512)


if __name__ == "__main__":
    unittest.main() 