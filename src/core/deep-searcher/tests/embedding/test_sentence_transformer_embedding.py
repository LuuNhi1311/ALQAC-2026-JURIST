import unittest
import sys
import logging
from unittest.mock import patch, MagicMock

# Disable logging for tests
logging.disable(logging.CRITICAL)

from deepsearcher.embedding import SentenceTransformerEmbedding


class TestSentenceTransformerEmbedding(unittest.TestCase):
    """Tests for the SentenceTransformerEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock module for sentence_transformers
        mock_st_module = MagicMock()
        
        # Create mock SentenceTransformer class
        self.mock_sentence_transformer = MagicMock()
        mock_st_module.SentenceTransformer = self.mock_sentence_transformer
        
        # Add the mock module to sys.modules
        self.module_patcher = patch.dict('sys.modules', {'sentence_transformers': mock_st_module})
        self.module_patcher.start()
        
        # Set up mock instance
        self.mock_model = MagicMock()
        self.mock_sentence_transformer.return_value = self.mock_model
        
        # Configure mock encode method
        mock_embedding = [[0.1, 0.2, 0.3] * 341 + [0.4]]  # 1024 dimensions
        self.mock_model.encode.return_value = MagicMock()
        self.mock_model.encode.return_value.tolist.return_value = mock_embedding
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init(self):
        """Test initialization."""
        # Create instance to test
        embedding = SentenceTransformerEmbedding(model="BAAI/bge-m3")
        
        # Check that SentenceTransformer was called with the right model
        self.mock_sentence_transformer.assert_called_once_with("BAAI/bge-m3")
        
        # Check that model and client were set correctly
        self.assertEqual(embedding.model, "BAAI/bge-m3")
        self.assertEqual(embedding.client, self.mock_model)
        
        # Check batch size default
        self.assertEqual(embedding.batch_size, 32)
        
        # Test with model_name parameter
        self.mock_sentence_transformer.reset_mock()
        embedding = SentenceTransformerEmbedding(model_name="BAAI/bge-large-zh-v1.5")
        self.mock_sentence_transformer.assert_called_once_with("BAAI/bge-large-zh-v1.5")
        self.assertEqual(embedding.model, "BAAI/bge-large-zh-v1.5")
        
        # Test with custom batch size
        self.mock_sentence_transformer.reset_mock()
        embedding = SentenceTransformerEmbedding(batch_size=64)
        self.assertEqual(embedding.batch_size, 64)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_query(self):
        """Test embedding a single query."""
        # Create instance to test
        embedding = SentenceTransformerEmbedding(model="BAAI/bge-m3")
        
        # Mock the encode response for a single query
        single_embedding = [0.1, 0.2, 0.3] * 341 + [0.4]  # 1024 dimensions
        self.mock_model.encode.return_value = MagicMock()
        self.mock_model.encode.return_value.tolist.return_value = [single_embedding]
        
        # Call the method
        result = embedding.embed_query("test query")
        
        # Verify encode was called correctly
        self.mock_model.encode.assert_called_once_with("test query")
        
        # Check the result
        self.assertEqual(len(result), 1024)
        self.assertEqual(result, single_embedding)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents_small_batch(self):
        """Test embedding documents with a small batch (less than batch size)."""
        # Create instance to test
        embedding = SentenceTransformerEmbedding(model="BAAI/bge-m3")
        
        # Mock the encode response for documents
        batch_embeddings = [
            [0.1, 0.2, 0.3] * 341 + [0.4],  # 1024 dimensions
            [0.4, 0.5, 0.6] * 341 + [0.7],
            [0.7, 0.8, 0.9] * 341 + [0.1]
        ]
        self.mock_model.encode.return_value = MagicMock()
        self.mock_model.encode.return_value.tolist.return_value = batch_embeddings
        
        # Create test texts
        texts = ["text 1", "text 2", "text 3"]
        
        # Call the method
        results = embedding.embed_documents(texts)
        
        # Verify encode was called correctly
        self.mock_model.encode.assert_called_once_with(texts)
        
        # Check the results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(len(result), 1024)
            self.assertEqual(result, batch_embeddings[i])
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents_large_batch(self):
        """Test embedding documents with a large batch (more than batch size)."""
        # Create instance to test with small batch size
        embedding = SentenceTransformerEmbedding(model="BAAI/bge-m3", batch_size=2)
        
        # Mock the encode response for the first batch
        batch1_embeddings = [
            [0.1, 0.2, 0.3] * 341 + [0.4],  # 1024 dimensions
            [0.4, 0.5, 0.6] * 341 + [0.7]
        ]
        # Mock the encode response for the second batch
        batch2_embeddings = [
            [0.7, 0.8, 0.9] * 341 + [0.1]
        ]
        
        # Set up the mock to return different values on each call
        self.mock_model.encode.side_effect = [
            MagicMock(tolist=lambda: batch1_embeddings),
            MagicMock(tolist=lambda: batch2_embeddings)
        ]
        
        # Create test texts
        texts = ["text 1", "text 2", "text 3"]
        
        # Call the method
        results = embedding.embed_documents(texts)
        
        # Verify encode was called twice with the right batches
        self.assertEqual(self.mock_model.encode.call_count, 2)
        self.mock_model.encode.assert_any_call(["text 1", "text 2"])
        self.mock_model.encode.assert_any_call(["text 3"])
        
        # Check the results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], batch1_embeddings[0])
        self.assertEqual(results[1], batch1_embeddings[1])
        self.assertEqual(results[2], batch2_embeddings[0])
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents_no_batching(self):
        """Test embedding documents with batching disabled."""
        # Create instance to test with batching disabled
        embedding = SentenceTransformerEmbedding(model="BAAI/bge-m3", batch_size=0)
        
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
        embedding = SentenceTransformerEmbedding(model="BAAI/bge-m3")
        
        # Check dimension for BAAI/bge-m3
        self.assertEqual(embedding.dimension, 1024)
        
        # Test with different models
        self.mock_sentence_transformer.reset_mock()
        embedding = SentenceTransformerEmbedding(model="BAAI/bge-large-zh-v1.5")
        self.assertEqual(embedding.dimension, 1024)
        
        self.mock_sentence_transformer.reset_mock()
        embedding = SentenceTransformerEmbedding(model="BAAI/bge-large-en-v1.5")
        self.assertEqual(embedding.dimension, 1024)


if __name__ == "__main__":
    unittest.main() 