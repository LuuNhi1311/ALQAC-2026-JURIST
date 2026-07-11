import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from deepsearcher.embedding import MilvusEmbedding


class TestMilvusEmbedding(unittest.TestCase):
    """Tests for the MilvusEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_pymilvus = MagicMock()
        self.mock_model = MagicMock()
        self.mock_default_embedding = MagicMock()
        self.mock_jina_embedding = MagicMock()
        self.mock_sentence_transformer = MagicMock()
        
        # Set up the mock module structure
        self.mock_pymilvus.model = self.mock_model
        self.mock_model.DefaultEmbeddingFunction = MagicMock(return_value=self.mock_default_embedding)
        self.mock_model.dense = MagicMock()
        self.mock_model.dense.JinaEmbeddingFunction = MagicMock(return_value=self.mock_jina_embedding)
        self.mock_model.dense.SentenceTransformerEmbeddingFunction = MagicMock(return_value=self.mock_sentence_transformer)
        
        # Set up default dimensions and responses
        self.mock_default_embedding.dim = 768
        self.mock_jina_embedding.dim = 1024
        self.mock_sentence_transformer.dim = 1024
        
        # Set up mock responses for encoding
        self.mock_default_embedding.encode_queries.return_value = [np.array([0.1] * 768)]
        self.mock_default_embedding.encode_documents.return_value = [np.array([0.1] * 768)]
        
        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'pymilvus': self.mock_pymilvus})
        self.module_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_default(self):
        """Test initialization with default parameters."""
        embedding = MilvusEmbedding()
        
        # Check that default model was initialized
        self.mock_model.DefaultEmbeddingFunction.assert_called_once()
        self.assertEqual(embedding.model, self.mock_default_embedding)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_jina_model(self):
        """Test initialization with Jina model."""
        embedding = MilvusEmbedding(model='jina-embeddings-v3')
        
        # Check that Jina model was initialized
        self.mock_model.dense.JinaEmbeddingFunction.assert_called_once_with('jina-embeddings-v3')
        self.assertEqual(embedding.model, self.mock_jina_embedding)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_bge_model(self):
        """Test initialization with BGE model."""
        embedding = MilvusEmbedding(model='BAAI/bge-large-en-v1.5')
        
        # Check that SentenceTransformer model was initialized
        self.mock_model.dense.SentenceTransformerEmbeddingFunction.assert_called_once_with('BAAI/bge-large-en-v1.5')
        self.assertEqual(embedding.model, self.mock_sentence_transformer)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_invalid_model(self):
        """Test initialization with invalid model raises error."""
        with self.assertRaises(ValueError):
            MilvusEmbedding(model='invalid-model')
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_query(self):
        """Test embedding a single query."""
        embedding = MilvusEmbedding()
        query = "This is a test query"
        
        result = embedding.embed_query(query)
        
        # Check that encode_queries was called correctly
        self.mock_default_embedding.encode_queries.assert_called_once_with([query])
        
        # Convert numpy array to list for comparison
        expected = [0.1] * 768
        np.testing.assert_array_almost_equal(result, expected)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        embedding = MilvusEmbedding()
        texts = ["text 1", "text 2", "text 3"]
        
        # Set up mock response for multiple documents
        mock_embeddings = [np.array([0.1 * (i + 1)] * 768) for i in range(3)]
        self.mock_default_embedding.encode_documents.return_value = mock_embeddings
        
        results = embedding.embed_documents(texts)
        
        # Check that encode_documents was called correctly
        self.mock_default_embedding.encode_documents.assert_called_once_with(texts)
        
        # Check the results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            expected = [0.1 * (i + 1)] * 768
            np.testing.assert_array_almost_equal(result, expected)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_dimension_property(self):
        """Test the dimension property."""
        # For default model
        embedding = MilvusEmbedding()
        self.assertEqual(embedding.dimension, 768)
        
        # For Jina model
        embedding = MilvusEmbedding(model='jina-embeddings-v3')
        self.assertEqual(embedding.dimension, 1024)
        
        # For BGE model
        embedding = MilvusEmbedding(model='BAAI/bge-large-en-v1.5')
        self.assertEqual(embedding.dimension, 1024)


if __name__ == "__main__":
    unittest.main() 