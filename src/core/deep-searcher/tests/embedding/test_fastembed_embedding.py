import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import logging

# Disable logging for tests
logging.disable(logging.CRITICAL)

from deepsearcher.embedding import FastEmbedEmbedding


class TestFastEmbedEmbedding(unittest.TestCase):
    """Tests for the FastEmbedEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_fastembed = MagicMock()
        self.mock_text_embedding = MagicMock()
        self.mock_fastembed.TextEmbedding = MagicMock(return_value=self.mock_text_embedding)
        
        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'fastembed': self.mock_fastembed})
        self.module_patcher.start()
        
        # Set up mock embeddings
        self.mock_embedding = np.array([0.1] * 384)  # BGE-small has 384 dimensions
        self.mock_text_embedding.query_embed.return_value = iter([self.mock_embedding])
        self.mock_text_embedding.embed.return_value = [self.mock_embedding] * 3
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_default(self):
        """Test initialization with default parameters."""
        # Create instance to test
        embedding = FastEmbedEmbedding()
        
        # Access a method to trigger lazy loading
        embedding.embed_query("test")
        
        # Check that TextEmbedding was initialized correctly
        self.mock_fastembed.TextEmbedding.assert_called_once_with(
            model_name="BAAI/bge-small-en-v1.5"
        )
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        custom_model = "custom/model-name"
        embedding = FastEmbedEmbedding(model=custom_model)
        
        # Access a method to trigger lazy loading
        embedding.embed_query("test")
        
        self.mock_fastembed.TextEmbedding.assert_called_with(
            model_name=custom_model
        )
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_kwargs(self):
        """Test initialization with additional kwargs."""
        kwargs = {"batch_size": 32, "max_length": 512}
        embedding = FastEmbedEmbedding(**kwargs)
        
        # Access a method to trigger lazy loading
        embedding.embed_query("test")
        
        self.mock_fastembed.TextEmbedding.assert_called_with(
            model_name="BAAI/bge-small-en-v1.5",
            **kwargs
        )
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_query(self):
        """Test embedding a single query."""
        # Create instance to test
        embedding = FastEmbedEmbedding()
        
        query = "test query"
        result = embedding.embed_query(query)
        
        # Check that query_embed was called correctly
        self.mock_text_embedding.query_embed.assert_called_once_with([query])
        
        # Check result
        self.assertEqual(len(result), 384)
        np.testing.assert_array_equal(result, [0.1] * 384)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        # Create instance to test
        embedding = FastEmbedEmbedding()
        
        texts = ["text 1", "text 2", "text 3"]
        results = embedding.embed_documents(texts)
        
        # Check that embed was called correctly
        self.mock_text_embedding.embed.assert_called_once_with(texts)
        
        # Check results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(len(result), 384)
            np.testing.assert_array_equal(result, [0.1] * 384)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_dimension_property(self):
        """Test the dimension property."""
        # Create instance to test
        embedding = FastEmbedEmbedding()
        
        # Mock a sample embedding
        sample_embedding = np.array([0.1] * 384)
        self.mock_text_embedding.query_embed.return_value = iter([sample_embedding])
        
        # Check dimension
        self.assertEqual(embedding.dimension, 384)
        
        # Verify that query_embed was called with sample text
        self.mock_text_embedding.query_embed.assert_called_with(["SAMPLE TEXT"])
    
    @patch.dict('os.environ', {}, clear=True)
    def test_lazy_loading(self):
        """Test that the model is loaded lazily."""
        # Create a new instance
        embedding = FastEmbedEmbedding()
        
        # Check that TextEmbedding wasn't called during initialization
        self.mock_fastembed.TextEmbedding.reset_mock()
        self.mock_fastembed.TextEmbedding.assert_not_called()
        
        # Access a method that requires the model
        embedding.embed_query("test")
        
        # Now TextEmbedding should have been called
        self.mock_fastembed.TextEmbedding.assert_called_once()


if __name__ == "__main__":
    unittest.main() 