import unittest
from typing import List
from unittest.mock import patch, MagicMock

from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.loader.splitter import Chunk


class ConcreteEmbedding(BaseEmbedding):
    """A concrete implementation of BaseEmbedding for testing."""
    
    def __init__(self, dimension=768):
        self._dimension = dimension
    
    def embed_query(self, text: str) -> List[float]:
        """Simple implementation that returns a vector of the given dimension."""
        return [0.1] * self._dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension


class TestBaseEmbedding(unittest.TestCase):
    """Tests for the BaseEmbedding base class."""
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_query(self):
        """Test the embed_query method."""
        embedding = ConcreteEmbedding()
        result = embedding.embed_query("test text")
        self.assertEqual(len(result), 768)
        self.assertEqual(result, [0.1] * 768)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents(self):
        """Test the embed_documents method."""
        embedding = ConcreteEmbedding()
        texts = ["text 1", "text 2", "text 3"]
        results = embedding.embed_documents(texts)
        
        # Check we got the right number of embeddings
        self.assertEqual(len(results), 3)
        
        # Check each embedding
        for result in results:
            self.assertEqual(len(result), 768)
            self.assertEqual(result, [0.1] * 768)
    
    @patch('deepsearcher.embedding.base.tqdm')
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_chunks(self, mock_tqdm):
        """Test the embed_chunks method."""
        embedding = ConcreteEmbedding()
        
        # Set up mock tqdm to just return the iterable
        mock_tqdm.return_value = lambda x, desc: x
        
        # Create test chunks
        chunks = [
            Chunk(text="text 1", reference="ref1"),
            Chunk(text="text 2", reference="ref2"),
            Chunk(text="text 3", reference="ref3")
        ]
        
        # Create a spy on embed_documents
        original_embed_documents = embedding.embed_documents
        embed_documents_calls = []
        
        def mock_embed_documents(texts):
            embed_documents_calls.append(texts)
            return original_embed_documents(texts)
        
        embedding.embed_documents = mock_embed_documents
        
        # Mock tqdm to return the batch_texts directly
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Call the method
        result_chunks = embedding.embed_chunks(chunks, batch_size=2)
        
        # Verify embed_documents was called correctly
        self.assertEqual(len(embed_documents_calls), 2)  # Should be called twice with batch_size=2
        self.assertEqual(embed_documents_calls[0], ["text 1", "text 2"])
        self.assertEqual(embed_documents_calls[1], ["text 3"])
        
        # Verify chunks were updated with embeddings
        self.assertEqual(len(result_chunks), 3)
        for chunk in result_chunks:
            self.assertEqual(len(chunk.embedding), 768)
            self.assertEqual(chunk.embedding, [0.1] * 768)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_dimension_property(self):
        """Test the dimension property."""
        embedding = ConcreteEmbedding()
        self.assertEqual(embedding.dimension, 768)
        
        # Test with different dimension
        embedding = ConcreteEmbedding(dimension=1024)
        self.assertEqual(embedding.dimension, 1024)


if __name__ == "__main__":
    unittest.main() 