import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys

class TestQdrant(unittest.TestCase):
    """Tests for the Qdrant vector database implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock modules
        self.mock_qdrant = MagicMock()
        self.mock_models = MagicMock()
        self.mock_qdrant.models = self.mock_models
        
        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {
            'qdrant_client': self.mock_qdrant,
            'qdrant_client.models': self.mock_models
        })
        self.module_patcher.start()
        
        # Import after mocking
        from deepsearcher.vector_db import Qdrant
        from deepsearcher.loader.splitter import Chunk
        from deepsearcher.vector_db.base import RetrievalResult
        
        self.Qdrant = Qdrant
        self.Chunk = Chunk
        self.RetrievalResult = RetrievalResult

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    @patch('qdrant_client.QdrantClient')
    def test_init(self, mock_client_class):
        """Test basic initialization."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        qdrant = self.Qdrant(
            location="memory",
            url="http://custom:6333",
            port=6333,
            api_key="test_key",
            default_collection="custom"
        )
        
        # Verify initialization - just check basic properties
        self.assertEqual(qdrant.default_collection, "custom")
        self.assertIsNotNone(qdrant.client)

    @patch('qdrant_client.QdrantClient')
    def test_init_collection(self, mock_client_class):
        """Test collection initialization."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        qdrant = self.Qdrant()
        
        # Test collection initialization
        d = 8
        collection = "test_collection"
        
        try:
            qdrant.init_collection(dim=d, collection=collection)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "init_collection should work")

    @patch('qdrant_client.QdrantClient')
    def test_insert_data(self, mock_client_class):
        """Test inserting data."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.upsert.return_value = None
        
        qdrant = self.Qdrant()
        
        # Create test data
        d = 8
        collection = "test_collection"
        rng = np.random.default_rng(seed=42)
        
        # Create test chunks with numpy arrays converted to lists
        chunks = [
            self.Chunk(
                embedding=rng.random(d).tolist(),  # Convert to list
                text="hello world",
                reference="test.txt",
                metadata={"key": "value1"}
            ),
            self.Chunk(
                embedding=rng.random(d).tolist(),  # Convert to list
                text="hello qdrant",
                reference="test.txt",
                metadata={"key": "value2"}
            )
        ]
        
        try:
            qdrant.insert_data(collection=collection, chunks=chunks)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "insert_data should work")

    @patch('qdrant_client.QdrantClient')
    def test_search_data(self, mock_client_class):
        """Test search functionality."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock search results
        d = 8
        rng = np.random.default_rng(seed=42)
        mock_point1 = MagicMock()
        mock_point1.vector = rng.random(d)
        mock_point1.payload = {
            "text": "hello world",
            "reference": "test.txt",
            "metadata": {"key": "value1"}
        }
        mock_point1.score = 0.95

        mock_point2 = MagicMock()
        mock_point2.vector = rng.random(d)
        mock_point2.payload = {
            "text": "hello qdrant",
            "reference": "test.txt",
            "metadata": {"key": "value2"}
        }
        mock_point2.score = 0.85

        mock_response = MagicMock()
        mock_response.points = [mock_point1, mock_point2]
        mock_client.query_points.return_value = mock_response
        
        qdrant = self.Qdrant()
        
        # Test search
        collection = "test_collection"
        query_vector = rng.random(d)
        
        try:
            results = qdrant.search_data(
                collection=collection, 
                vector=query_vector, 
                top_k=2
            )
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "search_data should work")
        if test_passed:
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
            # Verify results are RetrievalResult objects
            for result in results:
                self.assertIsInstance(result, self.RetrievalResult)

    @patch('qdrant_client.QdrantClient')
    def test_clear_collection(self, mock_client_class):
        """Test clearing collection."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.delete_collection.return_value = None
        
        qdrant = self.Qdrant()
        collection = "test_collection"
        
        try:
            qdrant.clear_db(collection=collection)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "clear_db should work")


if __name__ == "__main__":
    unittest.main() 