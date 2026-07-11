import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import warnings

# Filter out the pkg_resources deprecation warning from milvus_lite
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

from deepsearcher.vector_db import Milvus
from deepsearcher.loader.splitter import Chunk
from deepsearcher.vector_db.base import RetrievalResult


class TestMilvus(unittest.TestCase):
    """Simple tests for the Milvus vector database implementation."""

    def test_init(self):
        """Test basic initialization."""
        milvus = Milvus(
            default_collection="test_collection",
            uri="./milvus.db",
            hybrid=False
        )
        
        # Verify initialization - just check basic properties
        self.assertEqual(milvus.default_collection, "test_collection")
        self.assertFalse(milvus.hybrid)
        self.assertIsNotNone(milvus.client)

    def test_init_collection(self):
        """Test collection initialization."""
        milvus = Milvus(uri="./milvus.db")
        
        # Test collection initialization
        d = 8
        collection = "hello_deepsearcher"
        
        try:
            milvus.init_collection(dim=d, collection=collection)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "init_collection should work")

    def test_insert_data_with_retrieval_results(self):
        """Test inserting data using RetrievalResult objects."""
        milvus = Milvus(uri="./milvus.db")
        
        # Create test data
        d = 8
        collection = "hello_deepsearcher"
        rng = np.random.default_rng(seed=19530)
        
        # Create RetrievalResult objects
        test_data = [
            RetrievalResult(
                embedding=rng.random((1, d))[0],
                text="hello world",
                reference="local file: hi.txt",
                metadata={"a": 1},
            ),
            RetrievalResult(
                embedding=rng.random((1, d))[0],
                text="hello milvus",
                reference="local file: hi.txt",
                metadata={"a": 1},
            ),
        ]
        
        try:
            milvus.insert_data(collection=collection, chunks=test_data)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "insert_data should work with RetrievalResult objects")

    def test_search_data(self):
        """Test search functionality."""
        milvus = Milvus(uri="./milvus.db")
        
        # Test search
        d = 8
        collection = "hello_deepsearcher"
        rng = np.random.default_rng(seed=19530)
        query_vector = rng.random((1, d))[0]
        
        try:
            top_2 = milvus.search_data(
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
            self.assertIsInstance(top_2, list)
            # Note: In an empty collection, we might not get 2 results
            self.assertIsInstance(top_2[0], RetrievalResult) if top_2 else None

    def test_clear_collection(self):
        """Test clearing collection."""
        milvus = Milvus(uri="./milvus.db")
        
        collection = "hello_deepsearcher"
        
        try:
            milvus.clear_db(collection=collection)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "clear_db should work")

    def test_list_collections(self):
        """Test listing collections."""
        milvus = Milvus(uri="./milvus.db")
        
        try:
            collections = milvus.list_collections()
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "list_collections should work")
        if test_passed:
            self.assertIsInstance(collections, list)
            self.assertGreaterEqual(len(collections), 0)


if __name__ == "__main__":
    unittest.main() 