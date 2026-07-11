import unittest
import numpy as np
from typing import List

from deepsearcher.vector_db.base import (
    RetrievalResult,
    deduplicate_results,
    CollectionInfo,
    BaseVectorDB,
)
from deepsearcher.loader.splitter import Chunk


class TestRetrievalResult(unittest.TestCase):
    """Tests for the RetrievalResult class."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedding = np.array([0.1, 0.2, 0.3])
        self.text = "Test text"
        self.reference = "test.txt"
        self.metadata = {"key": "value"}
        self.score = 0.95

    def test_init(self):
        """Test initialization of RetrievalResult."""
        result = RetrievalResult(
            embedding=self.embedding,
            text=self.text,
            reference=self.reference,
            metadata=self.metadata,
            score=self.score,
        )

        self.assertTrue(np.array_equal(result.embedding, self.embedding))
        self.assertEqual(result.text, self.text)
        self.assertEqual(result.reference, self.reference)
        self.assertEqual(result.metadata, self.metadata)
        self.assertEqual(result.score, self.score)

    def test_init_default_score(self):
        """Test initialization of RetrievalResult with default score."""
        result = RetrievalResult(
            embedding=self.embedding,
            text=self.text,
            reference=self.reference,
            metadata=self.metadata,
        )
        self.assertEqual(result.score, 0.0)

    def test_repr(self):
        """Test string representation of RetrievalResult."""
        result = RetrievalResult(
            embedding=self.embedding,
            text=self.text,
            reference=self.reference,
            metadata=self.metadata,
            score=self.score,
        )
        expected = f"RetrievalResult(score={self.score}, embedding={self.embedding}, text={self.text}, reference={self.reference}), metadata={self.metadata}"
        self.assertEqual(repr(result), expected)


class TestDeduplicateResults(unittest.TestCase):
    """Tests for the deduplicate_results function."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedding1 = np.array([0.1, 0.2, 0.3])
        self.embedding2 = np.array([0.4, 0.5, 0.6])
        self.text1 = "Text 1"
        self.text2 = "Text 2"
        self.reference = "test.txt"
        self.metadata = {"key": "value"}

    def test_no_duplicates(self):
        """Test deduplication with no duplicate results."""
        results = [
            RetrievalResult(self.embedding1, self.text1, self.reference, self.metadata),
            RetrievalResult(self.embedding2, self.text2, self.reference, self.metadata),
        ]
        deduplicated = deduplicate_results(results)
        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(deduplicated, results)

    def test_with_duplicates(self):
        """Test deduplication with duplicate results."""
        results = [
            RetrievalResult(self.embedding1, self.text1, self.reference, self.metadata),
            RetrievalResult(self.embedding2, self.text2, self.reference, self.metadata),
            RetrievalResult(self.embedding1, self.text1, self.reference, self.metadata),
        ]
        deduplicated = deduplicate_results(results)
        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(deduplicated[0].text, self.text1)
        self.assertEqual(deduplicated[1].text, self.text2)

    def test_empty_list(self):
        """Test deduplication with empty list."""
        results = []
        deduplicated = deduplicate_results(results)
        self.assertEqual(len(deduplicated), 0)


class TestCollectionInfo(unittest.TestCase):
    """Tests for the CollectionInfo class."""

    def test_init(self):
        """Test initialization of CollectionInfo."""
        name = "test_collection"
        description = "Test collection description"
        collection_info = CollectionInfo(name, description)
        
        self.assertEqual(collection_info.collection_name, name)
        self.assertEqual(collection_info.description, description)


class MockVectorDB(BaseVectorDB):
    """Mock implementation of BaseVectorDB for testing."""

    def init_collection(self, dim, collection, description, force_new_collection=False, *args, **kwargs):
        pass

    def insert_data(self, collection, chunks, *args, **kwargs):
        pass

    def search_data(self, collection, vector, *args, **kwargs) -> List[RetrievalResult]:
        return []

    def clear_db(self, *args, **kwargs):
        pass


class TestBaseVectorDB(unittest.TestCase):
    """Tests for the BaseVectorDB class."""

    def setUp(self):
        """Set up test fixtures."""
        self.db = MockVectorDB()

    def test_init_default(self):
        """Test initialization with default collection name."""
        self.assertEqual(self.db.default_collection, "deepsearcher")

    def test_init_custom_collection(self):
        """Test initialization with custom collection name."""
        custom_collection = "custom_collection"
        db = MockVectorDB(default_collection=custom_collection)
        self.assertEqual(db.default_collection, custom_collection)

    def test_list_collections_default(self):
        """Test default list_collections implementation."""
        self.assertIsNone(self.db.list_collections())


if __name__ == "__main__":
    unittest.main() 