import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys

from deepsearcher.vector_db import AzureSearch
from deepsearcher.vector_db.base import RetrievalResult


class TestAzureSearch(unittest.TestCase):
    """Tests for the Azure Search vector database implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock modules
        self.mock_azure = MagicMock()
        self.mock_search = MagicMock()
        self.mock_indexes = MagicMock()
        self.mock_models = MagicMock()
        self.mock_credentials = MagicMock()
        self.mock_exceptions = MagicMock()
        
        # Setup nested structure
        self.mock_azure.search = self.mock_search
        self.mock_search.documents = self.mock_search
        self.mock_search.documents.indexes = self.mock_indexes
        self.mock_indexes.models = self.mock_models
        self.mock_azure.core = self.mock_credentials
        self.mock_azure.core.credentials = self.mock_credentials
        self.mock_azure.core.exceptions = self.mock_exceptions
        
        # Mock specific models needed for init_collection
        self.mock_models.SearchableField = MagicMock()
        self.mock_models.SimpleField = MagicMock()
        self.mock_models.SearchField = MagicMock()
        self.mock_models.SearchIndex = MagicMock()
        
        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {
            'azure': self.mock_azure,
            'azure.core': self.mock_credentials,
            'azure.core.credentials': self.mock_credentials,
            'azure.core.exceptions': self.mock_exceptions,
            'azure.search': self.mock_search,
            'azure.search.documents': self.mock_search,
            'azure.search.documents.indexes': self.mock_indexes,
            'azure.search.documents.indexes.models': self.mock_models
        })
        
        # Start the patcher
        self.module_patcher.start()
        
        # Import after mocking
        from deepsearcher.vector_db import AzureSearch
        from deepsearcher.vector_db.base import RetrievalResult
        
        self.AzureSearch = AzureSearch
        self.RetrievalResult = RetrievalResult

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    def test_init(self):
        """Test basic initialization."""
        # Setup mock
        mock_client = MagicMock()
        self.mock_search.SearchClient.return_value = mock_client
        
        azure_search = self.AzureSearch(
            endpoint="https://test-search.search.windows.net",
            index_name="test-index",
            api_key="test-key",
            vector_field="content_vector"
        )
        
        # Verify initialization
        self.assertEqual(azure_search.index_name, "test-index")
        self.assertEqual(azure_search.endpoint, "https://test-search.search.windows.net")
        self.assertEqual(azure_search.api_key, "test-key")
        self.assertEqual(azure_search.vector_field, "content_vector")
        self.assertIsNotNone(azure_search.client)

    def test_init_collection(self):
        """Test collection initialization."""
        # Setup mock
        mock_index_client = MagicMock()
        self.mock_indexes.SearchIndexClient.return_value = mock_index_client
        mock_index_client.create_index.return_value = None
        
        azure_search = self.AzureSearch(
            endpoint="https://test-search.search.windows.net",
            index_name="test-index",
            api_key="test-key",
            vector_field="content_vector"
        )
        
        azure_search.init_collection()
        self.assertTrue(mock_index_client.create_index.called)

    def test_insert_data(self):
        """Test inserting data."""
        # Setup mock
        mock_client = MagicMock()
        self.mock_search.SearchClient.return_value = mock_client
        
        # Mock successful upload result
        mock_result = [MagicMock(succeeded=True) for _ in range(2)]
        mock_client.upload_documents.return_value = mock_result
        
        azure_search = self.AzureSearch(
            endpoint="https://test-search.search.windows.net",
            index_name="test-index",
            api_key="test-key",
            vector_field="content_vector"
        )
        
        # Create test data
        d = 1536  # Azure Search expects 1536 dimensions
        rng = np.random.default_rng(seed=42)
        
        test_docs = [
            {
                "text": "hello world",
                "vector": rng.random(d).tolist(),
                "id": "doc1"
            },
            {
                "text": "hello azure search",
                "vector": rng.random(d).tolist(),
                "id": "doc2"
            }
        ]
        
        results = azure_search.insert_data(documents=test_docs)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(results))

    def test_search_data(self):
        """Test search functionality."""
        # Setup mock
        mock_client = MagicMock()
        self.mock_search.SearchClient.return_value = mock_client
        
        # Mock search results
        d = 1536
        rng = np.random.default_rng(seed=42)
        
        mock_results = MagicMock()
        mock_results.results = [
            {
                "content": "hello world",
                "id": "doc1",
                "@search.score": 0.95
            },
            {
                "content": "hello azure search",
                "id": "doc2",
                "@search.score": 0.85
            }
        ]
        mock_client._client.documents.search_post.return_value = mock_results
        
        azure_search = self.AzureSearch(
            endpoint="https://test-search.search.windows.net",
            index_name="test-index",
            api_key="test-key",
            vector_field="content_vector"
        )
        
        # Test search
        query_vector = rng.random(d).tolist()
        results = azure_search.search_data(
            collection="test-index",
            vector=query_vector,
            top_k=2
        )
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        # Verify results are RetrievalResult objects
        for result in results:
            self.assertIsInstance(result, self.RetrievalResult)

    def test_clear_db(self):
        """Test clearing database."""
        # Setup mock
        mock_client = MagicMock()
        self.mock_search.SearchClient.return_value = mock_client
        
        # Mock search results for documents to delete
        mock_client.search.return_value = [
            {"id": "doc1"},
            {"id": "doc2"}
        ]
        
        azure_search = self.AzureSearch(
            endpoint="https://test-search.search.windows.net",
            index_name="test-index",
            api_key="test-key",
            vector_field="content_vector"
        )
        
        deleted_count = azure_search.clear_db()
        self.assertEqual(deleted_count, 2)

    def test_list_collections(self):
        """Test listing collections."""
        # Setup mock
        mock_index_client = MagicMock()
        self.mock_indexes.SearchIndexClient.return_value = mock_index_client
        
        # Mock list_indexes response
        mock_index1 = MagicMock()
        mock_index1.name = "test-index-1"
        mock_index1.fields = ["field1", "field2"]
        
        mock_index2 = MagicMock()
        mock_index2.name = "test-index-2"
        mock_index2.fields = ["field1", "field2", "field3"]
        
        mock_index_client.list_indexes.return_value = [mock_index1, mock_index2]
        
        azure_search = self.AzureSearch(
            endpoint="https://test-search.search.windows.net",
            index_name="test-index",
            api_key="test-key",
            vector_field="content_vector"
        )
        
        collections = azure_search.list_collections()
        self.assertIsInstance(collections, list)
        self.assertEqual(len(collections), 2)


if __name__ == "__main__":
    unittest.main() 