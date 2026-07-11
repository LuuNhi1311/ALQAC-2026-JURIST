import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import json

from deepsearcher.vector_db.base import RetrievalResult
from deepsearcher.loader.splitter import Chunk
import logging
logging.disable(logging.CRITICAL)

class TestOracleDB(unittest.TestCase):
    """Tests for the Oracle vector database implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock modules
        self.mock_oracledb = MagicMock()
        
        # Setup mock DB_TYPE_VECTOR
        self.mock_oracledb.DB_TYPE_VECTOR = "VECTOR"
        self.mock_oracledb.defaults = MagicMock()
        
        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {
            'oracledb': self.mock_oracledb
        })
        
        # Start the patcher
        self.module_patcher.start()
        
        # Import after mocking
        from deepsearcher.vector_db import OracleDB
        self.OracleDB = OracleDB

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    def test_init(self):
        """Test basic initialization."""
        # Setup mock
        mock_pool = MagicMock()
        self.mock_oracledb.create_pool.return_value = mock_pool
        
        oracle_db = self.OracleDB(
            user="test_user",
            password="test_password",
            dsn="test_dsn",
            config_dir="/test/config",
            wallet_location="/test/wallet",
            wallet_password="test_wallet_pwd",
            default_collection="test_collection"
        )
        
        # Verify initialization
        self.assertEqual(oracle_db.default_collection, "test_collection")
        self.assertIsNotNone(oracle_db.client)
        self.mock_oracledb.create_pool.assert_called_once()
        self.assertTrue(self.mock_oracledb.defaults.fetch_lobs is False)

    def test_insert_data(self):
        """Test inserting data."""
        # Setup mock
        mock_pool = MagicMock()
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        
        self.mock_oracledb.create_pool.return_value = mock_pool
        mock_pool.acquire.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        oracle_db = self.OracleDB(
            user="test_user",
            password="test_password",
            dsn="test_dsn",
            config_dir="/test/config",
            wallet_location="/test/wallet",
            wallet_password="test_wallet_pwd"
        )
        
        # Create test data
        d = 8
        rng = np.random.default_rng(seed=42)
        test_chunks = [
            Chunk(
                embedding=rng.random(d).tolist(),
                text="hello world",
                reference="test.txt",
                metadata={"key": "value1"}
            ),
            Chunk(
                embedding=rng.random(d).tolist(),
                text="hello oracle",
                reference="test.txt",
                metadata={"key": "value2"}
            )
        ]
        
        oracle_db.insert_data(collection="test_collection", chunks=test_chunks)
        self.assertTrue(mock_cursor.execute.called)
        self.assertTrue(mock_connection.commit.called)

    def test_search_data(self):
        """Test search functionality."""
        # Setup mock
        mock_pool = MagicMock()
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        
        self.mock_oracledb.create_pool.return_value = mock_pool
        mock_pool.acquire.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock search results
        mock_cursor.description = [("embedding",), ("text",), ("reference",), ("distance",), ("metadata",)]
        mock_cursor.fetchall.return_value = [
            (
                np.array([0.1, 0.2, 0.3]),
                "hello world",
                "test.txt",
                0.95,
                json.dumps({"key": "value1"})
            ),
            (
                np.array([0.4, 0.5, 0.6]),
                "hello oracle",
                "test.txt",
                0.85,
                json.dumps({"key": "value2"})
            )
        ]
        
        oracle_db = self.OracleDB(
            user="test_user",
            password="test_password",
            dsn="test_dsn",
            config_dir="/test/config",
            wallet_location="/test/wallet",
            wallet_password="test_wallet_pwd"
        )
        
        # Test search
        d = 8
        rng = np.random.default_rng(seed=42)
        query_vector = rng.random(d)
        
        results = oracle_db.search_data(
            collection="test_collection",
            vector=query_vector,
            top_k=2
        )
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, RetrievalResult)

    def test_list_collections(self):
        """Test listing collections."""
        # Setup mock
        mock_pool = MagicMock()
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        
        self.mock_oracledb.create_pool.return_value = mock_pool
        mock_pool.acquire.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock list_collections response
        mock_cursor.description = [("collection",), ("description",)]
        mock_cursor.fetchall.return_value = [
            ("test_collection_1", "Test collection 1"),
            ("test_collection_2", "Test collection 2")
        ]
        
        oracle_db = self.OracleDB(
            user="test_user",
            password="test_password",
            dsn="test_dsn",
            config_dir="/test/config",
            wallet_location="/test/wallet",
            wallet_password="test_wallet_pwd"
        )
        
        collections = oracle_db.list_collections()
        self.assertIsInstance(collections, list)
        self.assertEqual(len(collections), 2)

    def test_clear_db(self):
        """Test clearing database."""
        # Setup mock
        mock_pool = MagicMock()
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        
        self.mock_oracledb.create_pool.return_value = mock_pool
        mock_pool.acquire.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        oracle_db = self.OracleDB(
            user="test_user",
            password="test_password",
            dsn="test_dsn",
            config_dir="/test/config",
            wallet_location="/test/wallet",
            wallet_password="test_wallet_pwd"
        )
        
        oracle_db.clear_db("test_collection")
        self.assertTrue(mock_cursor.execute.called)
        self.assertTrue(mock_connection.commit.called)

    def test_has_collection(self):
        """Test checking if collection exists."""
        # Setup mock
        mock_pool = MagicMock()
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        
        self.mock_oracledb.create_pool.return_value = mock_pool
        mock_pool.acquire.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock check_table response first (called during init)
        mock_cursor.description = [("table_name",)]
        mock_cursor.fetchall.return_value = [
            ("DEEPSEARCHER_COLLECTION_INFO",),
            ("DEEPSEARCHER_COLLECTION_ITEM",)
        ]
        
        oracle_db = self.OracleDB(
            user="test_user",
            password="test_password",
            dsn="test_dsn",
            config_dir="/test/config",
            wallet_location="/test/wallet",
            wallet_password="test_wallet_pwd"
        )
        
        # Now mock has_collection response - collection exists
        mock_cursor.description = [("rowcnt",)]
        mock_cursor.fetchall.return_value = [(1,)]  # Return tuple, not dict
        
        result = oracle_db.has_collection("test_collection")
        self.assertTrue(result)
        
        # Test collection doesn't exist
        mock_cursor.fetchall.return_value = [(0,)]  # Return tuple, not dict
        result = oracle_db.has_collection("nonexistent_collection")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main() 