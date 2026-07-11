import unittest
import asyncio
from unittest.mock import patch, MagicMock
import warnings

from langchain_core.documents import Document

from deepsearcher.loader.web_crawler import Crawl4AICrawler


class TestCrawl4AICrawler(unittest.TestCase):
    """Tests for the Crawl4AICrawler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock for the crawl4ai module
        warnings.filterwarnings('ignore', message='coroutine.*never awaited')
        self.crawl4ai_patcher = patch.dict('sys.modules', {'crawl4ai': MagicMock()})
        self.crawl4ai_patcher.start()
        
        # Create mocks for the classes
        self.mock_async_web_crawler = MagicMock()
        self.mock_browser_config = MagicMock()
        
        # Set up the from_kwargs method
        self.mock_config_instance = MagicMock()
        self.mock_browser_config.from_kwargs.return_value = self.mock_config_instance
        
        # Add the mocks to the crawl4ai module
        import sys
        sys.modules['crawl4ai'].AsyncWebCrawler = self.mock_async_web_crawler
        sys.modules['crawl4ai'].BrowserConfig = self.mock_browser_config
        
        # Set up mock instances
        self.mock_crawler_instance = MagicMock()
        self.mock_async_web_crawler.return_value = self.mock_crawler_instance
        
        # For context manager behavior
        self.mock_crawler_instance.__aenter__.return_value = self.mock_crawler_instance
        self.mock_crawler_instance.__aexit__.return_value = None
        
        # Create test browser_config
        self.test_browser_config = {"headless": True}
        
        # Create the crawler
        self.crawler = Crawl4AICrawler(browser_config=self.test_browser_config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.crawl4ai_patcher.stop()
    
    def test_init(self):
        """Test initialization."""
        # Verify that the browser_config was stored
        self.assertEqual(self.crawler.browser_config, self.test_browser_config)
        
        # Verify that the crawler is not initialized
        self.assertIsNone(self.crawler.crawler)
    
    def test_lazy_init(self):
        """Test the lazy initialization of the crawler."""
        # Call _lazy_init method
        self.crawler._lazy_init()
        
        # Verify BrowserConfig.from_kwargs was called
        self.mock_browser_config.from_kwargs.assert_called_once_with(self.test_browser_config)
        
        # Verify AsyncWebCrawler was initialized
        self.mock_async_web_crawler.assert_called_once_with(config=self.mock_config_instance)
        
        # Verify that the crawler is now set
        self.assertEqual(self.crawler.crawler, self.mock_crawler_instance)
    
    @patch('deepsearcher.loader.web_crawler.crawl4ai_crawler.asyncio.run')
    def test_crawl_url(self, mock_asyncio_run):
        """Test crawling a single URL."""
        url = "https://example.com"
        
        # Set up mock document
        mock_document = Document(
            page_content="# Example Page\nThis is a test page.",
            metadata={"reference": url, "title": "Example Page"}
        )
        
        # Configure asyncio.run to return a document
        mock_asyncio_run.return_value = mock_document
        
        # Call the method
        documents = self.crawler.crawl_url(url)
        
        # Verify asyncio.run was called with _async_crawl
        mock_asyncio_run.assert_called_once()
        
        # Check results
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0], mock_document)
    
    @patch('deepsearcher.loader.web_crawler.crawl4ai_crawler.asyncio.run')
    def test_crawl_url_error(self, mock_asyncio_run):
        """Test error handling when crawling a URL."""
        url = "https://example.com"
        
        # Configure asyncio.run to raise an exception
        mock_asyncio_run.side_effect = Exception("Test error")
        
        # Call the method
        documents = self.crawler.crawl_url(url)
        
        # Should return empty list on error
        self.assertEqual(documents, [])
    
    @patch('deepsearcher.loader.web_crawler.crawl4ai_crawler.asyncio.run')
    def test_crawl_urls(self, mock_asyncio_run):
        """Test crawling multiple URLs."""
        urls = ["https://example.com", "https://example.org"]
        
        # Set up mock documents
        mock_documents = [
            Document(
                page_content="# Example Page 1\nThis is test page 1.",
                metadata={"reference": urls[0], "title": "Example Page 1"}
            ),
            Document(
                page_content="# Example Page 2\nThis is test page 2.",
                metadata={"reference": urls[1], "title": "Example Page 2"}
            )
        ]
        
        # Configure asyncio.run to return documents
        mock_asyncio_run.return_value = mock_documents
        
        # Call the method
        documents = self.crawler.crawl_urls(urls)
        
        # Verify asyncio.run was called with _async_crawl_many
        mock_asyncio_run.assert_called_once()
        
        # Check results
        self.assertEqual(documents, mock_documents)
    
    @patch('deepsearcher.loader.web_crawler.crawl4ai_crawler.asyncio.run')
    def test_crawl_urls_error(self, mock_asyncio_run):
        """Test error handling when crawling multiple URLs."""
        urls = ["https://example.com", "https://example.org"]
        
        # Configure asyncio.run to raise an exception
        mock_asyncio_run.side_effect = Exception("Test error")
        
        # Call the method
        documents = self.crawler.crawl_urls(urls)
        
        # Should return empty list on error
        self.assertEqual(documents, [])


if __name__ == "__main__":
    unittest.main() 