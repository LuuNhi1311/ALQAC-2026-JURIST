import unittest
from unittest.mock import patch, MagicMock

from deepsearcher.loader.web_crawler.base import BaseCrawler


class TestBaseCrawler(unittest.TestCase):
    """Tests for the BaseCrawler class."""
    
    def test_abstract_methods(self):
        """Test that BaseCrawler defines abstract methods."""
        # For abstract base classes, we can check if methods are defined
        # but not implemented in the base class
        self.assertTrue(hasattr(BaseCrawler, 'crawl_url'))
    
    def test_crawl_urls(self):
        """Test the crawl_urls method."""
        # Create a subclass of BaseCrawler for testing
        class TestCrawler(BaseCrawler):
            def crawl_url(self, url, **kwargs):
                # Mock implementation that returns a list of documents
                from langchain_core.documents import Document
                return [Document(
                    page_content=f"Content from {url}",
                    metadata={"reference": url, "kwargs": kwargs}
                )]
        
        # Create test URLs
        urls = [
            "https://example.com",
            "https://example.org",
            "https://example.net"
        ]
        
        # Test crawling multiple URLs
        crawler = TestCrawler()
        documents = crawler.crawl_urls(urls, param1="value1")
        
        # Check the results
        self.assertEqual(len(documents), 3)  # One document per URL
        
        # Verify each document
        references = [doc.metadata["reference"] for doc in documents]
        for url in urls:
            self.assertIn(url, references)
        
        # Check that kwargs were passed correctly
        for doc in documents:
            self.assertEqual(doc.metadata["kwargs"]["param1"], "value1")


if __name__ == "__main__":
    unittest.main() 