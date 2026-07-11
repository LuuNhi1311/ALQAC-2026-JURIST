import unittest
import os
from unittest.mock import patch, MagicMock

import requests
from langchain_core.documents import Document

from deepsearcher.loader.web_crawler import JinaCrawler


class TestJinaCrawler(unittest.TestCase):
    """Tests for the JinaCrawler class."""
    
    @patch.dict(os.environ, {"JINA_API_TOKEN": "fake-token"})
    def test_init_with_token(self):
        """Test initialization with API token in environment."""
        crawler = JinaCrawler()
        self.assertEqual(crawler.jina_api_token, "fake-token")
    
    @patch.dict(os.environ, {"JINAAI_API_KEY": "fake-key"})
    def test_init_with_alternative_key(self):
        """Test initialization with alternative API key in environment."""
        crawler = JinaCrawler()
        self.assertEqual(crawler.jina_api_token, "fake-key")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_token(self):
        """Test initialization without API token raises ValueError."""
        with self.assertRaises(ValueError):
            JinaCrawler()
    
    @patch.dict(os.environ, {"JINA_API_TOKEN": "fake-token"})
    @patch("requests.get")
    def test_crawl_url(self, mock_get):
        """Test crawling a URL."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.text = "# Markdown Content\nThis is a test."
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/markdown"}
        mock_get.return_value = mock_response
        
        # Create the crawler and crawl a test URL
        crawler = JinaCrawler()
        url = "https://example.com"
        documents = crawler.crawl_url(url)
        
        # Check that requests.get was called correctly
        mock_get.assert_called_once_with(
            f"https://r.jina.ai/{url}",
            headers={
                "Authorization": "Bearer fake-token",
                "X-Return-Format": "markdown",
            }
        )
        
        # Check the results
        self.assertEqual(len(documents), 1)
        document = documents[0]
        
        # Check the content
        self.assertEqual(document.page_content, mock_response.text)
        
        # Check the metadata
        self.assertEqual(document.metadata["reference"], url)
        self.assertEqual(document.metadata["status_code"], 200)
        self.assertEqual(document.metadata["headers"], {"Content-Type": "text/markdown"})
    
    @patch.dict(os.environ, {"JINA_API_TOKEN": "fake-token"})
    @patch("requests.get")
    def test_crawl_url_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        # Set up the mock response to raise an HTTPError
        mock_get.side_effect = requests.exceptions.HTTPError("404 Client Error")
        
        # Create the crawler
        crawler = JinaCrawler()
        
        # Crawl a URL and check that the error is propagated
        with self.assertRaises(requests.exceptions.HTTPError):
            crawler.crawl_url("https://example.com")
    
    @patch.dict(os.environ, {"JINA_API_TOKEN": "fake-token"})
    @patch("requests.get")
    def test_crawl_urls(self, mock_get):
        """Test crawling multiple URLs."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.text = "# Markdown Content\nThis is a test."
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/markdown"}
        mock_get.return_value = mock_response
        
        # Create the crawler and crawl multiple URLs
        crawler = JinaCrawler()
        urls = ["https://example.com", "https://example.org"]
        documents = crawler.crawl_urls(urls)
        
        # Check that requests.get was called twice
        self.assertEqual(mock_get.call_count, 2)
        
        # Check the results
        self.assertEqual(len(documents), 2)
        
        # Check that each document has the correct reference
        references = [doc.metadata["reference"] for doc in documents]
        for url in urls:
            self.assertIn(url, references)


if __name__ == "__main__":
    unittest.main() 