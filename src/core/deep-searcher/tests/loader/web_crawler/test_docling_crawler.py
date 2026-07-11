import unittest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from deepsearcher.loader.web_crawler import DoclingCrawler


class TestDoclingCrawler(unittest.TestCase):
    """Tests for the DoclingCrawler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mocks for the docling modules
        self.docling_patcher = patch.dict('sys.modules', {
            'docling': MagicMock(),
            'docling.document_converter': MagicMock(),
            'docling_core': MagicMock(),
            'docling_core.transforms': MagicMock(),
            'docling_core.transforms.chunker': MagicMock()
        })
        self.docling_patcher.start()
        
        # Create mocks for the classes
        self.mock_document_converter = MagicMock()
        self.mock_hierarchical_chunker = MagicMock()
        
        # Add the mocks to the modules
        import sys
        sys.modules['docling.document_converter'].DocumentConverter = self.mock_document_converter
        sys.modules['docling_core.transforms.chunker'].HierarchicalChunker = self.mock_hierarchical_chunker
        
        # Set up mock instances
        self.mock_converter_instance = MagicMock()
        self.mock_chunker_instance = MagicMock()
        self.mock_document_converter.return_value = self.mock_converter_instance
        self.mock_hierarchical_chunker.return_value = self.mock_chunker_instance
        
        # Create the crawler
        self.crawler = DoclingCrawler()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.docling_patcher.stop()
    
    def test_init(self):
        """Test initialization."""
        # Verify instances were created
        self.mock_document_converter.assert_called_once()
        self.mock_hierarchical_chunker.assert_called_once()
        
        # Check that the instances were assigned correctly
        self.assertEqual(self.crawler.converter, self.mock_converter_instance)
        self.assertEqual(self.crawler.chunker, self.mock_chunker_instance)
    
    def test_crawl_url(self):
        """Test crawling a URL."""
        url = "https://example.com"
        
        # Set up mock document and chunks
        mock_document = MagicMock()
        mock_conversion_result = MagicMock()
        mock_conversion_result.document = mock_document
        
        # Set up three mock chunks
        mock_chunks = []
        for i in range(3):
            chunk = MagicMock()
            chunk.text = f"Chunk {i} content"
            mock_chunks.append(chunk)
        
        # Configure mock converter and chunker
        self.mock_converter_instance.convert.return_value = mock_conversion_result
        self.mock_chunker_instance.chunk.return_value = mock_chunks
        
        # Call the method
        documents = self.crawler.crawl_url(url)
        
        # Verify converter was called correctly
        self.mock_converter_instance.convert.assert_called_once_with(url)
        
        # Verify chunker was called correctly
        self.mock_chunker_instance.chunk.assert_called_once_with(mock_document)
        
        # Check results
        self.assertEqual(len(documents), 3)
        
        # Check each document
        for i, document in enumerate(documents):
            self.assertEqual(document.page_content, f"Chunk {i} content")
            self.assertEqual(document.metadata["reference"], url)
            self.assertEqual(document.metadata["text"], f"Chunk {i} content")
    
    def test_crawl_url_error(self):
        """Test error handling when crawling a URL."""
        url = "https://example.com"
        
        # Configure converter to raise an exception
        self.mock_converter_instance.convert.side_effect = Exception("Test error")
        
        # Verify that the error is propagated
        with self.assertRaises(IOError):
            self.crawler.crawl_url(url)
    
    def test_supported_file_types(self):
        """Test the supported_file_types property."""
        file_types = self.crawler.supported_file_types
        
        # Check that all expected file types are included
        expected_types = [
            "pdf", "docx", "xlsx", "pptx", "md", "adoc", "asciidoc", 
            "html", "xhtml", "csv", "png", "jpg", "jpeg", "tif", "tiff", "bmp"
        ]
        
        for file_type in expected_types:
            self.assertIn(file_type, file_types)
        
        # Check that the count matches
        self.assertEqual(len(file_types), len(expected_types))
    
    def test_crawl_urls(self):
        """Test crawling multiple URLs."""
        urls = ["https://example.com", "https://example.org"]
        
        # Set up mock document and chunks for each URL
        mock_document = MagicMock()
        mock_conversion_result = MagicMock()
        mock_conversion_result.document = mock_document
        
        # Set up one mock chunk per URL
        mock_chunk = MagicMock()
        mock_chunk.text = "Test chunk content"
        
        # Configure mock converter and chunker
        self.mock_converter_instance.convert.return_value = mock_conversion_result
        self.mock_chunker_instance.chunk.return_value = [mock_chunk]
        
        # Call the method
        documents = self.crawler.crawl_urls(urls)
        
        # Verify converter was called for each URL
        self.assertEqual(self.mock_converter_instance.convert.call_count, 2)
        
        # Verify chunker was called for each document
        self.assertEqual(self.mock_chunker_instance.chunk.call_count, 2)
        
        # Check results
        self.assertEqual(len(documents), 2)
        
        # Each URL should have generated one document (with one chunk)
        for document in documents:
            self.assertEqual(document.page_content, "Test chunk content")
            self.assertIn(document.metadata["reference"], urls)


if __name__ == "__main__":
    unittest.main() 