import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from deepsearcher.loader.file_loader import DoclingLoader


class TestDoclingLoader(unittest.TestCase):
    """Tests for the DoclingLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patches for the docling modules
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
        
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test markdown file
        self.md_file_path = os.path.join(self.temp_dir.name, "test.md")
        with open(self.md_file_path, "w", encoding="utf-8") as f:
            f.write("# Test Markdown\nThis is a test markdown file.")
        
        # Create a test unsupported file
        self.unsupported_file_path = os.path.join(self.temp_dir.name, "test.xyz")
        with open(self.unsupported_file_path, "w", encoding="utf-8") as f:
            f.write("This is an unsupported file type.")
        
        # Create a subdirectory with a test file
        self.sub_dir = os.path.join(self.temp_dir.name, "subdir")
        os.makedirs(self.sub_dir, exist_ok=True)
        self.sub_file_path = os.path.join(self.sub_dir, "subfile.md")
        with open(self.sub_file_path, "w", encoding="utf-8") as f:
            f.write("# Subdir Test\nThis is a test markdown file in a subdirectory.")
        
        # Create the loader
        self.loader = DoclingLoader()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.docling_patcher.stop()
        self.temp_dir.cleanup()
    
    def test_init(self):
        """Test initialization."""
        # Verify instances were created
        self.mock_document_converter.assert_called_once()
        self.mock_hierarchical_chunker.assert_called_once()
        
        # Check that the instances were assigned correctly
        self.assertEqual(self.loader.converter, self.mock_converter_instance)
        self.assertEqual(self.loader.chunker, self.mock_chunker_instance)
    
    def test_supported_file_types(self):
        """Test the supported_file_types property."""
        file_types = self.loader.supported_file_types
        
        # Check that the common file types are included
        common_types = ["pdf", "docx", "md", "html", "csv", "jpg"]
        for file_type in common_types:
            self.assertIn(file_type, file_types)
    
    def test_load_file(self):
        """Test loading a single file."""
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
        documents = self.loader.load_file(self.md_file_path)
        
        # Verify converter was called correctly
        self.mock_converter_instance.convert.assert_called_once_with(self.md_file_path)
        
        # Verify chunker was called correctly
        self.mock_chunker_instance.chunk.assert_called_once_with(mock_document)
        
        # Check results
        self.assertEqual(len(documents), 3)
        
        # Check each document
        for i, document in enumerate(documents):
            self.assertEqual(document.page_content, f"Chunk {i} content")
            self.assertEqual(document.metadata["reference"], self.md_file_path)
            self.assertEqual(document.metadata["text"], f"Chunk {i} content")
    
    def test_load_file_not_found(self):
        """Test loading a non-existent file."""
        non_existent_file = os.path.join(self.temp_dir.name, "non_existent.md")
        with self.assertRaises(FileNotFoundError):
            self.loader.load_file(non_existent_file)
    
    def test_load_unsupported_file_type(self):
        """Test loading a file with unsupported extension."""
        with self.assertRaises(ValueError):
            self.loader.load_file(self.unsupported_file_path)
    
    def test_load_file_error(self):
        """Test error handling when loading a file."""
        # Configure converter to raise an exception
        self.mock_converter_instance.convert.side_effect = Exception("Test error")
        
        # Verify that the error is propagated
        with self.assertRaises(IOError):
            self.loader.load_file(self.md_file_path)
    
    def test_load_directory(self):
        """Test loading a directory."""
        # Set up mock document and chunks
        mock_document = MagicMock()
        mock_conversion_result = MagicMock()
        mock_conversion_result.document = mock_document
        
        # Set up a single mock chunk
        mock_chunk = MagicMock()
        mock_chunk.text = "Test chunk content"
        
        # Configure mock converter and chunker
        self.mock_converter_instance.convert.return_value = mock_conversion_result
        self.mock_chunker_instance.chunk.return_value = [mock_chunk]
        
        # Load the directory
        documents = self.loader.load_directory(self.temp_dir.name)
        
        # Verify converter was called twice (once for each MD file)
        self.assertEqual(self.mock_converter_instance.convert.call_count, 2)
        
        # Verify converter was called with both MD files
        self.mock_converter_instance.convert.assert_any_call(self.md_file_path)
        self.mock_converter_instance.convert.assert_any_call(self.sub_file_path)
        
        # Check results - should have two documents (one from each MD file)
        self.assertEqual(len(documents), 2)
        
        # Check each document
        for document in documents:
            self.assertEqual(document.page_content, "Test chunk content")
            self.assertEqual(document.metadata["text"], "Test chunk content")
            self.assertIn(document.metadata["reference"], [self.md_file_path, self.sub_file_path])
    
    def test_load_not_a_directory(self):
        """Test loading a path that is not a directory."""
        with self.assertRaises(NotADirectoryError):
            self.loader.load_directory(self.md_file_path)


if __name__ == "__main__":
    unittest.main() 