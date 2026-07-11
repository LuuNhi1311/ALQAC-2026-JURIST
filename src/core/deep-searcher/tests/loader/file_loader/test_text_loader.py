import unittest
import os
import tempfile

from deepsearcher.loader.file_loader import TextLoader


class TestTextLoader(unittest.TestCase):
    """Tests for the TextLoader class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.loader = TextLoader()
        
        # Create a temporary directory and file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = os.path.join(self.temp_dir.name, "test.txt")
        self.test_content = "This is a test file content.\nWith multiple lines."
        
        # Write test content to the file
        with open(self.test_file_path, "w", encoding="utf-8") as f:
            f.write(self.test_content)
    
    def tearDown(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()
    
    def test_supported_file_types(self):
        """Test the supported_file_types property."""
        supported_types = self.loader.supported_file_types
        self.assertIsInstance(supported_types, list)
        self.assertIn("txt", supported_types)
        self.assertIn("md", supported_types)
    
    def test_load_file(self):
        """Test loading a text file."""
        documents = self.loader.load_file(self.test_file_path)
        
        # Check that we got a list with one document
        self.assertIsInstance(documents, list)
        self.assertEqual(len(documents), 1)
        
        # Check the document content
        document = documents[0]
        self.assertEqual(document.page_content, self.test_content)
        
        # Check the metadata
        self.assertIn("reference", document.metadata)
        self.assertEqual(document.metadata["reference"], self.test_file_path)
    
    def test_load_directory(self):
        """Test loading a directory with text files."""
        # Create additional test files
        md_file_path = os.path.join(self.temp_dir.name, "test.md")
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write("# Markdown Test\nThis is a markdown file.")
        
        # Create a non-supported file
        pdf_file_path = os.path.join(self.temp_dir.name, "test.pdf")
        with open(pdf_file_path, "w", encoding="utf-8") as f:
            f.write("PDF content")
        
        # Load the directory
        documents = self.loader.load_directory(self.temp_dir.name)
        
        # Check that we got documents for supported files only
        self.assertEqual(len(documents), 2)
        
        # Get references
        references = [doc.metadata["reference"] for doc in documents]
        
        # Check that supported files were loaded
        self.assertIn(self.test_file_path, references)
        self.assertIn(md_file_path, references)
        
        # Check that unsupported file was not loaded
        for doc in documents:
            self.assertNotEqual(doc.metadata["reference"], pdf_file_path)


if __name__ == "__main__":
    unittest.main() 