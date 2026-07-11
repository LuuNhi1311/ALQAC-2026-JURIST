import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from deepsearcher.loader.file_loader.base import BaseLoader


class TestBaseLoader(unittest.TestCase):
    """Tests for the BaseLoader class."""
    
    def test_abstract_methods(self):
        """Test that BaseLoader defines abstract methods."""
        # For abstract base classes, we can check if methods are defined
        # but not implemented in the base class
        self.assertTrue(hasattr(BaseLoader, 'load_file'))
        self.assertTrue(hasattr(BaseLoader, 'supported_file_types'))
    
    def test_load_directory(self):
        """Test the load_directory method."""
        # Create a subclass of BaseLoader for testing
        class TestLoader(BaseLoader):
            @property
            def supported_file_types(self):
                return [".txt", ".md"]
                
            def load_file(self, file_path):
                # Mock implementation that returns a simple Document
                return [Document(page_content=f"Content of {file_path}", metadata={"reference": file_path})]
        
        # Create a temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            file_paths = [
                os.path.join(temp_dir, "test1.txt"),
                os.path.join(temp_dir, "test2.md"),
                os.path.join(temp_dir, "test3.pdf"),  # Unsupported format
                os.path.join(temp_dir, "subdir", "test4.txt")
            ]
            
            # Create subdirectory
            os.makedirs(os.path.join(temp_dir, "subdir"), exist_ok=True)
            
            # Create files
            for path in file_paths:
                # Skip the file if it's in a subdirectory that doesn't exist
                if not os.path.exists(os.path.dirname(path)):
                    continue
                with open(path, 'w') as f:
                    f.write(f"Content of {path}")
            
            # Test loading the directory
            loader = TestLoader()
            documents = loader.load_directory(temp_dir)
            
            # Check the results
            self.assertEqual(len(documents), 3)  # Should find 3 supported files
            
            # Verify each document
            references = [doc.metadata["reference"] for doc in documents]
            self.assertIn(file_paths[0], references)  # test1.txt
            self.assertIn(file_paths[1], references)  # test2.md
            self.assertNotIn(file_paths[2], references)  # test3.pdf (unsupported)
            self.assertIn(file_paths[3], references)  # subdir/test4.txt


if __name__ == "__main__":
    unittest.main() 