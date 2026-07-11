import unittest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from deepsearcher.loader.file_loader import UnstructuredLoader


class TestUnstructuredLoader(unittest.TestCase):
    """Tests for the UnstructuredLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for tests
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test file
        self.test_file_path = os.path.join(self.temp_dir.name, "test.txt")
        with open(self.test_file_path, "w", encoding="utf-8") as f:
            f.write("This is a test file.")
        
        # Path for mock processed outputs
        self.mock_output_dir = os.path.join(self.temp_dir.name, "mock_outputs")
        os.makedirs(self.mock_output_dir, exist_ok=True)
        
        # Create a mock JSON output file
        self.mock_json_path = os.path.join(self.mock_output_dir, "test_output.json")
        with open(self.mock_json_path, "w", encoding="utf-8") as f:
            f.write('{"elements": [{"text": "This is extracted text.", "metadata": {"filename": "test.txt"}}]}')
        
        # Set up patches for unstructured modules
        self.unstructured_modules = {
            'unstructured_ingest': MagicMock(),
            'unstructured_ingest.interfaces': MagicMock(),
            'unstructured_ingest.pipeline': MagicMock(),
            'unstructured_ingest.pipeline.pipeline': MagicMock(),
            'unstructured_ingest.processes': MagicMock(),
            'unstructured_ingest.processes.connectors': MagicMock(),
            'unstructured_ingest.processes.connectors.local': MagicMock(),
            'unstructured_ingest.processes.partitioner': MagicMock(),
            'unstructured': MagicMock(),
            'unstructured.staging': MagicMock(),
            'unstructured.staging.base': MagicMock(),
        }
        
        self.patches = []
        for module_name, mock_module in self.unstructured_modules.items():
            patcher = patch.dict('sys.modules', {module_name: mock_module})
            patcher.start()
            self.patches.append(patcher)
        
        # Create mock Pipeline class
        self.mock_pipeline = MagicMock()
        self.unstructured_modules['unstructured_ingest.pipeline.pipeline'].Pipeline = self.mock_pipeline
        self.mock_pipeline.from_configs.return_value = self.mock_pipeline
        
        # Create mock Element class
        self.mock_element = MagicMock()
        self.mock_element.text = "This is extracted text."
        self.mock_element.metadata = MagicMock()
        self.mock_element.metadata.to_dict.return_value = {"filename": "test.txt"}
        
        # Set up elements_from_json mock
        self.unstructured_modules['unstructured.staging.base'].elements_from_json = MagicMock()
        self.unstructured_modules['unstructured.staging.base'].elements_from_json.return_value = [self.mock_element]
        
        # Patch makedirs and rmtree but don't assert on them
        with patch('os.makedirs'):
            with patch('shutil.rmtree'):
                self.loader = UnstructuredLoader()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Stop all patches
        for patcher in self.patches:
            patcher.stop()
        
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.loader.directory_with_results, "./pdf_processed_outputs")
    
    def test_supported_file_types(self):
        """Test the supported_file_types property."""
        file_types = self.loader.supported_file_types
        
        # Check that common file types are included
        common_types = ["pdf", "docx", "txt", "html", "md", "jpg"]
        for file_type in common_types:
            self.assertIn(file_type, file_types)
        
        # Check total number of supported types (should be extensive)
        self.assertGreater(len(file_types), 20)
    
    @patch('os.listdir')
    def test_load_file(self, mock_listdir):
        """Test loading a single file."""
        # Configure mocks
        mock_listdir.return_value = ["test_output.json"]
        
        # Call the method
        documents = self.loader.load_file(self.test_file_path)
        
        # Verify Pipeline.from_configs was called
        self.mock_pipeline.from_configs.assert_called_once()
        self.mock_pipeline.run.assert_called_once()
        
        # Verify elements_from_json was called
        self.unstructured_modules['unstructured.staging.base'].elements_from_json.assert_called_once()
        
        # Check results
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "This is extracted text.")
        self.assertEqual(documents[0].metadata["reference"], self.test_file_path)
        self.assertEqual(documents[0].metadata["filename"], "test.txt")
    
    @patch('os.listdir')
    def test_load_directory(self, mock_listdir):
        """Test loading a directory."""
        # Configure mocks
        mock_listdir.return_value = ["test_output.json"]
        
        # Call the method
        documents = self.loader.load_directory(self.temp_dir.name)
        
        # Verify Pipeline.from_configs was called
        self.mock_pipeline.from_configs.assert_called_once()
        self.mock_pipeline.run.assert_called_once()
        
        # Check results
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "This is extracted text.")
        self.assertEqual(documents[0].metadata["reference"], self.temp_dir.name)
    
    @patch('os.listdir')
    def test_load_with_api(self, mock_listdir):
        """Test loading with API environment variables."""
        # Create a mock for os.environ.get
        with patch('os.environ.get') as mock_env_get:
            # Configure environment variables
            mock_env_get.side_effect = lambda key, default=None: {
                "UNSTRUCTURED_API_KEY": "test-key",
                "UNSTRUCTURED_API_URL": "https://api.example.com"
            }.get(key, default)
            
            # Configure listdir mock
            mock_listdir.return_value = ["test_output.json"]
            
            # Create a mock for PartitionerConfig
            mock_partitioner_config = MagicMock()
            self.unstructured_modules['unstructured_ingest.processes.partitioner'].PartitionerConfig = mock_partitioner_config
            
            # Call the method
            documents = self.loader.load_file(self.test_file_path)
            
            # Verify Pipeline.from_configs was called
            self.mock_pipeline.from_configs.assert_called_once()
            
            # Check that PartitionerConfig was called with correct parameters
            mock_partitioner_config.assert_called_once()
            args, kwargs = mock_partitioner_config.call_args
            self.assertTrue(kwargs.get('partition_by_api'))
            self.assertEqual(kwargs.get('api_key'), "test-key")
            self.assertEqual(kwargs.get('partition_endpoint'), "https://api.example.com")
            
            # Check results
            self.assertEqual(len(documents), 1)
    
    @patch('os.listdir')
    def test_empty_output(self, mock_listdir):
        """Test handling of empty output directory."""
        # Configure listdir to return no JSON files
        mock_listdir.return_value = []
        
        # Call the method
        documents = self.loader.load_file(self.test_file_path)
        
        # Check results
        self.assertEqual(len(documents), 0)
    
    @patch('os.listdir')
    def test_error_reading_json(self, mock_listdir):
        """Test handling of errors when reading JSON files."""
        # Configure listdir mock
        mock_listdir.return_value = ["test_output.json"]
        
        # Configure elements_from_json to raise an IOError
        self.unstructured_modules['unstructured.staging.base'].elements_from_json.side_effect = IOError("Test error")
        
        # Call the method
        documents = self.loader.load_file(self.test_file_path)
        
        # Check results (should be empty)
        self.assertEqual(len(documents), 0)


if __name__ == "__main__":
    unittest.main() 