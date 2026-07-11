import unittest
import os
from unittest.mock import patch, MagicMock
import logging
import warnings
import multiprocessing.resource_tracker

# Disable logging for tests
logging.disable(logging.CRITICAL)

# Suppress resource tracker warning
warnings.filterwarnings("ignore", category=ResourceWarning)

# Patch resource tracker to avoid warnings
def _resource_tracker():
    pass
multiprocessing.resource_tracker._resource_tracker = _resource_tracker

from deepsearcher.embedding import GeminiEmbedding
from deepsearcher.embedding.gemini_embedding import GEMINI_MODEL_DIM_MAP


class TestGeminiEmbedding(unittest.TestCase):
    """Tests for the GeminiEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_genai = MagicMock()
        self.mock_client = MagicMock()
        self.mock_types = MagicMock()
        
        # Set up the mock module structure
        self.mock_genai.Client = MagicMock(return_value=self.mock_client)
        self.mock_genai.types = self.mock_types
        
        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'google.genai': self.mock_genai})
        self.module_patcher.start()
        
        # Set up mock response for embed_content
        self.mock_response = MagicMock()
        self.mock_response.embeddings = [
            MagicMock(values=[0.1] * 768)  # Default embedding for text-embedding-004
        ]
        self.mock_client.models.embed_content.return_value = self.mock_response
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_default(self):
        """Test initialization with default parameters."""
        # Create instance to test
        embedding = GeminiEmbedding()
        
        # Check that Client was initialized correctly
        self.mock_genai.Client.assert_called_once_with(api_key=None)
        
        # Check default model and dimension
        self.assertEqual(embedding.model, "text-embedding-004")
        self.assertEqual(embedding.dim, 768)
        self.assertEqual(embedding.dimension, 768)
    
    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_api_key_from_env'}, clear=True)
    def test_init_with_api_key_from_env(self):
        """Test initialization with API key from environment variable."""
        embedding = GeminiEmbedding()
        self.mock_genai.Client.assert_called_with(api_key='test_api_key_from_env')
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_api_key_parameter(self):
        """Test initialization with API key as parameter."""
        api_key = "test_api_key_param"
        embedding = GeminiEmbedding(api_key=api_key)
        self.mock_genai.Client.assert_called_with(api_key=api_key)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        model = "gemini-embedding-exp-03-07"
        embedding = GeminiEmbedding(model=model)
        
        self.assertEqual(embedding.model, model)
        self.assertEqual(embedding.dim, GEMINI_MODEL_DIM_MAP[model])
        self.assertEqual(embedding.dimension, 3072)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_custom_dimension(self):
        """Test initialization with custom dimension."""
        custom_dim = 1024
        embedding = GeminiEmbedding(dimension=custom_dim)
        
        self.assertEqual(embedding.dim, custom_dim)
        self.assertEqual(embedding.dimension, custom_dim)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_query_single_char(self):
        """Test embedding a single character query."""
        # Create instance to test
        embedding = GeminiEmbedding()
        
        query = "a"
        result = embedding.embed_query(query)
        
        # Check that embed_content was called correctly
        self.mock_client.models.embed_content.assert_called_once()
        call_args = self.mock_client.models.embed_content.call_args
        
        # For single character, it should be passed as is
        self.assertEqual(call_args[1]["model"], "text-embedding-004")
        self.assertEqual(call_args[1]["contents"], query)
        
        # Check result
        self.assertEqual(len(result), 768)
        self.assertEqual(result, [0.1] * 768)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_query_multi_char(self):
        """Test embedding a multi-character query."""
        # Create instance to test
        embedding = GeminiEmbedding()
        
        query = "test query"
        result = embedding.embed_query(query)
        
        # Check that embed_content was called correctly
        self.mock_client.models.embed_content.assert_called_once()
        call_args = self.mock_client.models.embed_content.call_args
        
        # For multi-character string, it should be joined with spaces
        self.assertEqual(call_args[1]["model"], "text-embedding-004")
        self.assertEqual(call_args[1]["contents"], "t e s t   q u e r y")
        
        # Check result
        self.assertEqual(len(result), 768)
        self.assertEqual(result, [0.1] * 768)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        # Create instance to test
        embedding = GeminiEmbedding()
        
        # Set up mock response for multiple documents
        mock_embeddings = [
            MagicMock(values=[0.1] * 768),
            MagicMock(values=[0.2] * 768),
            MagicMock(values=[0.3] * 768)
        ]
        self.mock_response.embeddings = mock_embeddings
        
        texts = ["text 1", "text 2", "text 3"]
        results = embedding.embed_documents(texts)
        
        # Check that embed_content was called correctly
        self.mock_client.models.embed_content.assert_called_once()
        call_args = self.mock_client.models.embed_content.call_args
        self.assertEqual(call_args[1]["model"], "text-embedding-004")
        self.assertEqual(call_args[1]["contents"], texts)
        
        # Check that EmbedContentConfig was used
        config_arg = call_args[1]["config"]
        self.mock_types.EmbedContentConfig.assert_called_once_with(output_dimensionality=768)
        
        # Check results
        self.assertEqual(len(results), 3)
        expected_results = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        for i, result in enumerate(results):
            self.assertEqual(len(result), 768)
            self.assertEqual(result, expected_results[i])
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_chunks(self):
        """Test embedding chunks with batch processing."""
        # Create instance to test
        embedding = GeminiEmbedding()
        
        # Set up mock response for batched documents
        batch1_embeddings = [MagicMock(values=[0.1] * 768)] * 100
        batch2_embeddings = [MagicMock(values=[0.2] * 768)] * 50
        
        # Mock multiple calls to embed_content
        self.mock_client.models.embed_content.side_effect = [
            MagicMock(embeddings=batch1_embeddings),
            MagicMock(embeddings=batch2_embeddings)
        ]
        
        # Create mock chunks
        class MockChunk:
            def __init__(self, text: str):
                self.text = text
                self.embedding = None
        
        chunks = [MockChunk(f"text {i}") for i in range(150)]
        results = embedding.embed_chunks(chunks, batch_size=100)
        
        # Check that embed_content was called twice (150 chunks split into 2 batches)
        self.assertEqual(self.mock_client.models.embed_content.call_count, 2)
        
        # Check that the same chunk objects are returned
        self.assertEqual(len(results), 150)
        self.assertEqual(results, chunks)
        
        # Check that each chunk has an embedding
        for i, chunk in enumerate(results):
            self.assertIsNotNone(chunk.embedding)
            if i < 100:
                self.assertEqual(chunk.embedding, [0.1] * 768)
            else:
                self.assertEqual(chunk.embedding, [0.2] * 768)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_dimension_property_different_models(self):
        """Test the dimension property for different models."""
        # Create instance to test
        embedding = GeminiEmbedding()
        
        # Test default model
        self.assertEqual(embedding.dimension, 768)
        
        # Test experimental model
        embedding_exp = GeminiEmbedding(model="gemini-embedding-exp-03-07")
        self.assertEqual(embedding_exp.dimension, 3072)
        
        # Test custom dimension
        embedding_custom = GeminiEmbedding(dimension=512)
        self.assertEqual(embedding_custom.dimension, 512)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_get_dim_method(self):
        """Test the private _get_dim method."""
        # Create instance to test
        embedding = GeminiEmbedding()
        
        # Test default dimension
        self.assertEqual(embedding._get_dim(), 768)
        
        # Test custom dimension
        embedding_custom = GeminiEmbedding(dimension=1024)
        self.assertEqual(embedding_custom._get_dim(), 1024)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_content_method(self):
        """Test the private _embed_content method."""
        # Create instance to test
        embedding = GeminiEmbedding()
        
        texts = ["test text 1", "test text 2"]
        result = embedding._embed_content(texts)
        
        # Check that embed_content was called with correct parameters
        self.mock_client.models.embed_content.assert_called_once()
        call_args = self.mock_client.models.embed_content.call_args
        
        self.assertEqual(call_args[1]["model"], "text-embedding-004")
        self.assertEqual(call_args[1]["contents"], texts)
        self.mock_types.EmbedContentConfig.assert_called_once_with(output_dimensionality=768)
        
        # Check that the response embeddings are returned
        self.assertEqual(result, self.mock_response.embeddings)


if __name__ == "__main__":
    unittest.main() 