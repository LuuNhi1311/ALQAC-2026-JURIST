import unittest
import json
import os
from unittest.mock import patch, MagicMock
import logging

# Disable logging for tests
logging.disable(logging.CRITICAL)

from deepsearcher.embedding import BedrockEmbedding
from deepsearcher.embedding.bedrock_embedding import (
    MODEL_ID_TITAN_TEXT_V2,
    MODEL_ID_TITAN_TEXT_G1,
    MODEL_ID_COHERE_ENGLISH_V3,
)


class TestBedrockEmbedding(unittest.TestCase):
    """Tests for the BedrockEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_boto3 = MagicMock()
        self.mock_client = MagicMock()
        self.mock_boto3.client = MagicMock(return_value=self.mock_client)
        
        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'boto3': self.mock_boto3})
        self.module_patcher.start()
        
        # Configure mock response
        self.mock_response = {
            "body": MagicMock(),
            "ResponseMetadata": {"HTTPStatusCode": 200}
        }
        self.mock_response["body"].read.return_value = json.dumps({"embedding": [0.1] * 1024})
        self.mock_client.invoke_model.return_value = self.mock_response
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_default(self):
        """Test initialization with default parameters."""
        # Create instance to test
        embedding = BedrockEmbedding()
        
        # Check that boto3 client was created correctly
        self.mock_boto3.client.assert_called_once_with(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=None,
            aws_secret_access_key=None
        )
        
        # Check default model
        self.assertEqual(embedding.model, MODEL_ID_TITAN_TEXT_V2)
        
        # Ensure no coroutine warnings
        self.mock_client.invoke_model.return_value = self.mock_response
    
    @patch.dict('os.environ', {
        'AWS_ACCESS_KEY_ID': 'test_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret'
    }, clear=True)
    def test_init_with_credentials(self):
        """Test initialization with AWS credentials."""
        embedding = BedrockEmbedding()
        self.mock_boto3.client.assert_called_with(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_different_models(self):
        """Test initialization with different models."""
        # Test Titan Text G1
        embedding = BedrockEmbedding(model=MODEL_ID_TITAN_TEXT_G1)
        self.assertEqual(embedding.model, MODEL_ID_TITAN_TEXT_G1)
        
        # Test Cohere English V3
        embedding = BedrockEmbedding(model=MODEL_ID_COHERE_ENGLISH_V3)
        self.assertEqual(embedding.model, MODEL_ID_COHERE_ENGLISH_V3)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_query(self):
        """Test embedding a single query."""
        # Create instance to test
        embedding = BedrockEmbedding()
        
        query = "test query"
        result = embedding.embed_query(query)
        
        # Check that invoke_model was called correctly
        self.mock_client.invoke_model.assert_called_once_with(
            modelId=MODEL_ID_TITAN_TEXT_V2,
            body=json.dumps({"inputText": query})
        )
        
        # Check result
        self.assertEqual(len(result), 1024)
        self.assertEqual(result, [0.1] * 1024)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        # Create instance to test
        embedding = BedrockEmbedding()
        
        texts = ["text 1", "text 2", "text 3"]
        results = embedding.embed_documents(texts)
        
        # Check that invoke_model was called for each text
        self.assertEqual(self.mock_client.invoke_model.call_count, 3)
        for text in texts:
            self.mock_client.invoke_model.assert_any_call(
                modelId=MODEL_ID_TITAN_TEXT_V2,
                body=json.dumps({"inputText": text})
            )
        
        # Check results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(len(result), 1024)
            self.assertEqual(result, [0.1] * 1024)
    
    @patch.dict('os.environ', {}, clear=True)
    def test_dimension_property(self):
        """Test the dimension property for different models."""
        # Create instance to test with Titan Text V2
        embedding = BedrockEmbedding()
        self.assertEqual(embedding.dimension, 1024)
        
        # Test Titan Text G1
        embedding = BedrockEmbedding(model=MODEL_ID_TITAN_TEXT_G1)
        self.assertEqual(embedding.dimension, 1536)
        
        # Test Cohere English V3
        embedding = BedrockEmbedding(model=MODEL_ID_COHERE_ENGLISH_V3)
        self.assertEqual(embedding.dimension, 1024)


if __name__ == "__main__":
    unittest.main() 