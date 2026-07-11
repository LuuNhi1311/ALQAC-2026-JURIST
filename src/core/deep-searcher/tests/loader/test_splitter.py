import unittest
from langchain_core.documents import Document

from deepsearcher.loader.splitter import Chunk, split_docs_to_chunks, _sentence_window_split


class TestSplitter(unittest.TestCase):
    """Tests for the splitter module."""
    
    def test_chunk_init(self):
        """Test initialization of Chunk class."""
        # Test with minimal parameters
        chunk = Chunk(text="Test text", reference="test_ref")
        self.assertEqual(chunk.text, "Test text")
        self.assertEqual(chunk.reference, "test_ref")
        self.assertEqual(chunk.metadata, {})
        self.assertIsNone(chunk.embedding)
        
        # Test with all parameters
        metadata = {"key": "value"}
        embedding = [0.1, 0.2, 0.3]
        chunk = Chunk(text="Test text", reference="test_ref", metadata=metadata, embedding=embedding)
        self.assertEqual(chunk.text, "Test text")
        self.assertEqual(chunk.reference, "test_ref")
        self.assertEqual(chunk.metadata, metadata)
        self.assertEqual(chunk.embedding, embedding)
    
    def test_sentence_window_split(self):
        """Test _sentence_window_split function."""
        # Create a test document
        original_text = "This is a test document. It has multiple sentences. This is for testing the splitter."
        original_doc = Document(page_content=original_text, metadata={"reference": "test_doc"})
        
        # Create split documents
        split_docs = [
            Document(page_content="This is a test document.", metadata={"reference": "test_doc"}),
            Document(page_content="It has multiple sentences.", metadata={"reference": "test_doc"}),
            Document(page_content="This is for testing the splitter.", metadata={"reference": "test_doc"})
        ]
        
        # Test with default offset
        chunks = _sentence_window_split(split_docs, original_doc)
        
        # Verify the results
        self.assertEqual(len(chunks), 3)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.text, split_docs[i].page_content)
            self.assertEqual(chunk.reference, "test_doc")
            self.assertIn("wider_text", chunk.metadata)
            # The wider text should contain the original text since our test document is short
            self.assertEqual(chunk.metadata["wider_text"], original_text)
        
        # Test with smaller offset
        chunks = _sentence_window_split(split_docs, original_doc, offset=10)
        
        # Verify the results with smaller context windows
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            # With smaller offset, wider_text should be shorter than the full original text
            self.assertLessEqual(len(chunk.metadata["wider_text"]), len(original_text))
    
    def test_split_docs_to_chunks(self):
        """Test split_docs_to_chunks function."""
        # Create test documents
        docs = [
            Document(
                page_content="This is document one. It has some content for testing.",
                metadata={"reference": "doc1"}
            ),
            Document(
                page_content="This is document two. It also has content for testing purposes.",
                metadata={"reference": "doc2"}
            )
        ]
        
        # Test with default parameters
        chunks = split_docs_to_chunks(docs)
        
        # Verify the results
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertIn(chunk.reference, ["doc1", "doc2"])
            self.assertIn("wider_text", chunk.metadata)
        
        # Test with custom chunk size and overlap
        chunks = split_docs_to_chunks(docs, chunk_size=10, chunk_overlap=2)
        
        # With small chunk size, we should get more chunks
        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertIn(chunk.reference, ["doc1", "doc2"])
            self.assertIn("wider_text", chunk.metadata)


if __name__ == "__main__":
    unittest.main() 