import unittest
from deepsearcher.llm.base import BaseLLM, ChatResponse
from unittest.mock import patch


class TestBaseLLM(unittest.TestCase):
    """Tests for the BaseLLM abstract base class."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear environment variables temporarily
        self.env_patcher = patch.dict('os.environ', {}, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()

    def test_chat_response_init(self):
        """Test ChatResponse initialization and representation."""
        content = "Test content"
        total_tokens = 100
        response = ChatResponse(content=content, total_tokens=total_tokens)
        
        self.assertEqual(response.content, content)
        self.assertEqual(response.total_tokens, total_tokens)
        self.assertEqual(
            repr(response),
            f"ChatResponse(content={content}, total_tokens={total_tokens})"
        )

    def test_literal_eval_python_code_block(self):
        """Test literal_eval with Python code block."""
        content = '''```python
{"key": "value", "number": 42}
```'''
        result = BaseLLM.literal_eval(content)
        self.assertEqual(result, {"key": "value", "number": 42})

    def test_literal_eval_json_code_block(self):
        """Test literal_eval with JSON code block."""
        content = '''```json
{"key": "value", "number": 42}
```'''
        result = BaseLLM.literal_eval(content)
        self.assertEqual(result, {"key": "value", "number": 42})

    def test_literal_eval_str_code_block(self):
        """Test literal_eval with str code block."""
        content = '''```str
{"key": "value", "number": 42}
```'''
        result = BaseLLM.literal_eval(content)
        self.assertEqual(result, {"key": "value", "number": 42})

    def test_literal_eval_plain_code_block(self):
        """Test literal_eval with plain code block."""
        content = '''```
{"key": "value", "number": 42}
```'''
        result = BaseLLM.literal_eval(content)
        self.assertEqual(result, {"key": "value", "number": 42})

    def test_literal_eval_raw_dict(self):
        """Test literal_eval with raw dictionary string."""
        content = '{"key": "value", "number": 42}'
        result = BaseLLM.literal_eval(content)
        self.assertEqual(result, {"key": "value", "number": 42})

    def test_literal_eval_raw_list(self):
        """Test literal_eval with raw list string."""
        content = '[1, 2, "three", {"four": 4}]'
        result = BaseLLM.literal_eval(content)
        self.assertEqual(result, [1, 2, "three", {"four": 4}])

    def test_literal_eval_with_whitespace(self):
        """Test literal_eval with extra whitespace."""
        content = '''
        
        {"key": "value"}
        
        '''
        result = BaseLLM.literal_eval(content)
        self.assertEqual(result, {"key": "value"})

    def test_literal_eval_nested_structures(self):
        """Test literal_eval with nested data structures."""
        content = '''
        {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "mixed": [1, {"key": "value"}, [2, 3]]
        }
        '''
        result = BaseLLM.literal_eval(content)
        expected = {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "mixed": [1, {"key": "value"}, [2, 3]]
        }
        self.assertEqual(result, expected)

    def test_literal_eval_invalid_format(self):
        """Test literal_eval with invalid format."""
        invalid_contents = [
            "Not a valid Python literal",
            "{invalid: json}",
            "[1, 2, 3",  # Unclosed bracket
            '{"key": undefined}',  # undefined is not a valid Python literal
        ]
        for content in invalid_contents:
            with self.assertRaises(ValueError):
                BaseLLM.literal_eval(content)

    def test_remove_think_with_tags(self):
        """Test remove_think with think tags."""
        content = '''<think>
        This is the reasoning process.
        Multiple lines of thought.
        </think>
        This is the actual response.'''
        result = BaseLLM.remove_think(content)
        self.assertEqual(result.strip(), "This is the actual response.")

    def test_remove_think_without_tags(self):
        """Test remove_think without think tags."""
        content = "This is a response without think tags."
        result = BaseLLM.remove_think(content)
        self.assertEqual(result.strip(), content.strip())

    def test_remove_think_multiple_tags(self):
        """Test remove_think with multiple think tags - should only remove first block."""
        content = '''<think>First think block</think>
        Actual response
        <think>Second think block</think>'''
        result = BaseLLM.remove_think(content)
        self.assertEqual(
            result.strip(),
            "Actual response\n        <think>Second think block</think>"
        )

    def test_remove_think_empty_tags(self):
        """Test remove_think with empty think tags."""
        content = "<think></think>Response"
        result = BaseLLM.remove_think(content)
        self.assertEqual(result.strip(), "Response")


if __name__ == "__main__":
    unittest.main() 