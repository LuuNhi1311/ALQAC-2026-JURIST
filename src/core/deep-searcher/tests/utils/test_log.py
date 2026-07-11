import unittest
from unittest.mock import patch, MagicMock, call
import logging
from termcolor import colored

from deepsearcher.utils import log
from deepsearcher.utils.log import ColoredFormatter


class TestColoredFormatter(unittest.TestCase):
    """Tests for the ColoredFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ColoredFormatter("%(levelname)s - %(message)s")

    def test_format_debug(self):
        """Test formatting debug level messages."""
        record = logging.LogRecord(
            "test", logging.DEBUG, "test.py", 1, "Debug message", (), None
        )
        formatted = self.formatter.format(record)
        expected = colored("DEBUG - Debug message", "cyan")
        self.assertEqual(formatted, expected)

    def test_format_info(self):
        """Test formatting info level messages."""
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "Info message", (), None
        )
        formatted = self.formatter.format(record)
        expected = colored("INFO - Info message", "green")
        self.assertEqual(formatted, expected)

    def test_format_warning(self):
        """Test formatting warning level messages."""
        record = logging.LogRecord(
            "test", logging.WARNING, "test.py", 1, "Warning message", (), None
        )
        formatted = self.formatter.format(record)
        expected = colored("WARNING - Warning message", "yellow")
        self.assertEqual(formatted, expected)

    def test_format_error(self):
        """Test formatting error level messages."""
        record = logging.LogRecord(
            "test", logging.ERROR, "test.py", 1, "Error message", (), None
        )
        formatted = self.formatter.format(record)
        expected = colored("ERROR - Error message", "red")
        self.assertEqual(formatted, expected)

    def test_format_critical(self):
        """Test formatting critical level messages."""
        record = logging.LogRecord(
            "test", logging.CRITICAL, "test.py", 1, "Critical message", (), None
        )
        formatted = self.formatter.format(record)
        expected = colored("CRITICAL - Critical message", "magenta")
        self.assertEqual(formatted, expected)

    def test_format_unknown_level(self):
        """Test formatting messages with unknown log level."""
        record = logging.LogRecord(
            "test", 60, "test.py", 1, "Custom level message", (), None
        )
        record.levelname = "CUSTOM"
        formatted = self.formatter.format(record)
        expected = colored("CUSTOM - Custom level message", "white")
        self.assertEqual(formatted, expected)


class TestLogFunctions(unittest.TestCase):
    """Tests for the logging functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset dev mode before each test
        log.set_dev_mode(False)
        
        # Create mock for dev_logger
        self.mock_dev_logger = MagicMock()
        self.dev_logger_patcher = patch("deepsearcher.utils.log.dev_logger", self.mock_dev_logger)
        self.dev_logger_patcher.start()
        
        # Create mock for progress_logger
        self.mock_progress_logger = MagicMock()
        self.progress_logger_patcher = patch("deepsearcher.utils.log.progress_logger", self.mock_progress_logger)
        self.progress_logger_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.dev_logger_patcher.stop()
        self.progress_logger_patcher.stop()

    def test_set_dev_mode(self):
        """Test setting development mode."""
        self.assertFalse(log.dev_mode)
        log.set_dev_mode(True)
        self.assertTrue(log.dev_mode)
        log.set_dev_mode(False)
        self.assertFalse(log.dev_mode)

    def test_set_level(self):
        """Test setting log level."""
        log.set_level(logging.DEBUG)
        self.mock_dev_logger.setLevel.assert_called_once_with(logging.DEBUG)

    def test_debug_in_dev_mode(self):
        """Test debug logging in dev mode."""
        log.set_dev_mode(True)
        log.debug("Test debug")
        self.mock_dev_logger.debug.assert_called_once_with("Test debug")

    def test_debug_not_in_dev_mode(self):
        """Test debug logging not in dev mode."""
        log.set_dev_mode(False)
        log.debug("Test debug")
        self.mock_dev_logger.debug.assert_not_called()

    def test_info_in_dev_mode(self):
        """Test info logging in dev mode."""
        log.set_dev_mode(True)
        log.info("Test info")
        self.mock_dev_logger.info.assert_called_once_with("Test info")

    def test_info_not_in_dev_mode(self):
        """Test info logging not in dev mode."""
        log.set_dev_mode(False)
        log.info("Test info")
        self.mock_dev_logger.info.assert_not_called()

    def test_warning_in_dev_mode(self):
        """Test warning logging in dev mode."""
        log.set_dev_mode(True)
        log.warning("Test warning")
        self.mock_dev_logger.warning.assert_called_once_with("Test warning")

    def test_warning_not_in_dev_mode(self):
        """Test warning logging not in dev mode."""
        log.set_dev_mode(False)
        log.warning("Test warning")
        self.mock_dev_logger.warning.assert_not_called()

    def test_error_in_dev_mode(self):
        """Test error logging in dev mode."""
        log.set_dev_mode(True)
        log.error("Test error")
        self.mock_dev_logger.error.assert_called_once_with("Test error")

    def test_error_not_in_dev_mode(self):
        """Test error logging not in dev mode."""
        log.set_dev_mode(False)
        log.error("Test error")
        self.mock_dev_logger.error.assert_not_called()

    def test_critical(self):
        """Test critical logging and exception raising."""
        with self.assertRaises(RuntimeError) as context:
            log.critical("Test critical")
        
        self.mock_dev_logger.critical.assert_called_once_with("Test critical")
        self.assertEqual(str(context.exception), "Test critical")

    def test_color_print(self):
        """Test color print function."""
        log.color_print("Test message")
        self.mock_progress_logger.info.assert_called_once_with("Test message")


if __name__ == "__main__":
    unittest.main() 