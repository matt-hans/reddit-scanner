"""Unit tests for input validation functions."""
import pytest


class TestValidateKeyword:
    """Tests for validate_keyword() function."""

    def test_empty_string_returns_false(self):
        """Empty string should return False."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("") is False

    def test_none_returns_false(self):
        """None should return False."""
        from reddit_scanner import validate_keyword
        assert validate_keyword(None) is False

    def test_non_string_returns_false(self):
        """Non-string types should return False."""
        from reddit_scanner import validate_keyword
        assert validate_keyword(123) is False
        assert validate_keyword([]) is False
        assert validate_keyword({}) is False

    def test_too_short_returns_false(self):
        """Keywords shorter than 2 characters should return False."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("a") is False

    def test_too_long_returns_false(self):
        """Keywords longer than 50 characters should return False."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("a" * 51) is False

    def test_valid_simple_keyword(self):
        """Simple alphanumeric keywords should return True."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("python") is True
        assert validate_keyword("Python3") is True
        assert validate_keyword("ai") is True

    def test_valid_keyword_with_spaces(self):
        """Keywords with spaces should return True."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("python automation") is True
        assert validate_keyword("machine learning tools") is True

    def test_valid_keyword_with_hyphens(self):
        """Keywords with hyphens should return True."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("no-code") is True
        assert validate_keyword("low-code-platform") is True

    def test_valid_keyword_with_periods(self):
        """Keywords with periods should return True."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("node.js") is True
        assert validate_keyword("v2.0") is True

    def test_valid_keyword_with_underscores(self):
        """Keywords with underscores should return True."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("python_dev") is True
        assert validate_keyword("my_keyword") is True

    def test_boundary_length_2_chars(self):
        """2-character keyword (minimum) should return True."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("ab") is True

    def test_boundary_length_50_chars(self):
        """50-character keyword (maximum) should return True."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("a" * 50) is True

    def test_invalid_special_characters(self):
        """Keywords with invalid special characters should return False."""
        from reddit_scanner import validate_keyword
        assert validate_keyword("python!") is False
        assert validate_keyword("test@keyword") is False
        assert validate_keyword("hello#world") is False
        assert validate_keyword("price$value") is False
        assert validate_keyword("test%complete") is False
        assert validate_keyword("path/to/file") is False


class TestValidateKeywordList:
    """Tests for validate_keyword_list() function."""

    def test_empty_list_returns_false(self):
        """Empty list should return False."""
        from reddit_scanner import validate_keyword_list
        assert validate_keyword_list([]) is False

    def test_non_list_returns_false(self):
        """Non-list types should return False."""
        from reddit_scanner import validate_keyword_list
        assert validate_keyword_list(None) is False
        assert validate_keyword_list("keyword") is False
        assert validate_keyword_list(123) is False
        assert validate_keyword_list({"key": "value"}) is False

    def test_list_with_invalid_keyword_returns_false(self):
        """List containing invalid keywords should return False."""
        from reddit_scanner import validate_keyword_list
        assert validate_keyword_list(["valid", ""]) is False
        assert validate_keyword_list(["valid", "a"]) is False  # too short
        assert validate_keyword_list(["valid", "test!"]) is False  # invalid char

    def test_list_with_non_string_returns_false(self):
        """List containing non-string should return False."""
        from reddit_scanner import validate_keyword_list
        assert validate_keyword_list(["valid", 123]) is False
        assert validate_keyword_list([None, "valid"]) is False

    def test_valid_single_keyword_list(self):
        """List with single valid keyword should return True."""
        from reddit_scanner import validate_keyword_list
        assert validate_keyword_list(["python"]) is True

    def test_valid_multiple_keywords_list(self):
        """List with multiple valid keywords should return True."""
        from reddit_scanner import validate_keyword_list
        assert validate_keyword_list(["python", "automation"]) is True
        assert validate_keyword_list(["node.js", "no-code", "api tools"]) is True

    def test_tuple_returns_false(self):
        """Tuple should return False (must be list)."""
        from reddit_scanner import validate_keyword_list
        assert validate_keyword_list(("python", "automation")) is False


class TestValidateBatchDelay:
    """Tests for validate_batch_delay() function."""

    def test_below_minimum_returns_false(self):
        """Delay below 0.5 should return False."""
        from reddit_scanner import validate_batch_delay
        assert validate_batch_delay(0.4) is False
        assert validate_batch_delay(0) is False
        assert validate_batch_delay(-1) is False

    def test_above_maximum_returns_false(self):
        """Delay above 10.0 should return False."""
        from reddit_scanner import validate_batch_delay
        assert validate_batch_delay(10.1) is False
        assert validate_batch_delay(100) is False

    def test_minimum_boundary(self):
        """Delay at 0.5 (minimum) should return True."""
        from reddit_scanner import validate_batch_delay
        assert validate_batch_delay(0.5) is True

    def test_maximum_boundary(self):
        """Delay at 10.0 (maximum) should return True."""
        from reddit_scanner import validate_batch_delay
        assert validate_batch_delay(10.0) is True

    def test_valid_float_in_range(self):
        """Valid float values in range should return True."""
        from reddit_scanner import validate_batch_delay
        assert validate_batch_delay(1.0) is True
        assert validate_batch_delay(5.5) is True
        assert validate_batch_delay(2.75) is True

    def test_valid_int_in_range(self):
        """Valid integer values in range should return True."""
        from reddit_scanner import validate_batch_delay
        assert validate_batch_delay(1) is True
        assert validate_batch_delay(5) is True
        assert validate_batch_delay(10) is True

    def test_non_numeric_returns_false(self):
        """Non-numeric types should return False."""
        from reddit_scanner import validate_batch_delay
        assert validate_batch_delay("1.0") is False
        assert validate_batch_delay(None) is False
        assert validate_batch_delay([1.0]) is False
        assert validate_batch_delay({"delay": 1.0}) is False
