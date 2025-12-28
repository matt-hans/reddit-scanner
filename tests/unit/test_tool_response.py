"""Unit tests for ToolResponse envelope class."""
import pytest
import json
from datetime import datetime
from mcp.types import TextContent


class TestToolResponse:
    """Tests for ToolResponse dataclass."""

    def test_default_values(self):
        """ToolResponse should have correct default values."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="test_tool")

        assert response.tool_name == "test_tool"
        assert response.results == []
        assert response.errors == []
        assert response.stats == {}
        assert response.partial is False

    def test_add_result_appends_to_results(self):
        """add_result() should append item to results list."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="test_tool")

        response.add_result({"id": "1", "data": "first"})
        response.add_result({"id": "2", "data": "second"})

        assert len(response.results) == 2
        assert response.results[0] == {"id": "1", "data": "first"}
        assert response.results[1] == {"id": "2", "data": "second"}

    def test_add_error_appends_to_errors_list(self):
        """add_error() should append error dict to errors list."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="test_tool")

        response.add_error("subreddit_xyz", "Subreddit not found")

        assert len(response.errors) == 1
        assert response.errors[0] == {
            "item": "subreddit_xyz",
            "reason": "Subreddit not found"
        }

    def test_add_error_sets_partial_true(self):
        """add_error() should automatically set partial=True."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="test_tool")
        assert response.partial is False

        response.add_error("item1", "Some error")

        assert response.partial is True

    def test_add_error_multiple_times(self):
        """add_error() should accumulate errors and keep partial=True."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="test_tool")

        response.add_error("item1", "Error 1")
        response.add_error("item2", "Error 2")
        response.add_error("item3", "Error 3")

        assert len(response.errors) == 3
        assert response.partial is True

    def test_to_response_returns_text_content_list(self):
        """to_response() should return List[TextContent]."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="test_tool")
        result = response.to_response()

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].type == "text"

    def test_to_response_envelope_structure(self):
        """to_response() should return proper envelope with all required fields."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="test_tool")
        response.add_result({"id": "1", "value": "test"})
        response.stats = {"processed": 1, "skipped": 0}

        result = response.to_response()
        envelope = json.loads(result[0].text)

        # Check top-level structure
        assert "results" in envelope
        assert "metadata" in envelope
        assert "errors" in envelope
        assert "partial" in envelope

        # Check results
        assert envelope["results"] == [{"id": "1", "value": "test"}]

        # Check metadata
        assert envelope["metadata"]["tool"] == "test_tool"
        assert "timestamp" in envelope["metadata"]
        assert envelope["metadata"]["stats"] == {"processed": 1, "skipped": 0}

        # Check errors and partial
        assert envelope["errors"] == []
        assert envelope["partial"] is False

    def test_to_response_with_errors(self):
        """to_response() should include errors and set partial flag correctly."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="scanner")
        response.add_result({"id": "good", "data": "success"})
        response.add_error("bad_item", "Failed to process")

        result = response.to_response()
        envelope = json.loads(result[0].text)

        assert len(envelope["results"]) == 1
        assert len(envelope["errors"]) == 1
        assert envelope["errors"][0]["item"] == "bad_item"
        assert envelope["errors"][0]["reason"] == "Failed to process"
        assert envelope["partial"] is True

    def test_to_response_timestamp_is_valid_iso_format(self):
        """to_response() timestamp should be valid ISO format."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="test_tool")
        result = response.to_response()
        envelope = json.loads(result[0].text)

        timestamp = envelope["metadata"]["timestamp"]
        # Should not raise if valid ISO format
        parsed = datetime.fromisoformat(timestamp)
        assert parsed is not None

    def test_stats_can_be_set_directly(self):
        """stats dictionary should be settable directly."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="test_tool")
        response.stats = {
            "requests_made": 50,
            "budget_remaining": 50,
            "subreddits_processed": 5
        }

        result = response.to_response()
        envelope = json.loads(result[0].text)

        assert envelope["metadata"]["stats"]["requests_made"] == 50
        assert envelope["metadata"]["stats"]["budget_remaining"] == 50
        assert envelope["metadata"]["stats"]["subreddits_processed"] == 5

    def test_empty_response(self):
        """Empty response should have correct structure."""
        from reddit_scanner import ToolResponse

        response = ToolResponse(tool_name="empty_tool")
        result = response.to_response()
        envelope = json.loads(result[0].text)

        assert envelope["results"] == []
        assert envelope["errors"] == []
        assert envelope["partial"] is False
        assert envelope["metadata"]["tool"] == "empty_tool"
        assert envelope["metadata"]["stats"] == {}
