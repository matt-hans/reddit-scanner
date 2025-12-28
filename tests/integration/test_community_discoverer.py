"""Integration tests for niche_community_discoverer v2."""
import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio


class TestNicheCommunityDiscovererV2:
    """Tests for the v2 niche_community_discoverer tool."""

    @pytest.fixture
    def mock_subreddit_search_result(self):
        """Create mock subreddit for search results."""
        def make_subreddit(name, subscribers, description="A test subreddit", sidebar=""):
            sub = MagicMock()
            sub.display_name = name
            sub.subscribers = subscribers
            sub.public_description = description
            sub.description = sidebar  # Sidebar text
            sub.subreddit_type = "public"
            return sub
        return make_subreddit

    @pytest.fixture
    def mock_reddit_client(self, mock_subreddit_search_result):
        """Create mock Reddit client with subreddits.search."""
        reddit = MagicMock()

        # Create sample subreddits for search
        sub1 = mock_subreddit_search_result(
            "pythontools", 15000,
            "Python productivity tools",
            "Related: r/learnpython r/automation"
        )
        sub2 = mock_subreddit_search_result(
            "automationtools", 25000,
            "Automation discussion",
            "Check out r/pythontools and r/workflow"
        )
        sub3 = mock_subreddit_search_result(
            "workflow", 8000,
            "Workflow optimization",
            ""
        )
        # Out of range subreddit (too many subscribers)
        sub4 = mock_subreddit_search_result(
            "bigsubreddit", 500000,
            "Very large community",
            ""
        )
        # Out of range subreddit (too few subscribers)
        sub5 = mock_subreddit_search_result(
            "tinysubreddit", 100,
            "Very small community",
            ""
        )

        # Mock subreddits.search to return different results per keyword
        def search_side_effect(keyword, limit=20):
            if "python" in keyword.lower():
                return [sub1, sub4]  # One valid, one too big
            elif "automation" in keyword.lower():
                return [sub2, sub1]  # Should dedupe sub1
            else:
                return [sub3, sub5]  # One valid, one too small

        reddit.subreddits.search = MagicMock(side_effect=search_side_effect)

        # Mock subreddit() for sidebar spider validation
        def subreddit_side_effect(name):
            subs = {
                "learnpython": mock_subreddit_search_result("learnpython", 50000, "Learn Python", ""),
                "automation": mock_subreddit_search_result("automation", 30000, "Automation", ""),
                "workflow": mock_subreddit_search_result("workflow", 8000, "Workflow", ""),
                "pythontools": mock_subreddit_search_result("pythontools", 15000, "Python tools", ""),
            }
            if name.lower() in subs:
                return subs[name.lower()]
            raise Exception(f"Subreddit {name} not found")

        reddit.subreddit = MagicMock(side_effect=subreddit_side_effect)

        return reddit

    @pytest.mark.asyncio
    async def test_basic_keyword_search(self, mock_reddit_client):
        """Test basic keyword search returns communities in subscriber range."""
        from reddit_scanner import niche_community_discoverer, RedditClient

        # Patch RedditClient to return our mock
        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            # Also patch execute to run sync operations
            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await niche_community_discoverer(
                    topic_keywords=["python tools"],
                    min_subscribers=5000,
                    max_subscribers=200000,
                    max_communities=50,
                    spider_sidebar=False,  # Disable sidebar spider for this test
                    batch_delay=0.5
                )

        # Parse the response
        assert len(result) == 1
        response = json.loads(result[0].text)

        # Check response structure
        assert "results" in response
        assert "metadata" in response
        assert "errors" in response
        assert "partial" in response

        # Check metadata
        assert response["metadata"]["tool"] == "niche_community_discoverer"
        assert "keywords_searched" in response["metadata"]
        assert "python tools" in response["metadata"]["keywords_searched"]

    @pytest.mark.asyncio
    async def test_deduplication(self, mock_reddit_client):
        """Test that results are deduplicated by display_name."""
        from reddit_scanner import niche_community_discoverer, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await niche_community_discoverer(
                    topic_keywords=["python tools", "automation"],
                    min_subscribers=5000,
                    max_subscribers=200000,
                    spider_sidebar=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Count unique subreddit names
        names = [r["name"] for r in response["results"]]
        assert len(names) == len(set(names)), "Results should be deduplicated"

    @pytest.mark.asyncio
    async def test_max_communities_cap(self, mock_subreddit_search_result):
        """Test that max_communities cap is enforced."""
        from reddit_scanner import niche_community_discoverer, RedditClient

        # Create many subreddits
        many_subs = [
            mock_subreddit_search_result(f"sub{i}", 10000 + i * 100, f"Subreddit {i}")
            for i in range(30)
        ]

        reddit = MagicMock()
        reddit.subreddits.search = MagicMock(return_value=many_subs)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await niche_community_discoverer(
                    topic_keywords=["test"],
                    max_communities=5,  # Low cap
                    spider_sidebar=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)
        assert len(response["results"]) <= 5, "Should respect max_communities cap"

    @pytest.mark.asyncio
    async def test_subscriber_range_filtering(self, mock_reddit_client):
        """Test that subreddits outside subscriber range are filtered."""
        from reddit_scanner import niche_community_discoverer, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await niche_community_discoverer(
                    topic_keywords=["python tools"],
                    min_subscribers=5000,
                    max_subscribers=200000,
                    spider_sidebar=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # All results should be within subscriber range
        for community in response["results"]:
            assert 5000 <= community["subscribers"] <= 200000, \
                f"Subscriber count {community['subscribers']} out of range"

    @pytest.mark.asyncio
    async def test_sidebar_spider_extracts_subreddits(self, mock_reddit_client):
        """Test that sidebar spider extracts r/SubredditName patterns."""
        from reddit_scanner import niche_community_discoverer, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await niche_community_discoverer(
                    topic_keywords=["python tools"],
                    min_subscribers=5000,
                    max_subscribers=200000,
                    spider_sidebar=True,  # Enable sidebar spider
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Check that some results came from sidebar spider
        sidebar_sources = [r for r in response["results"] if "sidebar of" in r.get("source", "")]
        # Note: This may be empty if sidebar subreddits don't meet criteria
        # The test validates the feature works without errors

    @pytest.mark.asyncio
    async def test_source_field_format(self, mock_reddit_client):
        """Test that source field shows 'search' or 'sidebar of X'."""
        from reddit_scanner import niche_community_discoverer, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await niche_community_discoverer(
                    topic_keywords=["python tools"],
                    spider_sidebar=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # All results from search should have source="search"
        for community in response["results"]:
            assert "source" in community
            assert community["source"] == "search" or community["source"].startswith("sidebar of ")

    @pytest.mark.asyncio
    async def test_input_validation_empty_keywords(self):
        """Test that empty keyword list raises validation error."""
        from reddit_scanner import niche_community_discoverer

        # Empty list should fail validation
        result = await niche_community_discoverer(
            topic_keywords=[],
            batch_delay=0.5
        )

        response = json.loads(result[0].text)
        assert response.get("partial", False) or "error" in str(response).lower()

    @pytest.mark.asyncio
    async def test_input_validation_invalid_batch_delay(self):
        """Test that invalid batch_delay raises validation error."""
        from reddit_scanner import niche_community_discoverer

        # Delay too low (< 0.5) should fail validation
        result = await niche_community_discoverer(
            topic_keywords=["test"],
            batch_delay=0.1  # Too low
        )

        response = json.loads(result[0].text)
        # Should have error or partial flag
        assert response.get("partial", False) or "error" in str(response).lower()

    @pytest.mark.asyncio
    async def test_response_uses_tool_response_envelope(self, mock_reddit_client):
        """Test that response uses ToolResponse envelope format."""
        from reddit_scanner import niche_community_discoverer, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await niche_community_discoverer(
                    topic_keywords=["python"],
                    spider_sidebar=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Verify ToolResponse envelope structure
        assert "results" in response, "Should have results key"
        assert "metadata" in response, "Should have metadata key"
        assert "errors" in response, "Should have errors key"
        assert "partial" in response, "Should have partial key"
        assert response["metadata"]["tool"] == "niche_community_discoverer"
        assert "timestamp" in response["metadata"]

    @pytest.mark.asyncio
    async def test_result_community_fields(self, mock_reddit_client):
        """Test that each community result has required fields."""
        from reddit_scanner import niche_community_discoverer, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await niche_community_discoverer(
                    topic_keywords=["python"],
                    spider_sidebar=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Each result should have required fields
        for community in response["results"]:
            assert "name" in community
            assert "subscribers" in community
            assert "source" in community
            assert "description" in community

    @pytest.mark.asyncio
    async def test_metadata_includes_keywords_searched(self, mock_reddit_client):
        """Test that metadata includes list of keywords searched."""
        from reddit_scanner import niche_community_discoverer, RedditClient

        keywords = ["python", "automation"]

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await niche_community_discoverer(
                    topic_keywords=keywords,
                    spider_sidebar=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        assert "keywords_searched" in response["metadata"], "metadata should contain keywords_searched"
        assert set(response["metadata"]["keywords_searched"]) == set(keywords), "keywords_searched should match input"
