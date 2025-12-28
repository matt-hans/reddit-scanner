"""Shared pytest fixtures for Reddit Scanner tests."""
import os
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Ensure we can import from the main module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_reddit():
    """Create a mock Reddit instance."""
    reddit = MagicMock()
    reddit.subreddit = MagicMock(return_value=MagicMock())
    reddit.subreddits = MagicMock()
    reddit.submission = MagicMock(return_value=MagicMock())
    return reddit


@pytest.fixture
def mock_subreddit():
    """Create a mock subreddit with common attributes."""
    sub = MagicMock()
    sub.display_name = "testsubreddit"
    sub.subscribers = 50000
    sub.public_description = "A test subreddit for testing"
    sub.description = "Sidebar text with r/relatedcommunity links"
    return sub


@pytest.fixture
def mock_post():
    """Create a mock Reddit post."""
    post = MagicMock()
    post.id = "abc123"
    post.title = "Test post title"
    post.selftext = "Test post content with frustrated and annoying keywords"
    post.score = 100
    post.upvote_ratio = 0.95
    post.num_comments = 50
    post.created_utc = 1703620800.0  # Fixed timestamp
    post.permalink = "/r/test/comments/abc123/test_post/"
    post.author = MagicMock()
    post.author.name = "testuser"
    return post


@pytest.fixture
def mock_comment():
    """Create a mock Reddit comment."""
    comment = MagicMock()
    comment.id = "comment123"
    comment.body = "Test comment with step by step process and manual work"
    comment.score = 25
    comment.depth = 2
    comment.created_utc = 1703620900.0
    comment.permalink = "/r/test/comments/abc123/test_post/comment123/"
    comment.author = MagicMock()
    comment.author.name = "commenter"
    return comment


@pytest.fixture
def known_subreddit():
    """Subreddit guaranteed to exist for integration testing."""
    return "learnpython"


@pytest.fixture
def sample_topic_keywords():
    """Sample topic keywords for discovery testing."""
    return ["python automation", "productivity tools"]


@pytest.fixture
def sample_workflow_signals():
    """Sample workflow signals for thread inspection."""
    return ["step", "process", "manual", "export", "click"]
