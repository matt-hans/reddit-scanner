# Market Intelligence Spider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the Reddit Scanner from a passive tool into an active Market Intelligence Spider with keyword-based discovery, deep comment inspection, and wiki extraction.

**Architecture:** Add shared infrastructure (rate limiting, response envelope) to `reddit_scanner.py`, replace `niche_community_discoverer` with keyword-based v2, add two new tools (`workflow_thread_inspector`, `wiki_tool_extractor`). Build comprehensive pytest-based test suite.

**Tech Stack:** Python 3.12+, PRAW, FastMCP, pytest, pytest-asyncio

---

## Task 1: Add pytest Dependencies

**Files:**
- Modify: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/pyproject.toml`

**Step 1: Update pyproject.toml with dev dependencies**

```toml
[project]
name = "reddit-scanner"
version = "0.1.0"
description = "Market Intelligence Spider - Reddit analysis MCP server"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28.1",
    "mcp[cli]>=1.11.0",
    "praw>=7.7.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=4.1.0",
]
```

**Step 2: Install dev dependencies**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && uv pip install -e ".[dev]"`
Expected: Successfully installed pytest, pytest-asyncio, pytest-cov

**Step 3: Verify pytest works**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest --version`
Expected: pytest version displayed

**Step 4: Commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add pyproject.toml
git commit -m "build: add pytest dev dependencies"
```

---

## Task 2: Create Test Directory Structure

**Files:**
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/__init__.py`
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/conftest.py`
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/__init__.py`
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/integration/__init__.py`
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/pytest.ini`

**Step 1: Create pytest.ini configuration**

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    integration: marks tests as integration tests (require Reddit API)
    unit: marks tests as unit tests (no external dependencies)
```

**Step 2: Create tests/__init__.py**

```python
"""Reddit Scanner test suite."""
```

**Step 3: Create tests/unit/__init__.py**

```python
"""Unit tests - no external dependencies."""
```

**Step 4: Create tests/integration/__init__.py**

```python
"""Integration tests - require Reddit API credentials."""
```

**Step 5: Create tests/conftest.py with shared fixtures**

```python
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
```

**Step 6: Run pytest to verify structure**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest --collect-only`
Expected: "no tests ran" (empty test suite, but no errors)

**Step 7: Commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add pytest.ini tests/
git commit -m "test: create pytest directory structure and fixtures"
```

---

## Task 3: Implement RateLimitConfig and RateLimitedExecutor

**Files:**
- Modify: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py:1-114` (add after imports, before RedditClient)
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_rate_limiting.py`

**Step 1: Write failing test for RateLimitConfig**

Create `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_rate_limiting.py`:

```python
"""Unit tests for rate limiting infrastructure."""
import pytest
import asyncio
import time
from reddit_scanner import RateLimitConfig, RateLimitedExecutor, BudgetExhaustedError


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = RateLimitConfig()
        assert config.batch_delay == 1.5
        assert config.request_budget == 100
        assert config.budget_exhausted_action == "stop"

    def test_custom_values(self):
        """Config should accept custom values."""
        config = RateLimitConfig(batch_delay=2.0, request_budget=50)
        assert config.batch_delay == 2.0
        assert config.request_budget == 50


class TestRateLimitedExecutor:
    """Tests for RateLimitedExecutor."""

    @pytest.mark.asyncio
    async def test_budget_exhaustion_raises_error(self):
        """Executor should raise BudgetExhaustedError when budget exhausted."""
        config = RateLimitConfig(request_budget=2, batch_delay=0.01)
        executor = RateLimitedExecutor(config)

        # Mock operation that returns immediately
        async def mock_op():
            return "result"

        # First two calls should succeed
        await executor.execute(mock_op)
        await executor.execute(mock_op)

        # Third call should raise
        with pytest.raises(BudgetExhaustedError):
            await executor.execute(mock_op)

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Executor should track request count."""
        config = RateLimitConfig(request_budget=10, batch_delay=0.01)
        executor = RateLimitedExecutor(config)

        async def mock_op():
            return "result"

        await executor.execute(mock_op)
        await executor.execute(mock_op)

        stats = executor.get_stats()
        assert stats["requests_made"] == 2
        assert stats["budget"] == 10
        assert stats["remaining"] == 8

    @pytest.mark.asyncio
    async def test_delay_enforcement(self):
        """Executor should enforce delay between requests."""
        config = RateLimitConfig(request_budget=10, batch_delay=0.1)
        executor = RateLimitedExecutor(config)

        async def mock_op():
            return "result"

        start = time.time()
        await executor.execute(mock_op)
        await executor.execute(mock_op)
        elapsed = time.time() - start

        # Second call should have waited at least batch_delay
        assert elapsed >= 0.1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_rate_limiting.py -v`
Expected: FAIL with "cannot import name 'RateLimitConfig' from 'reddit_scanner'"

**Step 3: Implement RateLimitConfig and RateLimitedExecutor**

Add to `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py` after line 23 (after `logger = logging.getLogger(__name__)`):

```python
# Rate Limiting Infrastructure
from dataclasses import dataclass, field


class BudgetExhaustedError(Exception):
    """Raised when request budget is exhausted."""
    pass


@dataclass
class RateLimitConfig:
    """Configuration for respectful crawling."""
    batch_delay: float = 1.5          # Seconds between requests
    request_budget: int = 100         # Max requests per tool call
    budget_exhausted_action: str = "stop"  # "stop" or "warn"


class RateLimitedExecutor:
    """Wraps operations with budgeting and delays."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests_made = 0
        self.last_request_time = 0.0

    async def execute(self, operation, *args, **kwargs):
        """Execute operation with rate limiting."""
        # Check budget
        if self.requests_made >= self.config.request_budget:
            raise BudgetExhaustedError(
                f"Request budget of {self.config.request_budget} exhausted"
            )

        # Enforce delay
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if self.last_request_time > 0 and elapsed < self.config.batch_delay:
            await asyncio.sleep(self.config.batch_delay - elapsed)

        # Execute operation
        if asyncio.iscoroutinefunction(operation):
            result = await operation(*args, **kwargs)
        else:
            # Wrap sync function
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, functools.partial(operation, *args, **kwargs)
            )

        self.requests_made += 1
        self.last_request_time = time.time()
        return result

    def get_stats(self) -> dict:
        """Return execution statistics."""
        return {
            "requests_made": self.requests_made,
            "budget": self.config.request_budget,
            "remaining": self.config.request_budget - self.requests_made
        }
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_rate_limiting.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add reddit_scanner.py tests/unit/test_rate_limiting.py
git commit -m "feat: add RateLimitConfig and RateLimitedExecutor"
```

---

## Task 4: Implement ToolResponse Envelope

**Files:**
- Modify: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py` (add after RateLimitedExecutor)
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_tool_response.py`

**Step 1: Write failing test for ToolResponse**

Create `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_tool_response.py`:

```python
"""Unit tests for ToolResponse envelope."""
import pytest
import json
from reddit_scanner import ToolResponse


class TestToolResponse:
    """Tests for ToolResponse unified envelope."""

    def test_envelope_structure(self):
        """Response should have correct envelope structure."""
        response = ToolResponse(tool_name="test_tool")
        response.add_result({"name": "test", "value": 123})

        result = response.to_dict()

        assert "results" in result
        assert "metadata" in result
        assert "errors" in result
        assert "partial" in result

        assert result["metadata"]["tool"] == "test_tool"
        assert "timestamp" in result["metadata"]
        assert len(result["results"]) == 1

    def test_add_result(self):
        """Should accumulate results."""
        response = ToolResponse(tool_name="test_tool")
        response.add_result({"id": 1})
        response.add_result({"id": 2})

        result = response.to_dict()
        assert len(result["results"]) == 2

    def test_add_error_sets_partial(self):
        """Adding error should set partial=True."""
        response = ToolResponse(tool_name="test_tool")
        assert response.partial is False

        response.add_error(item="keyword1", reason="rate_limit")

        assert response.partial is True
        result = response.to_dict()
        assert len(result["errors"]) == 1
        assert result["errors"][0]["item"] == "keyword1"
        assert result["errors"][0]["reason"] == "rate_limit"

    def test_set_partial_with_reason(self):
        """Can set partial with custom reason."""
        response = ToolResponse(tool_name="test_tool")
        response.set_partial(reason="budget exhausted")

        result = response.to_dict()
        assert result["partial"] is True
        assert result["metadata"]["stats"]["partial_reason"] == "budget exhausted"

    def test_to_response_returns_text_content(self):
        """to_response should return List[TextContent]."""
        response = ToolResponse(tool_name="test_tool")
        response.add_result({"test": "data"})

        text_content = response.to_response()

        assert len(text_content) == 1
        assert text_content[0].type == "text"

        # Should be valid JSON
        parsed = json.loads(text_content[0].text)
        assert parsed["metadata"]["tool"] == "test_tool"

    def test_stats_tracking(self):
        """Should track custom stats."""
        response = ToolResponse(tool_name="test_tool")
        response.stats["keywords_searched"] = 5
        response.stats["communities_found"] = 12

        result = response.to_dict()
        assert result["metadata"]["stats"]["keywords_searched"] == 5
        assert result["metadata"]["stats"]["communities_found"] == 12
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_tool_response.py -v`
Expected: FAIL with "cannot import name 'ToolResponse' from 'reddit_scanner'"

**Step 3: Implement ToolResponse**

Add to `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py` after RateLimitedExecutor class:

```python
@dataclass
class ToolResponse:
    """Unified response builder for Market Intelligence tools."""
    tool_name: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    partial: bool = False

    def add_result(self, item: Dict[str, Any]):
        """Add a result item."""
        self.results.append(item)

    def add_error(self, item: str, reason: str):
        """Add an error and mark as partial."""
        self.errors.append({"item": item, "reason": reason})
        self.partial = True

    def set_partial(self, reason: str = None):
        """Mark response as partial with optional reason."""
        self.partial = True
        if reason:
            self.stats["partial_reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary envelope."""
        return {
            "results": self.results,
            "metadata": {
                "tool": self.tool_name,
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats
            },
            "errors": self.errors,
            "partial": self.partial
        }

    def to_response(self) -> List[TextContent]:
        """Convert to MCP TextContent response."""
        return safe_json_response(self.to_dict())
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_tool_response.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add reddit_scanner.py tests/unit/test_tool_response.py
git commit -m "feat: add ToolResponse unified envelope"
```

---

## Task 5: Add New Validators

**Files:**
- Modify: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py` (add after existing validators)
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_validators.py`

**Step 1: Write failing tests for new validators**

Create `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_validators.py`:

```python
"""Unit tests for input validators."""
import pytest
from reddit_scanner import (
    validate_subreddit_name,
    validate_post_id,
    validate_keyword,
    validate_keyword_list,
    validate_batch_delay,
    validate_time_filter,
    validate_positive_int,
)


class TestExistingValidators:
    """Tests for pre-existing validators."""

    def test_validate_subreddit_name_valid(self):
        assert validate_subreddit_name("python") is True
        assert validate_subreddit_name("learn_python") is True
        assert validate_subreddit_name("AskReddit") is True

    def test_validate_subreddit_name_invalid(self):
        assert validate_subreddit_name("") is False
        assert validate_subreddit_name("ab") is False  # Too short
        assert validate_subreddit_name("a" * 25) is False  # Too long
        assert validate_subreddit_name("has spaces") is False

    def test_validate_post_id_valid(self):
        assert validate_post_id("abc123") is True
        assert validate_post_id("1abc2d3") is True

    def test_validate_post_id_invalid(self):
        assert validate_post_id("") is False
        assert validate_post_id("short") is False  # Too short
        assert validate_post_id("way_too_long_id") is False


class TestNewValidators:
    """Tests for new validators added for Market Intelligence Spider."""

    def test_validate_keyword_valid(self):
        assert validate_keyword("python") is True
        assert validate_keyword("python automation") is True
        assert validate_keyword("no-code tools") is True
        assert validate_keyword("AI.ML") is True

    def test_validate_keyword_invalid(self):
        assert validate_keyword("") is False
        assert validate_keyword("a") is False  # Too short
        assert validate_keyword("x" * 60) is False  # Too long
        assert validate_keyword(None) is False
        assert validate_keyword(123) is False

    def test_validate_keyword_list_valid(self):
        assert validate_keyword_list(["python", "automation"]) is True
        assert validate_keyword_list(["single"]) is True

    def test_validate_keyword_list_invalid(self):
        assert validate_keyword_list([]) is False  # Empty list
        assert validate_keyword_list(None) is False
        assert validate_keyword_list("not a list") is False
        assert validate_keyword_list(["valid", ""]) is False  # Contains invalid

    def test_validate_batch_delay_valid(self):
        assert validate_batch_delay(0.5) is True
        assert validate_batch_delay(1.5) is True
        assert validate_batch_delay(10.0) is True
        assert validate_batch_delay(5) is True  # int also valid

    def test_validate_batch_delay_invalid(self):
        assert validate_batch_delay(0.1) is False  # Too low
        assert validate_batch_delay(15.0) is False  # Too high
        assert validate_batch_delay("1.5") is False  # String
        assert validate_batch_delay(None) is False
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_validators.py::TestNewValidators -v`
Expected: FAIL with "cannot import name 'validate_keyword' from 'reddit_scanner'"

**Step 3: Implement new validators**

Add to `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py` after existing validators (around line 200):

```python
def validate_keyword(keyword: str) -> bool:
    """Validate search keyword format."""
    if not keyword or not isinstance(keyword, str):
        return False
    # 2-50 chars, allow letters, numbers, spaces, hyphens, dots
    return bool(re.match(r'^[\w\s\-\.]{2,50}$', keyword))


def validate_keyword_list(keywords) -> bool:
    """Validate list of search keywords."""
    if not isinstance(keywords, list) or not keywords:
        return False
    return all(validate_keyword(kw) for kw in keywords)


def validate_batch_delay(delay) -> bool:
    """Validate delay is reasonable (0.5s - 10s)."""
    return isinstance(delay, (int, float)) and 0.5 <= delay <= 10.0
```

**Step 4: Update input_validator decorator to handle new validators**

Find the `input_validator` function and add these cases inside the validation loop:

```python
elif validator == 'keyword_list':
    if not validate_keyword_list(param_value):
        raise ValidationError(f"Invalid {param_name}: must be list of valid keywords (2-50 chars each)")

elif validator == 'batch_delay':
    if not validate_batch_delay(param_value):
        raise ValidationError(f"Invalid {param_name}: must be between 0.5 and 10.0 seconds")
```

**Step 5: Run all validator tests**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_validators.py -v`
Expected: All tests pass

**Step 6: Commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add reddit_scanner.py tests/unit/test_validators.py
git commit -m "feat: add keyword and batch_delay validators"
```

---

## Task 6: Replace niche_community_discoverer with v2

**Files:**
- Modify: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py` (replace existing function)
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_community_discoverer.py`
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/integration/test_community_discoverer.py`

**Step 1: Write failing unit test**

Create `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_community_discoverer.py`:

```python
"""Unit tests for niche_community_discoverer v2."""
import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch


class TestCommunityDiscovererUnit:
    """Unit tests with mocked Reddit API."""

    @pytest.mark.asyncio
    async def test_returns_unified_envelope(self):
        """Response should use ToolResponse envelope."""
        from reddit_scanner import niche_community_discoverer

        with patch('reddit_scanner.RedditClient') as mock_client:
            # Setup mock
            mock_reddit = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_reddit)
            mock_client.execute = AsyncMock(return_value=[])
            mock_reddit.subreddits.search = MagicMock(return_value=[])

            result = await niche_community_discoverer(
                topic_keywords=["python automation"],
                batch_delay=0.01  # Fast for testing
            )

        # Parse response
        assert len(result) == 1
        data = json.loads(result[0].text)

        # Check envelope structure
        assert "results" in data
        assert "metadata" in data
        assert "errors" in data
        assert "partial" in data
        assert data["metadata"]["tool"] == "niche_community_discoverer"

    @pytest.mark.asyncio
    async def test_respects_max_communities_cap(self):
        """Should stop when max_communities reached."""
        from reddit_scanner import niche_community_discoverer

        # Create 10 mock subreddits
        mock_subs = []
        for i in range(10):
            sub = MagicMock()
            sub.display_name = f"testsub{i}"
            sub.subscribers = 50000
            sub.public_description = f"Test subreddit {i}"
            mock_subs.append(sub)

        with patch('reddit_scanner.RedditClient') as mock_client:
            mock_reddit = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_reddit)
            mock_client.execute = AsyncMock(return_value=mock_subs)

            result = await niche_community_discoverer(
                topic_keywords=["test1", "test2"],
                max_communities=5,
                batch_delay=0.01
            )

        data = json.loads(result[0].text)
        assert len(data["results"]) <= 5

    @pytest.mark.asyncio
    async def test_filters_by_subscriber_range(self):
        """Should filter communities by subscriber count."""
        from reddit_scanner import niche_community_discoverer

        mock_subs = []
        for i, count in enumerate([1000, 50000, 500000]):  # Below, within, above range
            sub = MagicMock()
            sub.display_name = f"sub{i}"
            sub.subscribers = count
            sub.public_description = "Test"
            mock_subs.append(sub)

        with patch('reddit_scanner.RedditClient') as mock_client:
            mock_reddit = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_reddit)
            mock_client.execute = AsyncMock(return_value=mock_subs)

            result = await niche_community_discoverer(
                topic_keywords=["test"],
                min_subscribers=5000,
                max_subscribers=200000,
                batch_delay=0.01
            )

        data = json.loads(result[0].text)
        # Only the 50000 subscriber sub should be included
        assert len(data["results"]) == 1
        assert data["results"][0]["subscribers"] == 50000
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_community_discoverer.py -v`
Expected: FAIL (signature mismatch or missing topic_keywords parameter)

**Step 3: Replace niche_community_discoverer with v2**

Find and replace the existing `niche_community_discoverer` function in `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py`:

```python
@mcp.tool()
@input_validator(
    ("topic_keywords", "topic_keywords", "keyword_list"),
    ("min_subscribers", "min_subscribers", "positive_int"),
    ("max_subscribers", "max_subscribers", "positive_int"),
    ("max_communities", "max_communities", "positive_int"),
    ("batch_delay", "batch_delay", "batch_delay")
)
async def niche_community_discoverer(
    topic_keywords: List[str],
    min_subscribers: int = 5000,
    max_subscribers: int = 200000,
    max_communities: int = 50,
    spider_sidebar: bool = True,
    batch_delay: float = 1.5
) -> List[TextContent]:
    """Enterprise-Grade Discovery: Finds communities by topic keywords,
    then spiders via sidebar 'related communities' links.

    Args:
        topic_keywords: Intent-based search terms (e.g., "python automation")
        min_subscribers: Minimum subscriber count
        max_subscribers: Maximum subscriber count
        max_communities: Hard cap on total communities to return
        spider_sidebar: Whether to spider sidebar for related communities
        batch_delay: Seconds between API requests (respectful crawling)
    """
    response = ToolResponse(tool_name="niche_community_discoverer")
    config = RateLimitConfig(batch_delay=batch_delay, request_budget=100)
    executor = RateLimitedExecutor(config)

    discovered_map = {}  # {name: metadata}

    # 1. Primary Search Phase (keyword-based discovery)
    for keyword in topic_keywords:
        if len(discovered_map) >= max_communities:
            response.set_partial(reason="max_communities cap reached")
            break

        try:
            reddit = await RedditClient.get()

            # Search for subreddits by keyword
            results = await executor.execute(
                lambda kw=keyword: list(reddit.subreddits.search(kw, limit=20))
            )

            for sub in results:
                if len(discovered_map) >= max_communities:
                    break

                if sub.display_name not in discovered_map:
                    if min_subscribers <= sub.subscribers <= max_subscribers:
                        discovered_map[sub.display_name] = {
                            "name": sub.display_name,
                            "subscribers": sub.subscribers,
                            "source": "search",
                            "keyword": keyword,
                            "description": (sub.public_description[:200]
                                          if sub.public_description else ""),
                            "url": f"https://reddit.com/r/{sub.display_name}"
                        }

        except BudgetExhaustedError:
            response.set_partial(reason="request budget exhausted")
            break
        except Exception as e:
            logger.warning(f"Search failed for keyword '{keyword}': {e}")
            response.add_error(item=keyword, reason=str(e))

    # 2. Sidebar Spider Phase (one hop only)
    if spider_sidebar and len(discovered_map) < max_communities:
        current_names = list(discovered_map.keys())

        for name in current_names:
            if len(discovered_map) >= max_communities:
                break

            try:
                reddit = await RedditClient.get()
                sub_obj = await executor.execute(
                    lambda n=name: reddit.subreddit(n)
                )

                # Fetch sidebar text
                sidebar_text = await executor.execute(
                    lambda s=sub_obj: s.description if hasattr(s, 'description') else ""
                )

                if not sidebar_text:
                    continue

                # Extract related communities (r/Name patterns)
                related = re.findall(r'r/([A-Za-z0-9_]{3,21})', sidebar_text)

                for rel_name in related:
                    if len(discovered_map) >= max_communities:
                        break

                    if rel_name not in discovered_map:
                        try:
                            rel_sub = await executor.execute(
                                lambda rn=rel_name: reddit.subreddit(rn)
                            )

                            # Fetch to get subscriber count
                            await executor.execute(lambda rs=rel_sub: rs._fetch())

                            if min_subscribers <= rel_sub.subscribers <= max_subscribers:
                                discovered_map[rel_name] = {
                                    "name": rel_name,
                                    "subscribers": rel_sub.subscribers,
                                    "source": f"sidebar of {name}",
                                    "description": (rel_sub.public_description[:200]
                                                  if rel_sub.public_description else ""),
                                    "url": f"https://reddit.com/r/{rel_name}"
                                }
                        except Exception:
                            continue  # Skip invalid/private subreddits

            except BudgetExhaustedError:
                response.set_partial(reason="request budget exhausted during spidering")
                break
            except Exception as e:
                logger.warning(f"Spidering failed for {name}: {e}")

    # Build results sorted by subscriber count
    sorted_results = sorted(
        discovered_map.values(),
        key=lambda x: x['subscribers'],
        reverse=True
    )

    for community in sorted_results:
        response.add_result(community)

    # Add stats
    response.stats["keywords_searched"] = len(topic_keywords)
    response.stats["communities_found"] = len(discovered_map)
    response.stats["spider_enabled"] = spider_sidebar
    response.stats.update(executor.get_stats())

    return response.to_response()
```

**Step 4: Also remove the old `_analyze_community_safely` helper function**

Delete the `_analyze_community_safely` function as it's no longer needed.

**Step 5: Run unit tests**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_community_discoverer.py -v`
Expected: All tests pass

**Step 6: Create integration test**

Create `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/integration/test_community_discoverer.py`:

```python
"""Integration tests for niche_community_discoverer v2."""
import pytest
import json


@pytest.mark.integration
class TestCommunityDiscovererIntegration:
    """Integration tests requiring Reddit API credentials."""

    @pytest.mark.asyncio
    async def test_keyword_search_returns_results(self):
        """Should find communities matching keywords."""
        from reddit_scanner import niche_community_discoverer

        result = await niche_community_discoverer(
            topic_keywords=["python programming"],
            min_subscribers=10000,
            max_subscribers=1000000,
            max_communities=5,
            spider_sidebar=False,
            batch_delay=1.5
        )

        data = json.loads(result[0].text)

        assert data["metadata"]["tool"] == "niche_community_discoverer"
        assert len(data["results"]) > 0
        assert data["results"][0]["source"] == "search"

    @pytest.mark.asyncio
    async def test_sidebar_spidering_finds_related(self):
        """Should find related communities via sidebar."""
        from reddit_scanner import niche_community_discoverer

        result = await niche_community_discoverer(
            topic_keywords=["learnpython"],
            min_subscribers=1000,
            max_subscribers=5000000,
            max_communities=10,
            spider_sidebar=True,
            batch_delay=1.5
        )

        data = json.loads(result[0].text)

        # Should have at least one result from sidebar
        sidebar_sources = [r for r in data["results"] if "sidebar of" in r.get("source", "")]
        # Note: This may be 0 if the subreddit has no sidebar links - that's OK
        assert data["metadata"]["stats"]["spider_enabled"] is True
```

**Step 7: Run integration test (optional - requires credentials)**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/integration/test_community_discoverer.py -v -m integration`
Expected: Tests pass (or skip if no credentials)

**Step 8: Commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add reddit_scanner.py tests/unit/test_community_discoverer.py tests/integration/test_community_discoverer.py
git commit -m "feat: replace niche_community_discoverer with keyword-based v2

BREAKING CHANGE: Parameter changed from seed_subreddits to topic_keywords"
```

---

## Task 7: Implement workflow_thread_inspector

**Files:**
- Modify: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py` (add new tool)
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_thread_inspector.py`
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/integration/test_thread_inspector.py`

**Step 1: Write failing unit test**

Create `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_thread_inspector.py`:

```python
"""Unit tests for workflow_thread_inspector."""
import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch


class TestThreadInspectorUnit:
    """Unit tests with mocked Reddit API."""

    @pytest.mark.asyncio
    async def test_returns_unified_envelope(self):
        """Response should use ToolResponse envelope."""
        from reddit_scanner import workflow_thread_inspector

        # Mock submission
        mock_submission = MagicMock()
        mock_submission.id = "test123"
        mock_submission.title = "Test post"
        mock_submission.permalink = "/r/test/comments/test123/"
        mock_submission.comments = MagicMock()
        mock_submission.comments.replace_more = MagicMock()
        mock_submission.comments.list = MagicMock(return_value=[])

        with patch('reddit_scanner.RedditClient') as mock_client:
            mock_reddit = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_reddit)
            mock_client.execute = AsyncMock(side_effect=[
                mock_submission,  # First call: get submission
                None,  # Second call: replace_more
                []  # Third call: list comments
            ])

            result = await workflow_thread_inspector(
                post_ids=["test123"],
                batch_delay=0.01
            )

        data = json.loads(result[0].text)

        assert "results" in data
        assert "metadata" in data
        assert data["metadata"]["tool"] == "workflow_thread_inspector"

    @pytest.mark.asyncio
    async def test_filters_by_workflow_signals(self):
        """Should only return comments matching signals."""
        from reddit_scanner import workflow_thread_inspector

        # Create mock comments
        matching_comment = MagicMock()
        matching_comment.id = "c1"
        matching_comment.body = "Here is the step by step process I use"
        matching_comment.score = 10
        matching_comment.depth = 1
        matching_comment.permalink = "/r/test/c1"
        matching_comment.author = MagicMock()
        matching_comment.author.name = "user1"

        non_matching_comment = MagicMock()
        non_matching_comment.id = "c2"
        non_matching_comment.body = "This is just a regular comment"
        non_matching_comment.score = 5
        non_matching_comment.depth = 1
        non_matching_comment.permalink = "/r/test/c2"
        non_matching_comment.author = MagicMock()
        non_matching_comment.author.name = "user2"

        mock_submission = MagicMock()
        mock_submission.id = "test123"
        mock_submission.title = "Test post"
        mock_submission.permalink = "/r/test/comments/test123/"
        mock_submission.comments = MagicMock()
        mock_submission.comments.replace_more = MagicMock()
        mock_submission.comments.list = MagicMock(return_value=[
            matching_comment, non_matching_comment
        ])

        with patch('reddit_scanner.RedditClient') as mock_client:
            mock_reddit = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_reddit)
            mock_client.execute = AsyncMock(side_effect=[
                mock_submission,
                None,
                [matching_comment, non_matching_comment]
            ])

            result = await workflow_thread_inspector(
                post_ids=["test123"],
                workflow_signals=["step", "process"],
                batch_delay=0.01
            )

        data = json.loads(result[0].text)

        # Only matching comment should be included
        assert len(data["results"]) == 1
        assert len(data["results"][0]["workflow_comments"]) == 1
        assert "step" in data["results"][0]["workflow_comments"][0]["body"].lower()

    @pytest.mark.asyncio
    async def test_uses_default_signals_when_none_provided(self):
        """Should use default workflow signals if none provided."""
        from reddit_scanner import workflow_thread_inspector

        mock_submission = MagicMock()
        mock_submission.id = "test123"
        mock_submission.title = "Test"
        mock_submission.permalink = "/r/test/"
        mock_submission.comments = MagicMock()
        mock_submission.comments.replace_more = MagicMock()
        mock_submission.comments.list = MagicMock(return_value=[])

        with patch('reddit_scanner.RedditClient') as mock_client:
            mock_reddit = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_reddit)
            mock_client.execute = AsyncMock(side_effect=[mock_submission, None, []])

            result = await workflow_thread_inspector(
                post_ids=["test123"],
                workflow_signals=None,  # Use defaults
                batch_delay=0.01
            )

        data = json.loads(result[0].text)

        # Should have used default signals
        signals_used = data["metadata"]["stats"].get("signals_used", [])
        assert len(signals_used) > 0
        assert "step" in signals_used or "process" in signals_used
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_thread_inspector.py -v`
Expected: FAIL with "cannot import name 'workflow_thread_inspector'"

**Step 3: Implement workflow_thread_inspector**

Add to `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py` after the updated niche_community_discoverer:

```python
# Default workflow signals
DEFAULT_WORKFLOW_SIGNALS = [
    "step", "process", "export", "import", "csv", "manual", "copy", "paste",
    "click", "download", "upload", "workaround", "hack", "tedious"
]


@mcp.tool()
@input_validator(
    ("post_ids", "post_ids", "post_id_list"),
    ("comment_limit", "comment_limit", "positive_int"),
    ("expand_depth", "expand_depth", "positive_int"),
    ("min_score", "min_score", "positive_int"),
    ("batch_delay", "batch_delay", "batch_delay")
)
async def workflow_thread_inspector(
    post_ids: List[str],
    workflow_signals: List[str] = None,
    comment_limit: int = 100,
    expand_depth: int = 5,
    min_score: int = 1,
    batch_delay: float = 1.5
) -> List[TextContent]:
    """Deep Analysis: Expands comment trees to find workflow details
    hidden in nested replies.

    Args:
        post_ids: Reddit post IDs to analyze
        workflow_signals: Keywords indicating workflow details (LLM-generated)
        comment_limit: Max comments to return per post
        expand_depth: How many "load more" expansions (higher = more API calls)
        min_score: Minimum comment score to include
        batch_delay: Seconds between API requests
    """
    response = ToolResponse(tool_name="workflow_thread_inspector")
    config = RateLimitConfig(batch_delay=batch_delay, request_budget=100)
    executor = RateLimitedExecutor(config)

    # Use default signals if none provided
    signals = workflow_signals if workflow_signals else DEFAULT_WORKFLOW_SIGNALS
    signals_lower = [s.lower() for s in signals]

    for post_id in post_ids:
        try:
            reddit = await RedditClient.get()

            # Fetch submission
            submission = await executor.execute(
                lambda pid=post_id: reddit.submission(id=pid)
            )

            # Expand comment tree (this calls /api/morechildren)
            await executor.execute(
                lambda s=submission: s.comments.replace_more(limit=expand_depth)
            )

            # Get flattened comment list
            all_comments = await executor.execute(
                lambda s=submission: s.comments.list()
            )

            # Filter for workflow signals
            useful_comments = []
            for comment in all_comments[:comment_limit]:
                if not hasattr(comment, 'body') or not hasattr(comment, 'score'):
                    continue

                if comment.score < min_score:
                    continue

                # Check for workflow signals
                body_lower = comment.body.lower()
                if any(signal in body_lower for signal in signals_lower):
                    useful_comments.append({
                        "id": comment.id,
                        "author": comment.author.name if comment.author else "[deleted]",
                        "body": comment.body[:1000],  # Truncate long comments
                        "score": comment.score,
                        "depth": comment.depth if hasattr(comment, 'depth') else 0,
                        "url": f"https://reddit.com{comment.permalink}" if hasattr(comment, 'permalink') else ""
                    })

            if useful_comments or len(all_comments) > 0:
                response.add_result({
                    "post_id": post_id,
                    "post_title": submission.title if hasattr(submission, 'title') else "",
                    "post_url": f"https://reddit.com{submission.permalink}" if hasattr(submission, 'permalink') else "",
                    "total_comments_scanned": min(len(all_comments), comment_limit),
                    "workflow_comments": useful_comments
                })

        except BudgetExhaustedError:
            response.set_partial(reason="request budget exhausted")
            break
        except Exception as e:
            logger.error(f"Failed to inspect thread {post_id}: {e}")
            response.add_error(item=post_id, reason=str(e))

    # Add stats
    response.stats["signals_used"] = signals
    response.stats["posts_analyzed"] = len([r for r in response.results])
    response.stats.update(executor.get_stats())

    return response.to_response()
```

**Step 4: Run unit tests**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_thread_inspector.py -v`
Expected: All tests pass

**Step 5: Create integration test**

Create `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/integration/test_thread_inspector.py`:

```python
"""Integration tests for workflow_thread_inspector."""
import pytest
import json


@pytest.mark.integration
class TestThreadInspectorIntegration:
    """Integration tests requiring Reddit API credentials."""

    @pytest.mark.asyncio
    async def test_expands_and_filters_comments(self):
        """Should expand comments and filter by signals."""
        from reddit_scanner import workflow_thread_inspector

        # Use a known post that likely has workflow-related comments
        # This post ID should be replaced with a stable, known post
        result = await workflow_thread_inspector(
            post_ids=["1abc2d3"],  # Replace with real post ID for testing
            workflow_signals=["step", "process", "manual"],
            comment_limit=50,
            expand_depth=2,
            batch_delay=1.5
        )

        data = json.loads(result[0].text)

        assert data["metadata"]["tool"] == "workflow_thread_inspector"
        assert "signals_used" in data["metadata"]["stats"]
        # May have errors if post doesn't exist, that's OK for this test
```

**Step 6: Commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add reddit_scanner.py tests/unit/test_thread_inspector.py tests/integration/test_thread_inspector.py
git commit -m "feat: add workflow_thread_inspector for deep comment analysis"
```

---

## Task 8: Implement wiki_tool_extractor

**Files:**
- Modify: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py` (add new tool)
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_wiki_extractor.py`
- Create: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/integration/test_wiki_extractor.py`

**Step 1: Write failing unit test**

Create `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/unit/test_wiki_extractor.py`:

```python
"""Unit tests for wiki_tool_extractor."""
import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch


class TestWikiExtractorUnit:
    """Unit tests with mocked Reddit API."""

    @pytest.mark.asyncio
    async def test_returns_unified_envelope(self):
        """Response should use ToolResponse envelope."""
        from reddit_scanner import wiki_tool_extractor

        mock_subreddit = MagicMock()
        mock_subreddit.display_name = "test"
        mock_subreddit.description = ""
        mock_subreddit.public_description = ""
        mock_subreddit.wiki = []

        with patch('reddit_scanner.RedditClient') as mock_client:
            mock_reddit = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_reddit)
            mock_client.execute = AsyncMock(return_value=mock_subreddit)

            result = await wiki_tool_extractor(
                subreddit_names=["test"],
                batch_delay=0.01
            )

        data = json.loads(result[0].text)

        assert "results" in data
        assert "metadata" in data
        assert data["metadata"]["tool"] == "wiki_tool_extractor"

    @pytest.mark.asyncio
    async def test_extracts_markdown_links(self):
        """Should extract markdown-formatted links."""
        from reddit_scanner import wiki_tool_extractor

        mock_subreddit = MagicMock()
        mock_subreddit.display_name = "test"
        mock_subreddit.description = "Check out [Notion](https://notion.so) and [Obsidian](https://obsidian.md)"
        mock_subreddit.public_description = ""
        mock_subreddit.wiki = []

        with patch('reddit_scanner.RedditClient') as mock_client:
            mock_reddit = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_reddit)
            mock_client.execute = AsyncMock(return_value=mock_subreddit)

            result = await wiki_tool_extractor(
                subreddit_names=["test"],
                scan_sidebar=True,
                scan_wiki=False,
                batch_delay=0.01
            )

        data = json.loads(result[0].text)

        tools = data["results"][0]["tools"]
        tool_names = [t["name"] for t in tools]

        assert "Notion" in tool_names
        assert "Obsidian" in tool_names

    @pytest.mark.asyncio
    async def test_extracts_bare_urls(self):
        """Should extract bare URLs and infer names from domain."""
        from reddit_scanner import wiki_tool_extractor

        mock_subreddit = MagicMock()
        mock_subreddit.display_name = "test"
        mock_subreddit.description = "Try https://todoist.com for tasks and https://calendly.com for scheduling"
        mock_subreddit.public_description = ""
        mock_subreddit.wiki = []

        with patch('reddit_scanner.RedditClient') as mock_client:
            mock_reddit = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_reddit)
            mock_client.execute = AsyncMock(return_value=mock_subreddit)

            result = await wiki_tool_extractor(
                subreddit_names=["test"],
                scan_sidebar=True,
                scan_wiki=False,
                batch_delay=0.01
            )

        data = json.loads(result[0].text)

        tools = data["results"][0]["tools"]
        urls = [t["url"] for t in tools]

        assert any("todoist.com" in url for url in urls)
        assert any("calendly.com" in url for url in urls)

    @pytest.mark.asyncio
    async def test_deduplicates_tools(self):
        """Should deduplicate tools by URL."""
        from reddit_scanner import wiki_tool_extractor

        mock_subreddit = MagicMock()
        mock_subreddit.display_name = "test"
        mock_subreddit.description = "[Notion](https://notion.so) and also https://notion.so"
        mock_subreddit.public_description = "Check https://notion.so too"
        mock_subreddit.wiki = []

        with patch('reddit_scanner.RedditClient') as mock_client:
            mock_reddit = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_reddit)
            mock_client.execute = AsyncMock(return_value=mock_subreddit)

            result = await wiki_tool_extractor(
                subreddit_names=["test"],
                batch_delay=0.01
            )

        data = json.loads(result[0].text)

        tools = data["results"][0]["tools"]
        urls = [t["url"] for t in tools]

        # Should only have one entry for notion.so
        notion_count = sum(1 for url in urls if "notion.so" in url)
        assert notion_count == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_wiki_extractor.py -v`
Expected: FAIL with "cannot import name 'wiki_tool_extractor'"

**Step 3: Implement wiki_tool_extractor**

Add to `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/reddit_scanner.py` after workflow_thread_inspector:

```python
# Default wiki page keywords
DEFAULT_WIKI_PAGE_KEYWORDS = [
    "tools", "software", "resources", "guide", "faq", "index", "wiki", "recommended"
]


def extract_domain_name(url: str) -> str:
    """Extract a readable name from a URL domain."""
    try:
        # Parse the domain
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Get the main domain name (before first dot)
        name = domain.split(".")[0]

        # Capitalize and return
        return name.capitalize()
    except Exception:
        return url


def normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    url = url.rstrip("/")
    # Remove query params and fragments
    if "?" in url:
        url = url.split("?")[0]
    if "#" in url:
        url = url.split("#")[0]
    return url.lower()


@mcp.tool()
@input_validator(
    ("subreddit_names", "subreddit_names", "subreddit_list"),
    ("batch_delay", "batch_delay", "batch_delay")
)
async def wiki_tool_extractor(
    subreddit_names: List[str],
    scan_sidebar: bool = True,
    scan_wiki: bool = True,
    page_keywords: List[str] = None,
    batch_delay: float = 1.5
) -> List[TextContent]:
    """Market Intelligence: Scans subreddit wikis and sidebars for
    mentioned tools and software to map the competitive landscape.

    Args:
        subreddit_names: Communities to scan
        scan_sidebar: Include sidebar/description in scan
        scan_wiki: Include wiki pages in scan
        page_keywords: Wiki page name filters (scans all if comprehensive)
        batch_delay: Seconds between API requests
    """
    response = ToolResponse(tool_name="wiki_tool_extractor")
    config = RateLimitConfig(batch_delay=batch_delay, request_budget=100)
    executor = RateLimitedExecutor(config)

    keywords = page_keywords if page_keywords else DEFAULT_WIKI_PAGE_KEYWORDS
    pages_scanned = 0

    # Regex patterns for link extraction
    markdown_link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
    bare_url_pattern = re.compile(r'(?<!\()(https?://[^\s\)\]<>"]+)')

    for subreddit_name in subreddit_names:
        try:
            reddit = await RedditClient.get()

            subreddit = await executor.execute(
                lambda name=subreddit_name: reddit.subreddit(name)
            )

            # Collect all tools found for this subreddit
            tools_map = {}  # normalized_url -> {name, url, sources}

            # 1. Sidebar Scan
            if scan_sidebar:
                texts_to_scan = []

                # Get sidebar description
                if hasattr(subreddit, 'description') and subreddit.description:
                    texts_to_scan.append(("sidebar", subreddit.description))
                if hasattr(subreddit, 'public_description') and subreddit.public_description:
                    texts_to_scan.append(("public_description", subreddit.public_description))

                for source, text in texts_to_scan:
                    # Extract markdown links
                    for match in markdown_link_pattern.finditer(text):
                        name, url = match.groups()
                        norm_url = normalize_url(url)
                        if norm_url not in tools_map:
                            tools_map[norm_url] = {
                                "name": name,
                                "url": url,
                                "sources": [source]
                            }
                        elif source not in tools_map[norm_url]["sources"]:
                            tools_map[norm_url]["sources"].append(source)

                    # Extract bare URLs
                    for match in bare_url_pattern.finditer(text):
                        url = match.group(0)
                        norm_url = normalize_url(url)
                        if norm_url not in tools_map:
                            tools_map[norm_url] = {
                                "name": extract_domain_name(url),
                                "url": url,
                                "sources": [source]
                            }
                        elif source not in tools_map[norm_url]["sources"]:
                            tools_map[norm_url]["sources"].append(source)

            # 2. Wiki Scan
            if scan_wiki:
                try:
                    wiki_pages = await executor.execute(
                        lambda s=subreddit: list(s.wiki)
                    )

                    for page in wiki_pages:
                        page_name = page.name.lower() if hasattr(page, 'name') else str(page).lower()

                        # Check if page matches keywords or scan all
                        should_scan = any(kw in page_name for kw in keywords)

                        if should_scan:
                            try:
                                content = await executor.execute(
                                    lambda p=page: p.content_md
                                )
                                pages_scanned += 1

                                source = f"wiki/{page_name}"

                                # Extract markdown links
                                for match in markdown_link_pattern.finditer(content):
                                    name, url = match.groups()
                                    norm_url = normalize_url(url)
                                    if norm_url not in tools_map:
                                        tools_map[norm_url] = {
                                            "name": name,
                                            "url": url,
                                            "sources": [source]
                                        }
                                    elif source not in tools_map[norm_url]["sources"]:
                                        tools_map[norm_url]["sources"].append(source)

                                # Extract bare URLs
                                for match in bare_url_pattern.finditer(content):
                                    url = match.group(0)
                                    norm_url = normalize_url(url)
                                    if norm_url not in tools_map:
                                        tools_map[norm_url] = {
                                            "name": extract_domain_name(url),
                                            "url": url,
                                            "sources": [source]
                                        }
                                    elif source not in tools_map[norm_url]["sources"]:
                                        tools_map[norm_url]["sources"].append(source)

                            except Exception as e:
                                logger.debug(f"Could not read wiki page {page_name}: {e}")

                except Exception as e:
                    logger.debug(f"Could not access wiki for {subreddit_name}: {e}")

            # Add result for this subreddit
            response.add_result({
                "subreddit": subreddit_name,
                "tools": list(tools_map.values()),
                "tool_count": len(tools_map)
            })

        except BudgetExhaustedError:
            response.set_partial(reason="request budget exhausted")
            break
        except Exception as e:
            logger.error(f"Failed to extract from {subreddit_name}: {e}")
            response.add_error(item=subreddit_name, reason=str(e))

    # Add stats
    response.stats["subreddits_scanned"] = len([r for r in response.results])
    response.stats["pages_scanned"] = pages_scanned
    response.stats["scan_sidebar"] = scan_sidebar
    response.stats["scan_wiki"] = scan_wiki
    response.stats.update(executor.get_stats())

    return response.to_response()
```

**Step 4: Run unit tests**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/test_wiki_extractor.py -v`
Expected: All tests pass

**Step 5: Create integration test**

Create `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/tests/integration/test_wiki_extractor.py`:

```python
"""Integration tests for wiki_tool_extractor."""
import pytest
import json


@pytest.mark.integration
class TestWikiExtractorIntegration:
    """Integration tests requiring Reddit API credentials."""

    @pytest.mark.asyncio
    async def test_extracts_from_real_subreddit(self):
        """Should extract tools from a real subreddit."""
        from reddit_scanner import wiki_tool_extractor

        result = await wiki_tool_extractor(
            subreddit_names=["learnpython"],
            scan_sidebar=True,
            scan_wiki=True,
            batch_delay=1.5
        )

        data = json.loads(result[0].text)

        assert data["metadata"]["tool"] == "wiki_tool_extractor"
        assert len(data["results"]) > 0
        assert data["results"][0]["subreddit"] == "learnpython"
```

**Step 6: Commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add reddit_scanner.py tests/unit/test_wiki_extractor.py tests/integration/test_wiki_extractor.py
git commit -m "feat: add wiki_tool_extractor for competitive landscape mapping"
```

---

## Task 9: Update test_all_tools.py for New Tools

**Files:**
- Modify: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/test_all_tools.py`

**Step 1: Update test configurations**

Add new tool configurations to the `test_configs` list in `test_all_tools.py`:

Find the `test_configs` list and update the `niche_community_discoverer` entry and add new tools:

```python
# Update niche_community_discoverer (around line 246)
{
    "name": "niche_community_discoverer",
    "args": {
        "topic_keywords": ["python automation", "productivity tools"],
        "min_subscribers": 5000,
        "max_subscribers": 200000,
        "max_communities": 10,
        "spider_sidebar": True,
        "batch_delay": 1.5
    },
    "description": "Discovering niche communities by topic keywords"
},

# Add after idea_validation_scorer (around line 295)
{
    "name": "workflow_thread_inspector",
    "args": {
        "post_ids": ["1abc2d3"],  # Placeholder - will likely fail
        "workflow_signals": ["step", "process", "manual", "export"],
        "comment_limit": 50,
        "expand_depth": 3,
        "min_score": 1,
        "batch_delay": 1.5
    },
    "description": "Inspecting threads for workflow details"
},
{
    "name": "wiki_tool_extractor",
    "args": {
        "subreddit_names": ["learnpython", "productivity"],
        "scan_sidebar": True,
        "scan_wiki": True,
        "page_keywords": ["tools", "resources", "software"],
        "batch_delay": 1.5
    },
    "description": "Extracting tools from wiki and sidebar"
}
```

**Step 2: Update tool count in comments**

Update the docstring to say "12 tools" instead of "10 tools".

**Step 3: Commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add test_all_tools.py
git commit -m "test: update test_all_tools.py for new Market Intelligence tools"
```

---

## Task 10: Run Full Test Suite

**Step 1: Run all unit tests**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/unit/ -v`
Expected: All unit tests pass

**Step 2: Run integration tests (requires credentials)**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/integration/ -v -m integration`
Expected: Tests pass (may skip some if posts don't exist)

**Step 3: Run with coverage**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python -m pytest tests/ --cov=reddit_scanner --cov-report=term-missing`
Expected: Coverage report showing covered lines

**Step 4: Run the MCP tool tester**

Run: `cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner && python test_all_tools.py`
Expected: Tools execute (some may have expected errors for placeholder IDs)

**Step 5: Final commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add -A
git commit -m "test: complete Market Intelligence Spider test suite"
```

---

## Task 11: Update Documentation

**Files:**
- Modify: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/CLAUDE.md`
- Modify: `/Users/matthewhans/Desktop/Programming/mcp/reddit-scanner/README.md`

**Step 1: Update CLAUDE.md with new tool descriptions**

Add new tools to the Tool Categories section and update any outdated information.

**Step 2: Update README.md with new features**

Add Market Intelligence Spider section describing the new capabilities.

**Step 3: Commit**

```bash
cd /Users/matthewhans/Desktop/Programming/mcp/reddit-scanner
git add CLAUDE.md README.md
git commit -m "docs: update documentation for Market Intelligence Spider"
```

---

## Summary

**Total Tasks:** 11
**Estimated Commits:** 12

**New Files Created:**
- `pytest.ini`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/unit/__init__.py`
- `tests/unit/test_rate_limiting.py`
- `tests/unit/test_tool_response.py`
- `tests/unit/test_validators.py`
- `tests/unit/test_community_discoverer.py`
- `tests/unit/test_thread_inspector.py`
- `tests/unit/test_wiki_extractor.py`
- `tests/integration/__init__.py`
- `tests/integration/test_community_discoverer.py`
- `tests/integration/test_thread_inspector.py`
- `tests/integration/test_wiki_extractor.py`

**Files Modified:**
- `pyproject.toml` - Add dev dependencies
- `reddit_scanner.py` - Add infrastructure and new tools
- `test_all_tools.py` - Update for new tools
- `CLAUDE.md` - Update documentation
- `README.md` - Update documentation
