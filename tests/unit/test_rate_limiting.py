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

        async def mock_op():
            return "result"

        await executor.execute(mock_op)
        await executor.execute(mock_op)

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

        assert elapsed >= 0.1
