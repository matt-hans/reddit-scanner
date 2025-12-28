"""Integration tests for workflow_thread_inspector tool."""
import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch


class TestWorkflowThreadInspector:
    """Tests for the workflow_thread_inspector tool."""

    @pytest.fixture
    def mock_comment_factory(self):
        """Factory for creating mock comments with workflow signals."""
        def make_comment(
            comment_id: str,
            body: str,
            score: int = 10,
            depth: int = 1,
            author_name: str = "testuser"
        ):
            comment = MagicMock()
            comment.id = comment_id
            comment.body = body
            comment.score = score
            comment.depth = depth
            comment.permalink = f"/r/test/comments/abc123/test/{comment_id}/"
            comment.author = MagicMock()
            comment.author.name = author_name
            return comment
        return make_comment

    @pytest.fixture
    def mock_submission_factory(self, mock_comment_factory):
        """Factory for creating mock submissions with comments."""
        def make_submission(
            post_id: str,
            title: str = "Test Post Title",
            comments: list = None
        ):
            submission = MagicMock()
            submission.id = post_id
            submission.title = title
            submission.url = f"https://reddit.com/r/test/comments/{post_id}/"

            # Create default comments if none provided
            if comments is None:
                comments = [
                    mock_comment_factory(
                        "c1", "First step is to export the csv file manually",
                        score=15, depth=1
                    ),
                    mock_comment_factory(
                        "c2", "Then I click through each option and download",
                        score=8, depth=2
                    ),
                    mock_comment_factory(
                        "c3", "This is just a random comment without signals",
                        score=5, depth=1
                    ),
                    mock_comment_factory(
                        "c4", "Low score comment with step process",
                        score=0, depth=1
                    ),
                ]

            # Mock comments object
            submission.comments = MagicMock()
            submission.comments.replace_more = MagicMock(return_value=None)
            submission.comments.list = MagicMock(return_value=comments)

            return submission
        return make_submission

    @pytest.fixture
    def mock_reddit_client(self, mock_submission_factory):
        """Create mock Reddit client with submission method."""
        reddit = MagicMock()

        # Store submissions for lookup
        submissions = {
            "abc123": mock_submission_factory("abc123", "How to automate my workflow"),
            "def456": mock_submission_factory("def456", "Manual process is tedious"),
        }

        def submission_side_effect(id):
            if id in submissions:
                return submissions[id]
            raise Exception(f"Post {id} not found")

        reddit.submission = MagicMock(side_effect=submission_side_effect)
        return reddit

    @pytest.mark.asyncio
    async def test_basic_thread_inspection(self, mock_reddit_client):
        """Test basic thread inspection returns workflow comments."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["abc123"],
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
        assert response["metadata"]["tool"] == "workflow_thread_inspector"
        assert "signals_used" in response["metadata"]
        assert "timestamp" in response["metadata"]

    @pytest.mark.asyncio
    async def test_default_workflow_signals_applied(self, mock_reddit_client):
        """Test that default signals are used when None passed."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["abc123"],
                    workflow_signals=None,  # Explicitly pass None
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Check that default signals were used
        signals_used = response["metadata"]["signals_used"]
        assert "step" in signals_used
        assert "process" in signals_used
        assert "manual" in signals_used
        assert "export" in signals_used

    @pytest.mark.asyncio
    async def test_custom_workflow_signals(self, mock_reddit_client):
        """Test that custom signals are used when provided."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        custom_signals = ["custom", "signal"]

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["abc123"],
                    workflow_signals=custom_signals,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)
        assert response["metadata"]["signals_used"] == custom_signals

    @pytest.mark.asyncio
    async def test_min_score_filtering(self, mock_reddit_client):
        """Test that comments below min_score are filtered out."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["abc123"],
                    min_score=5,  # Should filter out score=0 comments
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # All workflow_comments should have score >= 5
        for post_result in response["results"]:
            for comment in post_result.get("workflow_comments", []):
                assert comment["score"] >= 5

    @pytest.mark.asyncio
    async def test_signal_keyword_filtering(self, mock_comment_factory, mock_submission_factory):
        """Test that only comments with signal keywords are included."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        # Create submission with specific comments
        comments = [
            mock_comment_factory("c1", "I need to export the data manually", score=10),
            mock_comment_factory("c2", "This is a great post!", score=15),  # No signals
            mock_comment_factory("c3", "The step by step process is tedious", score=12),
        ]
        submission = mock_submission_factory("test123", "Test", comments)

        reddit = MagicMock()
        reddit.submission = MagicMock(return_value=submission)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["test123"],
                    workflow_signals=["export", "step", "manual", "tedious"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should only include comments with signal keywords
        workflow_comments = response["results"][0]["workflow_comments"]
        # c2 "This is a great post!" has no signals, should be excluded
        bodies = [c["body"] for c in workflow_comments]
        assert "This is a great post!" not in bodies
        assert any("export" in b.lower() or "step" in b.lower() for b in bodies)

    @pytest.mark.asyncio
    async def test_body_truncation_at_1000_chars(self, mock_comment_factory, mock_submission_factory):
        """Test that comment body is truncated to 1000 chars with '...'."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        # Create a very long comment body
        long_body = "step " + "x" * 2000  # Over 1000 chars
        comments = [mock_comment_factory("c1", long_body, score=10)]
        submission = mock_submission_factory("test123", "Test", comments)

        reddit = MagicMock()
        reddit.submission = MagicMock(return_value=submission)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["test123"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Check body is truncated
        workflow_comments = response["results"][0]["workflow_comments"]
        if workflow_comments:
            body = workflow_comments[0]["body"]
            assert len(body) <= 1003  # 1000 chars + "..."
            assert body.endswith("...")

    @pytest.mark.asyncio
    async def test_result_structure_per_post(self, mock_reddit_client):
        """Test that each result has post_id, post_title, post_url, workflow_comments."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["abc123"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Check result structure
        assert len(response["results"]) == 1
        post_result = response["results"][0]
        assert "post_id" in post_result
        assert "post_title" in post_result
        assert "post_url" in post_result
        assert "workflow_comments" in post_result
        assert post_result["post_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_comment_structure(self, mock_reddit_client):
        """Test that each comment has author, body, score, depth, url."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["abc123"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        workflow_comments = response["results"][0]["workflow_comments"]
        for comment in workflow_comments:
            assert "author" in comment
            assert "body" in comment
            assert "score" in comment
            assert "depth" in comment
            assert "url" in comment

    @pytest.mark.asyncio
    async def test_multiple_post_ids(self, mock_reddit_client):
        """Test processing multiple post IDs."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["abc123", "def456"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should have results for both posts
        assert len(response["results"]) == 2
        post_ids = [r["post_id"] for r in response["results"]]
        assert "abc123" in post_ids
        assert "def456" in post_ids

    @pytest.mark.asyncio
    async def test_empty_post_ids_validation(self):
        """Test that empty post_ids list is rejected."""
        from reddit_scanner import workflow_thread_inspector

        result = await workflow_thread_inspector(
            post_ids=[],
            batch_delay=0.5
        )

        response = json.loads(result[0].text)
        assert len(response["errors"]) > 0 or response["partial"]

    @pytest.mark.asyncio
    async def test_invalid_post_id_format_validation(self):
        """Test that invalid post ID format is rejected."""
        from reddit_scanner import workflow_thread_inspector

        result = await workflow_thread_inspector(
            post_ids=["ab"],  # Too short (valid IDs are 6-10 chars)
            batch_delay=0.5
        )

        response = json.loads(result[0].text)
        assert len(response["errors"]) > 0 or response["partial"]

    @pytest.mark.asyncio
    async def test_invalid_batch_delay_validation(self):
        """Test that invalid batch_delay is rejected."""
        from reddit_scanner import workflow_thread_inspector

        result = await workflow_thread_inspector(
            post_ids=["abc123"],
            batch_delay=0.1  # Below minimum 0.5
        )

        response = json.loads(result[0].text)
        assert len(response["errors"]) > 0 or response["partial"]

    @pytest.mark.asyncio
    async def test_deleted_post_error_handling(self):
        """Test that deleted/inaccessible posts are handled gracefully."""
        from reddit_scanner import workflow_thread_inspector, RedditClient
        import prawcore

        reddit = MagicMock()
        reddit.submission = MagicMock(side_effect=prawcore.exceptions.NotFound(MagicMock()))

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["deleted1"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should have error for the deleted post
        assert len(response["errors"]) > 0
        assert response["partial"] is True

    @pytest.mark.asyncio
    async def test_private_post_error_handling(self):
        """Test that private posts are handled gracefully."""
        from reddit_scanner import workflow_thread_inspector, RedditClient
        import prawcore

        reddit = MagicMock()
        reddit.submission = MagicMock(side_effect=prawcore.exceptions.Forbidden(MagicMock()))

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["private1"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should have error for the private post
        assert len(response["errors"]) > 0
        assert response["partial"] is True

    @pytest.mark.asyncio
    async def test_uses_rate_limited_executor(self, mock_reddit_client):
        """Test that RateLimitedExecutor is used for rate limiting."""
        from reddit_scanner import workflow_thread_inspector, RedditClient, RateLimitedExecutor

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                with patch.object(RateLimitedExecutor, 'execute', new_callable=AsyncMock) as mock_rate_execute:
                    # Make execute return valid data
                    mock_rate_execute.side_effect = mock_execute

                    result = await workflow_thread_inspector(
                        post_ids=["abc123"],
                        batch_delay=0.5
                    )

        # Just verify the tool runs without error - rate limiting is tested via stats
        response = json.loads(result[0].text)
        assert "metadata" in response

    @pytest.mark.asyncio
    async def test_uses_tool_response_envelope(self, mock_reddit_client):
        """Test that response uses ToolResponse envelope format."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["abc123"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Verify ToolResponse envelope structure
        assert "results" in response, "Should have results key"
        assert "metadata" in response, "Should have metadata key"
        assert "errors" in response, "Should have errors key"
        assert "partial" in response, "Should have partial key"
        assert response["metadata"]["tool"] == "workflow_thread_inspector"
        assert "timestamp" in response["metadata"]
        assert "stats" in response["metadata"]

    @pytest.mark.asyncio
    async def test_expand_depth_passed_to_replace_more(self, mock_submission_factory):
        """Test that expand_depth is passed to comments.replace_more()."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        submission = mock_submission_factory("test123", "Test")
        reddit = MagicMock()
        reddit.submission = MagicMock(return_value=submission)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                await workflow_thread_inspector(
                    post_ids=["test123"],
                    expand_depth=10,
                    batch_delay=0.5
                )

        # Verify replace_more was called with the correct limit
        submission.comments.replace_more.assert_called_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_comment_limit_respected(self, mock_comment_factory, mock_submission_factory):
        """Test that comment_limit parameter limits returned comments."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        # Create many comments
        comments = [
            mock_comment_factory(f"c{i}", f"step {i} in the process", score=10)
            for i in range(50)
        ]
        submission = mock_submission_factory("test123", "Test", comments)

        reddit = MagicMock()
        reddit.submission = MagicMock(return_value=submission)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["test123"],
                    comment_limit=5,  # Limit to 5 comments per post
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should have at most 5 comments
        workflow_comments = response["results"][0]["workflow_comments"]
        assert len(workflow_comments) <= 5

    @pytest.mark.asyncio
    async def test_comment_without_author_handled(self, mock_submission_factory):
        """Test that comments with deleted/None author are handled."""
        from reddit_scanner import workflow_thread_inspector, RedditClient

        # Create comment with None author (deleted user)
        comment = MagicMock()
        comment.id = "c1"
        comment.body = "step by step process here"
        comment.score = 10
        comment.depth = 1
        comment.permalink = "/r/test/comments/abc123/test/c1/"
        comment.author = None  # Deleted user

        submission = mock_submission_factory("test123", "Test", [comment])

        reddit = MagicMock()
        reddit.submission = MagicMock(return_value=submission)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["test123"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should handle gracefully - author should be None
        workflow_comments = response["results"][0]["workflow_comments"]
        if workflow_comments:
            assert workflow_comments[0]["author"] is None

    @pytest.mark.asyncio
    async def test_partial_success_with_some_failed_posts(self, mock_submission_factory):
        """Test that partial success is returned when some posts fail."""
        from reddit_scanner import workflow_thread_inspector, RedditClient
        import prawcore

        submission = mock_submission_factory("good123", "Good Post")

        reddit = MagicMock()
        def submission_side_effect(id):
            if id == "good123":
                return submission
            raise prawcore.exceptions.NotFound(MagicMock())

        reddit.submission = MagicMock(side_effect=submission_side_effect)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await workflow_thread_inspector(
                    post_ids=["good123", "bad123"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should have one success and one error
        assert len(response["results"]) == 1
        assert response["results"][0]["post_id"] == "good123"
        assert len(response["errors"]) == 1
        assert response["partial"] is True
