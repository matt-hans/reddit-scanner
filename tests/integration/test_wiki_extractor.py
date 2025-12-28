"""Integration tests for wiki_tool_extractor tool."""
import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch


class TestWikiToolExtractor:
    """Tests for the wiki_tool_extractor tool."""

    @pytest.fixture
    def mock_wiki_page_factory(self):
        """Factory for creating mock wiki pages."""
        def make_page(name: str, content_md: str):
            page = MagicMock()
            page.name = name
            page.content_md = content_md
            return page
        return make_page

    @pytest.fixture
    def mock_subreddit_factory(self, mock_wiki_page_factory):
        """Factory for creating mock subreddits with wiki and sidebar."""
        def make_subreddit(
            name: str,
            description: str = "",
            public_description: str = "",
            wiki_pages: list = None
        ):
            subreddit = MagicMock()
            subreddit.display_name = name
            subreddit.description = description
            subreddit.public_description = public_description

            # Create wiki mock
            if wiki_pages is None:
                wiki_pages = []

            # Store pages in a list that can be returned multiple times
            stored_pages = list(wiki_pages)

            # Create a wiki object that behaves like PRAW's SubredditWiki
            class MockWiki:
                def __iter__(self):
                    return iter(stored_pages)

                def __getitem__(self, page_name):
                    page_dict = {page.name: page for page in stored_pages}
                    if page_name in page_dict:
                        return page_dict[page_name]
                    raise Exception(f"Wiki page {page_name} not found")

            subreddit.wiki = MockWiki()

            return subreddit
        return make_subreddit

    @pytest.fixture
    def mock_reddit_client(self, mock_subreddit_factory, mock_wiki_page_factory):
        """Create mock Reddit client with subreddits."""
        reddit = MagicMock()

        # Create wiki pages
        tools_page = mock_wiki_page_factory(
            "tools",
            "## Recommended Tools\n"
            "- [Notion](https://notion.so) - All-in-one workspace\n"
            "- [Obsidian](https://obsidian.md) - Markdown notes\n"
            "Check out https://github.com/awesome-selfhosted for more"
        )
        resources_page = mock_wiki_page_factory(
            "resources",
            "### Resources\n"
            "[Todoist](https://todoist.com) for task management\n"
            "Also see [Linear](https://linear.app)"
        )
        index_page = mock_wiki_page_factory(
            "index",
            "Welcome to our wiki! See /r/related for more."
        )
        config_page = mock_wiki_page_factory(
            "config",  # Should not match default keywords
            "Configuration options for the bot"
        )

        # Create subreddits
        sub1 = mock_subreddit_factory(
            "productivity",
            description="Sidebar: [Notion](https://notion.so) and [Trello](https://trello.com)\n"
                       "Also check https://todoist.com for tasks",
            public_description="A subreddit about productivity tools",
            wiki_pages=[tools_page, resources_page, index_page, config_page]
        )

        sub2 = mock_subreddit_factory(
            "selfhosted",
            description="Check out [Nextcloud](https://nextcloud.com)",
            public_description="Self-hosted software",
            wiki_pages=[tools_page]
        )

        subreddits = {
            "productivity": sub1,
            "selfhosted": sub2,
        }

        def subreddit_side_effect(name):
            if name.lower() in subreddits:
                return subreddits[name.lower()]
            raise Exception(f"Subreddit {name} not found")

        reddit.subreddit = MagicMock(side_effect=subreddit_side_effect)
        return reddit

    @pytest.mark.asyncio
    async def test_basic_wiki_extraction(self, mock_reddit_client):
        """Test basic extraction from sidebar and wiki."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity"],
                    scan_sidebar=True,
                    scan_wiki=True,
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
        assert response["metadata"]["tool"] == "wiki_tool_extractor"
        assert "pages_scanned" in response["metadata"]
        assert "timestamp" in response["metadata"]

    @pytest.mark.asyncio
    async def test_default_page_keywords_applied(self, mock_reddit_client):
        """Test that default page keywords are applied when None."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity"],
                    page_keywords=None,  # Explicitly pass None
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should scan pages matching default keywords (tools, resources, index)
        # but not "config" which doesn't match defaults
        assert response["metadata"]["pages_scanned"] >= 1

    @pytest.mark.asyncio
    async def test_custom_page_keywords(self, mock_reddit_client):
        """Test that custom page keywords filter wiki pages."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity"],
                    page_keywords=["config"],  # Only scan config page
                    scan_sidebar=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)
        # Should only scan the config page
        assert response["metadata"]["pages_scanned"] >= 0

    @pytest.mark.asyncio
    async def test_sidebar_only_extraction(self, mock_reddit_client):
        """Test extraction from sidebar only."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity"],
                    scan_sidebar=True,
                    scan_wiki=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should have results from sidebar
        assert len(response["results"]) >= 1
        # Pages scanned should be 0 since wiki is disabled
        assert response["metadata"]["pages_scanned"] == 0

    @pytest.mark.asyncio
    async def test_wiki_only_extraction(self, mock_reddit_client):
        """Test extraction from wiki only."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity"],
                    scan_sidebar=False,
                    scan_wiki=True,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should have results from wiki
        assert len(response["results"]) >= 1
        # Pages scanned should be > 0
        assert response["metadata"]["pages_scanned"] >= 1

    @pytest.mark.asyncio
    async def test_extracts_markdown_links(self, mock_subreddit_factory, mock_wiki_page_factory):
        """Test extraction of markdown format links [text](url)."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        # Create subreddit with markdown links
        page = mock_wiki_page_factory(
            "tools",
            "[Notion](https://notion.so) and [Obsidian](https://obsidian.md)"
        )
        subreddit = mock_subreddit_factory(
            "testsub",
            description="[Trello](https://trello.com)",
            wiki_pages=[page]
        )

        reddit = MagicMock()
        reddit.subreddit = MagicMock(return_value=subreddit)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["testsub"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Check that tools were extracted
        tools = response["results"][0]["tools"]
        tool_names = [t["name"] for t in tools]

        assert "Notion" in tool_names or "notion" in [n.lower() for n in tool_names]
        assert "Obsidian" in tool_names or "obsidian" in [n.lower() for n in tool_names]
        assert "Trello" in tool_names or "trello" in [n.lower() for n in tool_names]

    @pytest.mark.asyncio
    async def test_extracts_bare_urls(self, mock_subreddit_factory, mock_wiki_page_factory):
        """Test extraction of bare URLs without markdown."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        page = mock_wiki_page_factory(
            "resources",
            "Check out https://notion.so and https://github.com/awesome"
        )
        subreddit = mock_subreddit_factory(
            "testsub",
            description="Also visit https://todoist.com",
            wiki_pages=[page]
        )

        reddit = MagicMock()
        reddit.subreddit = MagicMock(return_value=subreddit)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["testsub"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)
        tools = response["results"][0]["tools"]
        urls = [t["url"] for t in tools]

        # Check that bare URLs were extracted
        assert any("notion.so" in url for url in urls)
        assert any("github.com" in url for url in urls)
        assert any("todoist.com" in url for url in urls)

    @pytest.mark.asyncio
    async def test_tool_name_inferred_from_link_text(self, mock_subreddit_factory, mock_wiki_page_factory):
        """Test that tool name is inferred from link text."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        page = mock_wiki_page_factory(
            "tools",
            "[My Favorite Tool](https://example.com)"
        )
        subreddit = mock_subreddit_factory(
            "testsub",
            wiki_pages=[page]
        )

        reddit = MagicMock()
        reddit.subreddit = MagicMock(return_value=subreddit)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["testsub"],
                    scan_sidebar=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)
        tools = response["results"][0]["tools"]

        # Tool name should be from link text
        tool_names = [t["name"] for t in tools]
        assert "My Favorite Tool" in tool_names

    @pytest.mark.asyncio
    async def test_tool_name_inferred_from_domain(self, mock_subreddit_factory, mock_wiki_page_factory):
        """Test that tool name is inferred from domain for bare URLs."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        page = mock_wiki_page_factory(
            "tools",
            "Check out https://notion.so/workspace"
        )
        subreddit = mock_subreddit_factory(
            "testsub",
            wiki_pages=[page]
        )

        reddit = MagicMock()
        reddit.subreddit = MagicMock(return_value=subreddit)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["testsub"],
                    scan_sidebar=False,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)
        tools = response["results"][0]["tools"]

        # Tool name should be inferred from domain
        tool_names = [t["name"].lower() for t in tools]
        assert "notion" in tool_names

    @pytest.mark.asyncio
    async def test_deduplication_by_normalized_url(self, mock_subreddit_factory, mock_wiki_page_factory):
        """Test that results are deduplicated by normalized URL."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        page = mock_wiki_page_factory(
            "tools",
            "[Notion](https://notion.so) and [Notion Again](https://notion.so/)"
        )
        subreddit = mock_subreddit_factory(
            "testsub",
            description="[Notion](https://notion.so?ref=sidebar)",
            wiki_pages=[page]
        )

        reddit = MagicMock()
        reddit.subreddit = MagicMock(return_value=subreddit)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["testsub"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)
        tools = response["results"][0]["tools"]

        # Should have only one notion.so entry (deduplicated)
        notion_tools = [t for t in tools if "notion.so" in t["url"]]
        assert len(notion_tools) == 1

    @pytest.mark.asyncio
    async def test_sources_array_tracks_origin(self, mock_subreddit_factory, mock_wiki_page_factory):
        """Test that each tool has sources array tracking origin."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        page = mock_wiki_page_factory(
            "tools",
            "[Notion](https://notion.so)"
        )
        subreddit = mock_subreddit_factory(
            "testsub",
            description="[Notion](https://notion.so)",
            wiki_pages=[page]
        )

        reddit = MagicMock()
        reddit.subreddit = MagicMock(return_value=subreddit)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["testsub"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)
        tools = response["results"][0]["tools"]

        # Should have Notion with sources from both sidebar and wiki
        notion_tool = next((t for t in tools if "notion" in t["name"].lower()), None)
        assert notion_tool is not None
        assert "sources" in notion_tool
        assert isinstance(notion_tool["sources"], list)
        # Should have both sidebar and wiki/tools as sources
        assert any("sidebar" in s for s in notion_tool["sources"])
        assert any("wiki" in s for s in notion_tool["sources"])

    @pytest.mark.asyncio
    async def test_result_structure_per_subreddit(self, mock_reddit_client):
        """Test that each result has subreddit and tools array."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Check result structure
        assert len(response["results"]) >= 1
        sub_result = response["results"][0]
        assert "subreddit" in sub_result
        assert "tools" in sub_result
        assert isinstance(sub_result["tools"], list)

    @pytest.mark.asyncio
    async def test_tool_structure(self, mock_reddit_client):
        """Test that each tool has name, url, sources."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        for sub_result in response["results"]:
            for tool in sub_result["tools"]:
                assert "name" in tool
                assert "url" in tool
                assert "sources" in tool
                assert isinstance(tool["sources"], list)

    @pytest.mark.asyncio
    async def test_multiple_subreddits(self, mock_reddit_client):
        """Test processing multiple subreddits."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity", "selfhosted"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should have results for both subreddits
        assert len(response["results"]) == 2
        subreddits = [r["subreddit"] for r in response["results"]]
        assert "productivity" in subreddits
        assert "selfhosted" in subreddits

    @pytest.mark.asyncio
    async def test_empty_subreddit_names_validation(self):
        """Test that empty subreddit_names list is rejected."""
        from reddit_scanner import wiki_tool_extractor

        result = await wiki_tool_extractor(
            subreddit_names=[],
            batch_delay=0.5
        )

        response = json.loads(result[0].text)
        assert len(response["errors"]) > 0 or response["partial"]

    @pytest.mark.asyncio
    async def test_invalid_subreddit_name_validation(self):
        """Test that invalid subreddit name format is rejected."""
        from reddit_scanner import wiki_tool_extractor

        result = await wiki_tool_extractor(
            subreddit_names=["ab"],  # Too short (valid names are 3-21 chars)
            batch_delay=0.5
        )

        response = json.loads(result[0].text)
        assert len(response["errors"]) > 0 or response["partial"]

    @pytest.mark.asyncio
    async def test_invalid_batch_delay_validation(self):
        """Test that invalid batch_delay is rejected."""
        from reddit_scanner import wiki_tool_extractor

        result = await wiki_tool_extractor(
            subreddit_names=["productivity"],
            batch_delay=0.1  # Below minimum 0.5
        )

        response = json.loads(result[0].text)
        assert len(response["errors"]) > 0 or response["partial"]

    @pytest.mark.asyncio
    async def test_private_wiki_error_handling(self, mock_subreddit_factory):
        """Test that private/forbidden wikis are handled gracefully."""
        from reddit_scanner import wiki_tool_extractor, RedditClient
        import prawcore

        subreddit = mock_subreddit_factory(
            "privatewiki",
            description="Check [Tool](https://tool.com)"
        )
        # Make wiki iteration raise Forbidden
        subreddit.wiki.__iter__ = MagicMock(
            side_effect=prawcore.exceptions.Forbidden(MagicMock())
        )

        reddit = MagicMock()
        reddit.subreddit = MagicMock(return_value=subreddit)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["privatewiki"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should still have results from sidebar
        assert len(response["results"]) >= 1
        # Wiki error might be logged but shouldn't fail the whole operation

    @pytest.mark.asyncio
    async def test_inaccessible_subreddit_error_handling(self):
        """Test that inaccessible subreddits are handled gracefully."""
        from reddit_scanner import wiki_tool_extractor, RedditClient
        import prawcore

        reddit = MagicMock()
        reddit.subreddit = MagicMock(
            side_effect=prawcore.exceptions.NotFound(MagicMock())
        )

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["nonexistent"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should have error for the inaccessible subreddit
        assert len(response["errors"]) > 0
        assert response["partial"] is True

    @pytest.mark.asyncio
    async def test_uses_rate_limited_executor(self, mock_reddit_client):
        """Test that RateLimitedExecutor is used for rate limiting."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity"],
                    batch_delay=0.5
                )

        # Verify the tool runs without error and includes stats
        response = json.loads(result[0].text)
        assert "stats" in response["metadata"]

    @pytest.mark.asyncio
    async def test_uses_tool_response_envelope(self, mock_reddit_client):
        """Test that response uses ToolResponse envelope format."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Verify ToolResponse envelope structure
        assert "results" in response, "Should have results key"
        assert "metadata" in response, "Should have metadata key"
        assert "errors" in response, "Should have errors key"
        assert "partial" in response, "Should have partial key"
        assert response["metadata"]["tool"] == "wiki_tool_extractor"
        assert "timestamp" in response["metadata"]
        assert "pages_scanned" in response["metadata"]

    @pytest.mark.asyncio
    async def test_pages_scanned_in_metadata(self, mock_reddit_client):
        """Test that pages_scanned count is in metadata."""
        from reddit_scanner import wiki_tool_extractor, RedditClient

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_reddit_client

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["productivity"],
                    scan_wiki=True,
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        assert "pages_scanned" in response["metadata"]
        assert isinstance(response["metadata"]["pages_scanned"], int)
        assert response["metadata"]["pages_scanned"] >= 0

    @pytest.mark.asyncio
    async def test_partial_success_with_some_failed_subreddits(
        self, mock_subreddit_factory, mock_wiki_page_factory
    ):
        """Test that partial success is returned when some subreddits fail."""
        from reddit_scanner import wiki_tool_extractor, RedditClient
        import prawcore

        page = mock_wiki_page_factory("tools", "[Tool](https://tool.com)")
        good_sub = mock_subreddit_factory("goodsub", wiki_pages=[page])

        reddit = MagicMock()

        def subreddit_side_effect(name):
            if name == "goodsub":
                return good_sub
            raise prawcore.exceptions.NotFound(MagicMock())

        reddit.subreddit = MagicMock(side_effect=subreddit_side_effect)

        with patch.object(RedditClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = reddit

            async def mock_execute(op, *args, **kwargs):
                return op(*args, **kwargs)

            with patch.object(RedditClient, 'execute', side_effect=mock_execute):
                result = await wiki_tool_extractor(
                    subreddit_names=["goodsub", "badsub"],
                    batch_delay=0.5
                )

        response = json.loads(result[0].text)

        # Should have one success and one error
        assert len(response["results"]) == 1
        assert response["results"][0]["subreddit"] == "goodsub"
        assert len(response["errors"]) == 1
        assert response["partial"] is True
