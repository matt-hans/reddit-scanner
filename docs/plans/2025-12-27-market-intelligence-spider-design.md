# Market Intelligence Spider Design

**Date:** 2025-12-27
**Status:** Approved
**Author:** Brainstorming Session

## Overview

Transform the Reddit Scanner MCP server from a passive scanning tool into an active Market Intelligence Spider with integrated discovery, inspection, and extraction capabilities.

## Pipeline Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Community Spider   │────▶│  Thread Inspector    │────▶│  Wiki Extractor     │
│  (Discover)         │     │  (Analyze Depth)     │     │  (Map Competitors)  │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Breaking changes | Clean break | Simpler codebase, replace old tool |
| Rate limiting | Respectful crawler | Configurable delays, request budgets, prevents bans |
| Spider depth | Depth 1 + hard cap | Predictable results, max 50 communities |
| Workflow signals | Configurable parameter | LLM generates contextual signals |
| Wiki extraction | Comprehensive | Page name + content scan, markdown + bare URLs |
| Response format | Unified envelope | Consistent structure, tool-specific results |
| Error reporting | Per-item + partial flag | Maximum transparency for callers |

## Final Tool List (12 Tools)

| # | Tool | Status |
|---|------|--------|
| 1 | `subreddit_pain_point_scanner` | Existing |
| 2 | `solution_request_tracker` | Existing |
| 3 | `user_workflow_analyzer` | Existing |
| 4 | `competitor_mention_monitor` | Existing |
| 5 | `subreddit_engagement_analyzer` | Existing |
| 6 | `temporal_trend_analyzer` | Existing |
| 7 | `user_persona_extractor` | Existing |
| 8 | `idea_validation_scorer` | Existing |
| 9 | `niche_community_discoverer` | **Replaced** (v2) |
| 10 | `workflow_thread_inspector` | **New** |
| 11 | `wiki_tool_extractor` | **New** |

---

## Tool Specifications

### Tool 1: `niche_community_discoverer` (v2 - Replacement)

**Purpose:** Find communities by topic intent, then spider one hop via sidebar links.

**Signature:**
```python
@mcp.tool()
async def niche_community_discoverer(
    topic_keywords: List[str],          # Intent-based search terms
    min_subscribers: int = 5000,
    max_subscribers: int = 200000,
    max_communities: int = 50,          # Hard cap safety valve
    spider_sidebar: bool = True,        # Enable/disable sidebar crawl
    batch_delay: float = 1.5            # Seconds between API batches
) -> List[TextContent]:
```

**Algorithm:**
1. Primary Search Phase
   - For each keyword, call `reddit.subreddits.search(keyword, limit=20)`
   - Filter by subscriber range
   - Deduplicate by `display_name`
   - Stop if `max_communities` reached

2. Sidebar Spider Phase (if enabled and under cap)
   - For each discovered community, fetch `subreddit.description`
   - Extract `r/CommunityName` patterns via regex
   - Validate each: exists, public, within subscriber range
   - Add to results with `source: "sidebar of {parent}"`

**Response:**
```json
{
    "results": [
        {"name": "...", "subscribers": 12345, "source": "search|sidebar of X", "description": "..."}
    ],
    "metadata": {"tool": "niche_community_discoverer", "keywords_searched": [...], "timestamp": "..."},
    "errors": [{"item": "keyword", "reason": "..."}],
    "partial": false
}
```

---

### Tool 2: `workflow_thread_inspector`

**Purpose:** Expand nested comment trees to find workflow details buried in replies.

**Signature:**
```python
@mcp.tool()
async def workflow_thread_inspector(
    post_ids: List[str],                           # Reddit post IDs to analyze
    workflow_signals: List[str] = None,            # LLM-generated contextual keywords
    comment_limit: int = 100,                      # Max comments to return per post
    expand_depth: int = 5,                         # How many "load more" expansions
    min_score: int = 1,                            # Filter low-quality comments
    batch_delay: float = 1.5
) -> List[TextContent]:
```

**Default Signals:**
```python
["step", "process", "export", "import", "csv", "manual", "copy", "paste",
 "click", "download", "upload", "workaround", "hack", "tedious"]
```

**Algorithm:**
1. For each post_id, fetch submission
2. Call `comments.replace_more(limit=expand_depth)` to flatten tree
3. Filter by min_score and signal keyword presence
4. Capture: author, body (up to 1000 chars), score, depth, permalink

**Response:**
```json
{
    "results": [
        {
            "post_id": "...",
            "post_title": "...",
            "post_url": "...",
            "workflow_comments": [
                {"author": "...", "body": "...", "score": 5, "depth": 2, "url": "..."}
            ]
        }
    ],
    "metadata": {"tool": "workflow_thread_inspector", "signals_used": [...], "timestamp": "..."},
    "errors": [{"item": "post_id", "reason": "..."}],
    "partial": false
}
```

---

### Tool 3: `wiki_tool_extractor`

**Purpose:** Scan subreddit wikis and sidebars for mentioned tools and software.

**Signature:**
```python
@mcp.tool()
async def wiki_tool_extractor(
    subreddit_names: List[str],                    # Communities to scan
    scan_sidebar: bool = True,                     # Include sidebar/description
    scan_wiki: bool = True,                        # Include wiki pages
    page_keywords: List[str] = None,               # Wiki page name filters
    batch_delay: float = 1.5
) -> List[TextContent]:
```

**Default Page Keywords:**
```python
["tools", "software", "resources", "guide", "faq", "index", "wiki", "recommended"]
```

**Algorithm:**
1. Sidebar Scan (if enabled)
   - Fetch `subreddit.description` and `subreddit.public_description`
   - Extract markdown links and bare URLs
   - Infer tool name from domain

2. Wiki Scan (if enabled)
   - List wiki pages via `subreddit.wiki`
   - Filter by page_keywords OR scan all pages
   - Extract links from content

3. Deduplicate by normalized URL

**Response:**
```json
{
    "results": [
        {
            "subreddit": "...",
            "tools": [
                {"name": "Notion", "url": "https://notion.so", "sources": ["sidebar", "wiki/tools"]}
            ]
        }
    ],
    "metadata": {"tool": "wiki_tool_extractor", "pages_scanned": 5, "timestamp": "..."},
    "errors": [{"item": "subreddit", "reason": "..."}],
    "partial": false
}
```

---

## Shared Infrastructure

### RateLimitConfig

```python
@dataclass
class RateLimitConfig:
    batch_delay: float = 1.5          # Seconds between batches
    request_budget: int = 100         # Max requests per tool call
    budget_exhausted_action: str = "stop"
```

### RateLimitedExecutor

```python
class RateLimitedExecutor:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests_made = 0
        self.last_request_time = 0

    async def execute(self, operation, *args, **kwargs):
        # Check budget
        if self.requests_made >= self.config.request_budget:
            raise BudgetExhaustedError(...)

        # Enforce delay
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.batch_delay:
            await asyncio.sleep(self.config.batch_delay - elapsed)

        # Execute via existing client
        result = await RedditClient.execute(operation, *args, **kwargs)

        self.requests_made += 1
        self.last_request_time = time.time()
        return result
```

### ToolResponse Envelope

```python
@dataclass
class ToolResponse:
    tool_name: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    partial: bool = False

    def add_result(self, item: Dict[str, Any]):
        self.results.append(item)

    def add_error(self, item: str, reason: str):
        self.errors.append({"item": item, "reason": reason})
        self.partial = True

    def to_response(self) -> List[TextContent]:
        envelope = {
            "results": self.results,
            "metadata": {
                "tool": self.tool_name,
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats
            },
            "errors": self.errors,
            "partial": self.partial
        }
        return safe_json_response(envelope)
```

### New Validators

```python
def validate_keyword(keyword: str) -> bool:
    if not keyword or not isinstance(keyword, str):
        return False
    return bool(re.match(r'^[\w\s\-\.]{2,50}$', keyword))

def validate_keyword_list(keywords: List[str]) -> bool:
    if not isinstance(keywords, list) or not keywords:
        return False
    return all(validate_keyword(kw) for kw in keywords)

def validate_batch_delay(delay: float) -> bool:
    return isinstance(delay, (int, float)) and 0.5 <= delay <= 10.0
```

---

## Testing Strategy

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── unit/
│   ├── test_validation.py
│   ├── test_rate_limiting.py
│   ├── test_tool_response.py
│   ├── test_serializers.py
│   └── test_helpers.py
├── integration/
│   ├── test_pain_point_scanner.py
│   ├── test_solution_tracker.py
│   ├── test_workflow_analyzer.py
│   ├── test_competitor_monitor.py
│   ├── test_engagement_analyzer.py
│   ├── test_community_discoverer.py
│   ├── test_trend_analyzer.py
│   ├── test_persona_extractor.py
│   ├── test_validation_scorer.py
│   ├── test_thread_inspector.py
│   └── test_wiki_extractor.py
├── mocked/
│   └── test_api_failures.py
└── e2e/
    └── test_pipeline.py
```

### Run Commands

```bash
pytest tests/ -v                              # Full suite
pytest tests/integration/ -v -m integration   # Integration only
pytest tests/unit/ -v                         # Unit tests only
pytest tests/ --cov=reddit_scanner --cov-report=html  # With coverage
```

---

## Implementation Order

1. Shared infrastructure (RateLimitConfig, RateLimitedExecutor, ToolResponse)
2. New validators (keyword_list, batch_delay)
3. `niche_community_discoverer` v2 (replace existing)
4. `workflow_thread_inspector` (new)
5. `wiki_tool_extractor` (new)
6. Comprehensive test suite for all 12 tools
7. (Optional) Migrate existing tools to ToolResponse envelope
