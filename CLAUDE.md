# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reddit Scanner is an MCP (Model Context Protocol) server that provides AI-powered tools for analyzing Reddit content to discover software development opportunities, user pain points, and market trends. The server uses the PRAW (Python Reddit API Wrapper) library to interact with Reddit's API.

## Development Commands

This project uses `uv` as the Python package manager. Common commands:

```bash
# Install dependencies
uv pip install -e .

# Run the MCP server
python reddit_scanner.py

# Test the MCP server
python test-mcp-client.py
```

## Environment Setup

The Reddit Scanner requires Reddit API credentials. Create a `.env` file with:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=MCP Reddit Analyzer 1.0  # Optional, has default
```

To obtain Reddit API credentials:
1. Go to https://www.reddit.com/prefs/apps
2. Create a new app (script type)
3. Use the provided client ID and secret

## Architecture Overview

### Core Components

1. **MCP Server (reddit_scanner.py)**
   - Uses FastMCP framework for server implementation
   - Implements 10 specialized tools for Reddit analysis
   - Handles Reddit API authentication via environment variables
   - All tools return JSON-formatted results wrapped in TextContent

2. **Tool Categories**
   - **Pain Point Discovery**: `subreddit_pain_point_scanner` - Finds posts expressing frustrations
   - **Solution Seeking**: `solution_request_tracker` - Identifies software recommendation requests
   - **Workflow Analysis**: `user_workflow_analyzer` - Discovers automation opportunities
   - **Competitive Intelligence**: `competitor_mention_monitor` - Tracks competitor limitations
   - **Engagement Metrics**: `subreddit_engagement_analyzer` - Validates problem severity
   - **Community Discovery**: `niche_community_discoverer` - Finds underserved communities
   - **Trend Analysis**: `temporal_trend_analyzer` - Tracks problem evolution
   - **User Research**: `user_persona_extractor` - Builds user profiles
   - **Opportunity Scoring**: `idea_validation_scorer` - Ranks software opportunities

### Key Design Patterns

1. **Global Reddit Instance**: Single authenticated Reddit instance shared across all tools
2. **Error Handling**: Each tool has try-except blocks with logging for resilience
3. **Flexible Parameters**: Most tools have sensible defaults for optional parameters
4. **Batch Processing**: Tools process multiple subreddits/posts and return aggregated results
5. **Rate Limiting Awareness**: Uses PRAW's built-in rate limiting

### Data Flow

1. MCP client sends tool invocation request with parameters
2. Tool authenticates with Reddit API (if not already authenticated)
3. Tool queries Reddit using PRAW methods (search, subreddit operations)
4. Results are processed, filtered, and scored
5. JSON response is formatted and returned via TextContent

### Important Implementation Details

- **Authentication**: Reddit credentials are loaded once from environment variables
- **Search Depth**: Most tools limit results to prevent API exhaustion (typically 50-100 posts)
- **Comment Processing**: Tools that analyze comments use `replace_more(limit=0)` to avoid API overhead
- **Time Decay**: Engagement metrics can apply time-based decay to prioritize recent content
- **Related Subreddit Discovery**: Uses multiple methods including PRAW's recommended API, widget parsing, and keyword extraction

## Testing Approach

The included `test-mcp-client.py` provides a basic MCP client for testing:
- Sends initialization request
- Lists available tools
- Can be extended to test specific tool invocations

For production testing, use the MCP inspector or integrate with an MCP-compatible client.