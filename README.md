# Reddit Scanner MCP Server

An MCP (Model Context Protocol) server that provides AI-powered tools for analyzing Reddit content to discover software development opportunities, user pain points, and market trends.

## Features

- **Pain Point Discovery**: Automatically identify user frustrations and problems across subreddits
- **Solution Request Tracking**: Find posts where users are actively seeking software recommendations
- **Workflow Analysis**: Discover repetitive tasks and automation opportunities
- **Competitive Intelligence**: Monitor competitor mentions and limitations
- **Community Discovery**: Find niche communities with unmet software needs
- **Trend Analysis**: Track the evolution of problems and emerging needs over time
- **User Persona Building**: Extract user profiles and behavioral patterns
- **Opportunity Scoring**: Validate and rank software ideas based on market signals

## Installation

### Prerequisites

- Python 3.12 or higher
- Reddit API credentials (free)
- uv package manager (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd reddit-scanner
```

2. Install dependencies using uv:
```bash
uv pip install -e .
```

Or using pip:
```bash
pip install -e .
```

3. Create a Reddit application:
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Choose "script" as the application type
   - Note your client ID (under "personal use script") and client secret

4. Configure environment variables:
Create a `.env` file in the project root:
```env
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=MCP Reddit Analyzer 1.0  # Optional
```

## Usage

### Running the Server

Start the MCP server:
```bash
python reddit_scanner.py
```

The server will run in stdio transport mode, ready to receive MCP commands.

### Integration with MCP Clients

To use this server with an MCP client (like Claude Desktop), add it to your MCP configuration:

```json
{
  "reddit-scanner": {
    "command": "python",
    "args": ["/path/to/reddit_scanner.py"],
    "env": {
      "REDDIT_CLIENT_ID": "your_client_id",
      "REDDIT_CLIENT_SECRET": "your_client_secret"
    }
  }
}
```

### Testing

Run the included test client:
```bash
python test-mcp-client.py
```

## Available Tools

### 1. subreddit_pain_point_scanner
Scans subreddits for posts and comments expressing problems or frustrations.

**Parameters:**
- `subreddit_names`: List of subreddit names to scan
- `time_filter`: Time period ('hour', 'day', 'week', 'month', 'year', 'all')
- `limit`: Maximum posts to analyze per subreddit
- `pain_keywords`: Custom keywords indicating pain points
- `min_score`: Minimum score threshold
- `include_comments`: Whether to scan comments
- `comment_depth`: How deep to search comment threads

**Example Use Case**: Find common frustrations in r/webdev and r/programming about deployment workflows.

### 2. solution_request_tracker
Identifies posts where users are asking for software or tool recommendations.

**Parameters:**
- `subreddit_names`: Target subreddits
- `request_patterns`: Regex patterns for requests
- `exclude_solved`: Filter out already solved posts
- `min_engagement`: Minimum comment count
- `time_window`: Time period to search
- `category_keywords`: Keywords to categorize requests

**Example Use Case**: Track requests for automation tools in productivity subreddits.

### 3. user_workflow_analyzer
Analyzes posts describing workflows to find automation opportunities.

**Parameters:**
- `subreddit_names`: Subreddits to analyze
- `workflow_indicators`: Keywords like "process", "steps"
- `min_steps`: Minimum workflow complexity
- `efficiency_keywords`: Keywords indicating inefficiency

**Example Use Case**: Discover repetitive tasks in r/dataengineering that could be automated.

### 4. competitor_mention_monitor
Tracks mentions of competitors and their limitations.

**Parameters:**
- `competitor_names`: Software names to monitor
- `sentiment_threshold`: Sentiment score filter
- `limitation_keywords`: Keywords indicating problems

**Example Use Case**: Monitor discussions about Notion's limitations in productivity subreddits.

### 5. subreddit_engagement_analyzer
Analyzes engagement patterns to validate problem severity.

**Parameters:**
- `post_ids`: Reddit post IDs to analyze
- `engagement_metrics`: Metrics to calculate
- `time_decay`: Apply time-based decay to scores

**Example Use Case**: Validate which pain points get the most community engagement.

### 6. niche_community_discoverer
Finds smaller, specialized subreddits with unmet software needs.

**Parameters:**
- `seed_subreddits`: Starting subreddits
- `min_subscribers`: Minimum community size
- `max_subscribers`: Maximum community size
- `activity_threshold`: Minimum posts per day
- `related_depth`: How deep to explore related subreddits

**Example Use Case**: Discover niche communities related to r/productivity that might need specialized tools.

### 7. temporal_trend_analyzer
Tracks how problems and discussions evolve over time.

**Parameters:**
- `subreddit_names`: Subreddits to monitor
- `time_periods`: Time intervals to compare
- `trend_keywords`: Keywords to track
- `growth_threshold`: Minimum growth rate

**Example Use Case**: Identify emerging problems in r/webdev over the past month.

### 8. user_persona_extractor
Builds user profiles from Reddit activity patterns.

**Parameters:**
- `user_sample_size`: Number of users to analyze
- `activity_depth`: Posts/comments per user
- `need_categories`: Categories to track

**Example Use Case**: Understand the typical profile of users seeking productivity tools.

### 9. idea_validation_scorer
Scores and ranks software opportunities based on multiple factors.

**Parameters:**
- `pain_point_data`: Data about identified pain points
- `market_size_weight`: Weight for market size factor
- `severity_weight`: Weight for problem severity
- `competition_weight`: Weight for competition level
- `technical_feasibility`: Feasibility scores

**Example Use Case**: Rank discovered opportunities by market potential and feasibility.

## Example Workflows

### Finding SaaS Opportunities

1. Use `subreddit_pain_point_scanner` to identify frustrations in your target market
2. Run `solution_request_tracker` to find explicit software requests
3. Apply `competitor_mention_monitor` to understand existing solution limitations
4. Use `idea_validation_scorer` to rank opportunities

### Market Research

1. Start with `niche_community_discoverer` to find relevant communities
2. Use `user_persona_extractor` to understand your target users
3. Apply `temporal_trend_analyzer` to identify growing problems
4. Validate with `subreddit_engagement_analyzer`

### Competitive Analysis

1. Use `competitor_mention_monitor` to track competitor discussions
2. Apply `subreddit_pain_point_scanner` with competitor-related keywords
3. Run `user_workflow_analyzer` to find workflows using competitor tools

## Output Format

All tools return JSON-formatted results with relevant data:

```json
{
  "pain_points": [...],
  "category_counts": {...},
  "total_found": 42
}
```

Results include URLs, scores, and extracted insights for easy analysis.

## Rate Limits and Best Practices

- Reddit API has rate limits (60 requests per minute for OAuth)
- The server uses PRAW's built-in rate limiting
- Start with smaller `limit` values to test
- Cache results when doing repeated analysis
- Respect Reddit's terms of service

## Troubleshooting

### "Reddit API credentials not found"
Ensure your `.env` file exists and contains valid `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`.

### Empty results
- Check if the subreddit names are spelled correctly
- Try increasing the `limit` parameter
- Verify your search timeframe isn't too narrow
- Some subreddits may be private or quarantined

### Rate limit errors
- Reduce the number of simultaneous requests
- Add delays between tool calls
- Use smaller `limit` values

## Development

### Project Structure
```
reddit-scanner/
   reddit_scanner.py     # Main MCP server implementation
   test-mcp-client.py    # Test client
   pyproject.toml        # Project configuration
   .env                  # Environment variables (create this)
   README.md            # This file
```

### Adding New Tools

1. Create a new function decorated with `@mcp.tool()`
2. Add comprehensive docstring with parameter descriptions
3. Implement Reddit API calls using the global `reddit` instance
4. Return results as `List[TextContent]` with JSON data
5. Add error handling and logging

## Contributing

Contributions are welcome! Please:
- Follow the existing code style
- Add comprehensive docstrings
- Include error handling
- Test with the MCP client before submitting

## License

[Add your license here]

## Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp)
- Uses [PRAW](https://praw.readthedocs.io/) for Reddit API access
- Implements the [Model Context Protocol](https://modelcontextprotocol.io/)