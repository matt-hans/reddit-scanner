# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reddit Scanner is an MCP (Model Context Protocol) server that provides AI-powered tools for analyzing Reddit content to discover software development opportunities, user pain points, and market trends. The server uses the PRAW (Python Reddit API Wrapper) library to interact with Reddit's API, enhanced with spaCy NLP for advanced text analysis and semantic understanding.

## Development Commands

This project uses `uv` as the Python package manager. Common commands:

```bash
# Install dependencies
uv pip install -e .

# Download spaCy language models (CRITICAL for NLP features)
python -m spacy download en_core_web_lg    # Recommended: Best accuracy (~800MB)
python -m spacy download en_core_web_md    # Alternative: Good accuracy (~50MB)  
python -m spacy download en_core_web_sm    # Fallback: Basic accuracy (~15MB)

# Run the MCP server
python reddit_scanner.py

# Test the MCP server  
python test-mcp-client.py

# Check NLP setup status
# Run the server and look for "=== NLP SETUP STATUS ===" in logs
```

## spaCy Model Setup (IMPORTANT)

**⚠️ Without spaCy models, NLP features will be disabled and tools will fall back to basic keyword detection.**

### Quick Setup (Recommended)
```bash
# Install the large model for best results
python -m spacy download en_core_web_lg
```

### Troubleshooting NLP Issues

#### Problem: Tools show `"nlp_enhanced": false`
**Solution**: Install spaCy language models
```bash
python -m spacy download en_core_web_lg
# Then restart the server
```

#### Problem: "No spaCy model found" error  
**Solution**: Install any spaCy model
```bash
# Try each until one works:
python -m spacy download en_core_web_lg  # Best choice
python -m spacy download en_core_web_md  # Good alternative  
python -m spacy download en_core_web_sm  # Basic fallback
```

#### Problem: Slow NLP performance
**Solution**: Use a smaller model or disable clustering
```bash
python -m spacy download en_core_web_sm  # Faster but less accurate
```

#### Problem: "Limited features" warning
**Solution**: The small model lacks word vectors, install a larger model
```bash
python -m spacy download en_core_web_md  # Has word vectors for clustering
```

### Verifying NLP Setup
Use the diagnostic tool to check your setup:
```bash
# Call the nlp_health_check tool through your MCP client
# It will test entity extraction, sentiment analysis, and clustering
```

### Model Comparison
| Model | Size | Features | Use Case |
|-------|------|----------|----------|
| `en_core_web_lg` | ~800MB | Full NLP + vectors | Production, best accuracy |
| `en_core_web_md` | ~50MB | Most NLP + vectors | Development, good balance |
| `en_core_web_sm` | ~15MB | Basic NLP only | Testing, limited features |

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
   - Implements 11 specialized tools for Reddit analysis (10 enhanced + 1 new)
   - Handles Reddit API authentication via environment variables
   - Integrates spaCy NLP for advanced text analysis
   - All tools return JSON-formatted results wrapped in TextContent

2. **NLP Enhancement Layer**
   - **spaCy Integration**: Uses en_core_web_lg model for best accuracy
   - **Named Entity Recognition**: Automatically extracts companies, products, money amounts
   - **Dependency Parsing**: Understands grammatical relationships for better context
   - **Semantic Similarity**: Clusters related content using word vectors
   - **Advanced Sentiment Analysis**: Goes beyond keywords using contextual understanding
   - **Fallback Mechanisms**: Gracefully degrades to traditional methods if NLP unavailable

3. **Enhanced Tool Categories**
   - **Pain Point Discovery**: `subreddit_pain_point_scanner` - NLP-enhanced with dependency parsing and clustering
   - **Solution Seeking**: `solution_request_tracker` - Advanced request classification and entity extraction
   - **Workflow Analysis**: `user_workflow_analyzer` - Automated step extraction and automation scoring
   - **Competitive Intelligence**: `competitor_mention_monitor` - NER-based competitor discovery and feature analysis
   - **Content Clustering**: `nlp_content_clusterer` - NEW: Semantic clustering of Reddit content
   - **Engagement Metrics**: `subreddit_engagement_analyzer` - Validates problem severity
   - **Community Discovery**: `niche_community_discoverer` - Finds underserved communities
   - **Trend Analysis**: `temporal_trend_analyzer` - Tracks problem evolution
   - **User Research**: `user_persona_extractor` - Builds user profiles
   - **Opportunity Scoring**: `idea_validation_scorer` - Ranks software opportunities

### Key Design Patterns

1. **Global Reddit Instance**: Single authenticated Reddit instance shared across all tools
2. **Global spaCy Instance**: Lazy-loaded spaCy NLP model with automatic fallbacks
3. **Hybrid Analysis**: Combines traditional keyword matching with advanced NLP for robustness
4. **Error Handling**: Each tool has try-except blocks with logging for resilience
5. **Flexible Parameters**: Most tools have sensible defaults for optional parameters
6. **Batch Processing**: Tools process multiple subreddits/posts and return aggregated results
7. **Rate Limiting Awareness**: Uses PRAW's built-in rate limiting
8. **NLP Performance Optimization**: Uses spaCy's pipe() for efficient batch processing
9. **User-Friendly Status Communication**: All tools include nlp_status with setup guidance
10. **Smart Tool Defaults**: Simplified tools automatically configure optimal settings

### Simplified "Smart" Tools (NEW)

For ease of use, the system now includes simplified tools with intelligent defaults:

#### **`smart_pain_point_scanner(subreddit_names, time_filter='month', focus='all')`**
- Automatically configures NLP settings based on availability
- Focus options: 'pain_points', 'solutions', 'workflows', 'all'
- Optimizes parameters based on focus area

#### **`smart_solution_finder(subreddit_names, time_window='month')`**  
- Auto-tracks solution requests with optimal settings
- Automatically uses NLP when available
- Balanced confidence thresholds for best results

#### **`quick_market_analysis(subreddit_names, analysis_type='comprehensive')`**
- One-click comprehensive market analysis
- Combines multiple tools for complete insights  
- Analysis types: 'quick', 'comprehensive', 'competitive'
- Automatically includes clustering when NLP available

#### **`nlp_health_check()`**
- Diagnostic tool for troubleshooting NLP setup
- Tests entity extraction, sentiment analysis, and clustering
- Provides detailed status and functionality verification

### Data Flow

1. MCP client sends tool invocation request with parameters
2. Tool authenticates with Reddit API (if not already authenticated)
3. Tool initializes spaCy NLP model (if not already loaded)
4. Tool queries Reddit using PRAW methods (search, subreddit operations)
5. Raw text is processed through both traditional and NLP analysis pipelines
6. Results are enhanced with entities, sentiment, patterns, and clustering
7. JSON response is formatted and returned via TextContent

### Important Implementation Details

- **Authentication**: Reddit credentials are loaded once from environment variables
- **NLP Model Loading**: spaCy models are lazy-loaded with fallback hierarchy (lg → md → sm)
- **Search Depth**: Most tools limit results to prevent API exhaustion (typically 50-100 posts)
- **Comment Processing**: Tools that analyze comments use `replace_more(limit=0)` to avoid API overhead
- **Time Decay**: Engagement metrics can apply time-based decay to prioritize recent content
- **Related Subreddit Discovery**: Uses multiple methods including PRAW's recommended API, widget parsing, and keyword extraction
- **Performance Considerations**: NLP processing is optional and can be disabled for faster execution
- **Memory Management**: Large spaCy models require ~500MB RAM; use smaller models on constrained systems

## NLP Enhancement Features

### spaCy Integration

The Reddit Scanner now includes comprehensive spaCy NLP integration that transforms it from a keyword-based tool to a sophisticated language understanding system.

#### NLP Helper Functions

1. **`extract_entities_from_text()`** - Named Entity Recognition
   - Extracts companies (ORG), products (PRODUCT), money amounts (MONEY)
   - Identifies technology-related entities automatically

2. **`extract_pain_points_nlp()`** - Advanced Pain Point Detection  
   - Uses dependency parsing to find "struggling with X" patterns
   - Identifies "X is difficult/broken" constructions
   - Provides confidence scores for each detection

3. **`calculate_sentiment_nlp()`** - Contextual Sentiment Analysis
   - Uses word vector similarity for semantic sentiment analysis
   - Falls back to keyword-based analysis if vectors unavailable
   - More accurate than simple keyword matching

4. **`extract_workflow_steps_nlp()`** - Workflow Step Extraction
   - Uses sentence segmentation and dependency parsing
   - Identifies action-object relationships in imperative sentences
   - Calculates complexity scores for automation potential

5. **`classify_request_type_nlp()`** - Request Classification
   - Categorizes requests: tool_request, how_to, comparison, troubleshooting
   - Uses both pattern matching and POS tagging
   - Returns confidence scores and classification indicators

6. **`cluster_similar_texts()`** - Semantic Clustering
   - Groups similar content using spaCy document similarity
   - Configurable similarity thresholds
   - Efficient batch processing with spaCy's pipe()

#### Enhanced Tool Capabilities

**`subreddit_pain_point_scanner`** (Enhanced)
- New parameters: `use_nlp`, `cluster_similar`
- NLP-detected pain points with confidence scores
- Automatic entity extraction (companies, budget mentions)
- Semantic clustering of similar issues
- Maintains backward compatibility with keyword detection

**`competitor_mention_monitor`** (Enhanced)  
- New parameters: `use_nlp`, `auto_discover_competitors`
- Automatic competitor discovery via NER
- Feature extraction and missing feature identification
- Advanced sentiment analysis for competitor mentions
- Budget and pricing information extraction

**`user_workflow_analyzer`** (Enhanced)
- New parameters: `use_nlp`, `automation_score_threshold`
- Automated workflow step extraction from natural language
- Automation potential scoring (0-1 scale)
- Time mention extraction and complexity analysis
- Tool/technology identification in workflows

**`solution_request_tracker`** (Enhanced)
- New parameters: `use_nlp`, `confidence_threshold`
- Advanced request type classification
- Detailed requirement extraction using dependency parsing
- Budget analysis and urgency detection
- Enhanced categorization with confidence scores

**`nlp_content_clusterer`** (New Tool)
- Semantic clustering of Reddit posts and comments
- Trending topic identification
- Entity-based trend analysis
- Configurable similarity thresholds and cluster sizes
- Comprehensive clustering analytics

#### Performance and Reliability

- **Model Fallback**: Automatically tries en_core_web_lg → en_core_web_md → en_core_web_sm
- **Graceful Degradation**: Falls back to traditional methods if NLP fails
- **Optional NLP**: All tools have `use_nlp` parameter to disable NLP processing
- **Batch Processing**: Uses spaCy's `pipe()` for efficient processing of multiple texts
- **Error Handling**: Comprehensive error handling with logging for debugging

#### Setup Requirements

```bash
# Install the spaCy models (choose based on your system capabilities)
python -m spacy download en_core_web_lg    # Best accuracy, ~800MB
python -m spacy download en_core_web_md    # Good accuracy, ~50MB  
python -m spacy download en_core_web_sm    # Basic accuracy, ~15MB
```

## Testing Approach

### Comprehensive Test Suite

The project includes a comprehensive test suite (`test_reddit_scanner.py`) that covers all functionality:

```bash
# Install test dependencies
python run_tests.py install
# OR manually: uv pip install -e ".[test]"

# Run all tests
python run_tests.py all

# Run specific test categories
python run_tests.py auth        # Authentication tests only
python run_tests.py nlp         # NLP-related tests only  
python run_tests.py tools       # MCP tool tests only
python run_tests.py basic       # Basic functionality tests

# Run with coverage report
python run_tests.py coverage
```

### Test Coverage

The test suite includes **80+ comprehensive test cases** covering:

#### **Authentication & Initialization (8 tests)**
- Missing Reddit credentials validation
- Environment variable handling
- Default vs custom user agent
- Singleton behavior verification

#### **spaCy Model Loading (6 tests)**
- Large model loading success
- Fallback chain: lg → md → sm → None
- Error handling for missing models
- Singleton behavior verification

#### **NLP Helper Functions (30+ tests)**
- Entity extraction with/without spaCy
- Pain point detection using dependency parsing
- Sentiment analysis with vector fallbacks
- Workflow step extraction and complexity scoring
- Request classification with confidence thresholds
- Text clustering with similarity thresholds

#### **Enhanced MCP Tools (25+ tests)**
- All 5 enhanced tools with `use_nlp=True/False`
- Backward compatibility verification
- Combined traditional + NLP results
- JSON response format validation

#### **Error Handling & Edge Cases (15+ tests)**
- Reddit API failures and timeouts
- spaCy model unavailability scenarios
- Malformed data handling
- Empty inputs and boundary conditions
- Graceful degradation testing

### Testing Framework

- **pytest** + **pytest-asyncio** for async MCP tool testing
- **unittest.mock** for comprehensive API mocking
- **pytest-cov** for coverage reporting
- Custom fixtures for environment setup
- Parameterized tests for NLP on/off scenarios

### Manual Testing

For interactive testing, use the included `test-mcp-client.py`:
- Sends initialization request
- Lists available tools
- Can be extended to test specific tool invocations

For production testing, use the MCP inspector or integrate with an MCP-compatible client.

### Test Execution Examples

```bash
# Quick test run
python -m pytest test_reddit_scanner.py -v

# With coverage report
python -m pytest test_reddit_scanner.py --cov=reddit_scanner --cov-report=html

# Run only authentication tests
python -m pytest test_reddit_scanner.py::TestAuthentication -v

# Run tests matching pattern
python -m pytest test_reddit_scanner.py -k "nlp" -v
```