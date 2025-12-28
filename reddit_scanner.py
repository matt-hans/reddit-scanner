import os
import re
import json
import asyncio
import functools
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import logging

import praw
import prawcore
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Rate Limiting Infrastructure
class BudgetExhaustedError(Exception):
    """Raised when request budget is exhausted."""
    pass


@dataclass
class RateLimitConfig:
    """Configuration for respectful crawling."""
    batch_delay: float = 1.5          # Seconds between requests
    request_budget: int = 100         # Max requests per tool call
    budget_exhausted_action: str = "stop"  # Currently only "stop" is implemented


class RateLimitedExecutor:
    """Wraps operations with budgeting and delays."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests_made = 0
        self.last_request_time = 0.0

    async def execute(self, operation, *args, **kwargs) -> Any:
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


@dataclass
class ToolResponse:
    """Unified response envelope for all MCP tools.

    Provides consistent structure for tool responses including results,
    errors, metadata, and partial completion status.
    """
    tool_name: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    partial: bool = False
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, item: Dict[str, Any]) -> None:
        """Add a result item to the response.

        Args:
            item: Dictionary containing result data
        """
        self.results.append(item)

    def add_error(self, item: str, reason: str) -> None:
        """Add an error to the response and mark as partial.

        Args:
            item: Identifier for the item that caused the error
            reason: Description of what went wrong
        """
        self.errors.append({"item": item, "reason": reason})
        self.partial = True

    def to_response(self) -> List[TextContent]:
        """Convert to MCP TextContent response with envelope structure.

        Returns:
            List containing a single TextContent with JSON envelope
        """
        metadata = {
            "tool": self.tool_name,
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats,
            **self.extra_metadata
        }

        envelope = {
            "results": self.results,
            "metadata": metadata,
            "errors": self.errors,
            "partial": self.partial
        }
        return safe_json_response(envelope)


class RedditClient:
    """Robust Reddit client with health checks, retry logic, and rate limiting."""
    
    _instance: Optional[praw.Reddit] = None
    _lock = asyncio.Lock()
    _semaphore = asyncio.Semaphore(int(os.getenv("REDDIT_CONCURRENCY", "8")))
    _last_health_check = 0
    _health_check_interval = 300  # 5 minutes
    
    @classmethod
    async def get(cls) -> praw.Reddit:
        """Get or create Reddit instance with health checks."""
        async with cls._lock:
            current_time = time.time()
            
            # Check if we need to refresh the instance
            if (cls._instance is None or 
                current_time - cls._last_health_check > cls._health_check_interval):
                
                if cls._instance and not await cls._ping(cls._instance):
                    logger.warning("Reddit client health check failed, creating new instance")
                    cls._instance = None
                
                if cls._instance is None:
                    cls._instance = cls._build()
                    
                cls._last_health_check = current_time
                
            return cls._instance
    
    @classmethod
    def _build(cls) -> praw.Reddit:
        """Build new Reddit instance with validation."""
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "MCP Reddit Analyzer 1.0")
        
        if not client_id or not client_secret:
            raise ValueError(
                "Reddit API credentials not found. Please set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET environment variables in your .env file."
            )
        
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            timeout=30,
        )
    
    @classmethod
    async def _ping(cls, reddit_instance: praw.Reddit) -> bool:
        """Ping Reddit API to check connection health."""
        try:
            # Use thread pool to run blocking operation
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                lambda: reddit_instance.user.me()
            )
            return True
        except Exception as e:
            logger.warning(f"Reddit ping failed: {e}")
            return False
    
    @classmethod
    async def execute(cls, operation, *args, **kwargs):
        """Execute PRAW operation with rate limiting and error handling."""
        async with cls._semaphore:
            reddit = await cls.get()
            loop = asyncio.get_running_loop()
            
            for attempt in range(3):  # Retry up to 3 times
                try:
                    return await loop.run_in_executor(
                        None, 
                        functools.partial(operation, *args, **kwargs)
                    )
                except (prawcore.exceptions.ServerError, 
                        prawcore.exceptions.TooManyRequests) as e:
                    if attempt < 2:
                        delay = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Reddit API error (attempt {attempt + 1}): {e}. Retrying in {delay}s")
                        await asyncio.sleep(delay)
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Reddit operation failed: {e}")
                    raise

# JSON Serialization Utilities
class SafeJSONEncoder:
    """Safe JSON serialization handling datetime and PRAW objects."""
    
    @staticmethod
    def serialize_post(post) -> Dict[str, Any]:
        """Convert PRAW post to JSON-safe dictionary."""
        return {
            "id": post.id,
            "title": post.title,
            "author": post.author.name if post.author else None,
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "created_utc": post.created_utc,
            "url": f"https://reddit.com{post.permalink}",
            "selftext": post.selftext[:500] if post.selftext else "",
        }
    
    @staticmethod
    def serialize_comment(comment) -> Dict[str, Any]:
        """Convert PRAW comment to JSON-safe dictionary."""
        return {
            "id": comment.id,
            "author": comment.author.name if comment.author else None,
            "body": comment.body[:500] if hasattr(comment, 'body') else "",
            "score": comment.score,
            "created_utc": comment.created_utc,
            "url": f"https://reddit.com{comment.permalink}",
        }
    
    @staticmethod
    def serialize_subreddit(subreddit) -> Dict[str, Any]:
        """Convert PRAW subreddit to JSON-safe dictionary."""
        return {
            "name": subreddit.display_name,
            "subscribers": subreddit.subscribers,
            "description": subreddit.public_description[:200] if subreddit.public_description else "",
            "url": f"https://reddit.com/r/{subreddit.display_name}",
        }

def safe_json_response(data: Dict[str, Any]) -> List[TextContent]:
    """Create safe JSON response with proper error handling."""
    try:
        # Use json.dumps with default=str to handle remaining edge cases
        json_str = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        return [TextContent(type="text", text=json_str)]
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        error_response = {
            "error": "Serialization failed",
            "message": str(e),
            "data_summary": f"Response contained {len(data)} top-level keys"
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

# Input Validation
class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass

def validate_subreddit_name(name: str) -> bool:
    """Validate subreddit name format."""
    if not name or not isinstance(name, str):
        return False
    # Reddit subreddit names: 3-21 chars, letters, numbers, underscores
    return bool(re.match(r'^[A-Za-z0-9_]{3,21}$', name))

def validate_post_id(post_id: str) -> bool:
    """Validate Reddit post ID format."""
    if not post_id or not isinstance(post_id, str):
        return False
    # Reddit post IDs are alphanumeric, typically 6-7 characters
    return bool(re.match(r'^[A-Za-z0-9]{6,10}$', post_id))

def validate_user_name(username: str) -> bool:
    """Validate Reddit username format."""
    if not username or not isinstance(username, str):
        return False
    # Reddit usernames: 3-20 chars, letters, numbers, underscores, hyphens
    return bool(re.match(r'^[A-Za-z0-9_-]{3,20}$', username))

def validate_time_filter(time_filter: str) -> bool:
    """Validate Reddit time filter values."""
    valid_filters = {'hour', 'day', 'week', 'month', 'year', 'all'}
    return time_filter in valid_filters

def validate_positive_int(value: int, max_value: int = 1000) -> bool:
    """Validate positive integer within reasonable bounds."""
    return isinstance(value, int) and 0 < value <= max_value

def validate_keyword(keyword: str) -> bool:
    """Validate a single keyword for search operations.

    Keywords must be:
    - 2-50 characters long
    - Contain only alphanumeric characters, spaces, hyphens, periods, or underscores

    Args:
        keyword: The keyword string to validate

    Returns:
        True if valid, False otherwise
    """
    if not keyword or not isinstance(keyword, str):
        return False
    return bool(re.match(r'^[\w\s\-\.]{2,50}$', keyword))

def validate_keyword_list(keywords: List[str]) -> bool:
    """Validate a list of keywords for batch search operations.

    The list must be:
    - A non-empty list (not tuple or other sequence)
    - All elements must be valid keywords per validate_keyword()

    Args:
        keywords: List of keyword strings to validate

    Returns:
        True if valid list with all valid keywords, False otherwise
    """
    if not isinstance(keywords, list) or not keywords:
        return False
    return all(validate_keyword(kw) for kw in keywords)

def validate_batch_delay(delay: float) -> bool:
    """Validate batch delay value for rate limiting.

    Delay must be:
    - A numeric value (int or float)
    - Between 0.5 and 10.0 seconds (inclusive)

    Args:
        delay: The delay value in seconds

    Returns:
        True if valid, False otherwise
    """
    return isinstance(delay, (int, float)) and 0.5 <= delay <= 10.0

def input_validator(*validation_rules):
    """Decorator for input validation with custom rules."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Apply validation rules
                for rule in validation_rules:
                    rule_name, param_name, validator = rule
                    
                    # Get parameter value
                    if param_name in kwargs:
                        param_value = kwargs[param_name]
                    else:
                        # Handle positional args by parameter name
                        import inspect
                        sig = inspect.signature(func)
                        param_names = list(sig.parameters.keys())
                        if param_name in param_names:
                            param_index = param_names.index(param_name)
                            if param_index < len(args):
                                param_value = args[param_index]
                            else:
                                continue  # Parameter not provided, use default
                        else:
                            continue
                    
                    # Validate based on type
                    if validator == 'subreddit_list':
                        if not isinstance(param_value, list) or not param_value:
                            raise ValidationError(f"Invalid {param_name}: must be non-empty list")
                        for name in param_value:
                            if not validate_subreddit_name(name):
                                raise ValidationError(f"Invalid subreddit name: {name}")
                                
                    elif validator == 'post_id_list':
                        if not isinstance(param_value, list) or not param_value:
                            raise ValidationError(f"Invalid {param_name}: must be non-empty list")
                        for post_id in param_value:
                            if not validate_post_id(post_id):
                                raise ValidationError(f"Invalid post ID: {post_id}")
                                
                    elif validator == 'time_filter':
                        if not validate_time_filter(param_value):
                            raise ValidationError(f"Invalid time_filter: {param_value}")
                            
                    elif validator == 'positive_int':
                        if not validate_positive_int(param_value):
                            raise ValidationError(f"Invalid {param_name}: must be positive integer <= 1000")
                
                # Call original function if validation passes
                return await func(*args, **kwargs)
                
            except ValidationError as e:
                logger.warning(f"Validation failed for {func.__name__}: {e}")
                error_response = {
                    "error": "Invalid input parameters",
                    "message": str(e),
                    "tool": func.__name__
                }
                return safe_json_response(error_response)
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                error_response = {
                    "error": "Tool execution failed",
                    "message": str(e),
                    "tool": func.__name__
                }
                return safe_json_response(error_response)
        
        return wrapper
    return decorator

# Initialize FastMCP server
mcp = FastMCP("reddit_opportunity_finder_enhanced")

# Helper Functions
def calculate_sentiment(text: str, positive_keywords: Dict[str, float], negative_keywords: Dict[str, float]) -> float:
    """Calculate sentiment score based on weighted keyword presence."""
    text_lower = text.lower()
    score, count = 0.0, 0
    for kw, weight in {**positive_keywords, **negative_keywords}.items():
        if kw in text_lower:
            score += weight
            count += 1
    return score / max(count, 1) if count else 0.0

def extract_patterns(text: str, patterns: List[str]) -> List[str]:
    """Extract matches for given regex patterns."""
    matches = []
    for pattern in patterns:
        try:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        except re.error:
            logger.error(f"Invalid regex pattern: {pattern}")
    return matches

def normalize_engagement(post, time_decay: bool = True) -> float:
    """Calculate normalized engagement score."""
    engagement = (post.score * 0.4) + (post.num_comments * 0.4) + (post.upvote_ratio * 100 * 0.2)
    if time_decay and hasattr(post, 'created_utc'):
        post_time = datetime.fromtimestamp(post.created_utc)
        current_time = datetime.now()
        post_age_days = (current_time - post_time).days
        decay_factor = 1.0 / (1 + post_age_days * 0.1)
        engagement *= decay_factor
    return engagement

# Tools
@mcp.tool()
@input_validator(
    ("subreddit_names", "subreddit_names", "subreddit_list"),
    ("time_filter", "time_filter", "time_filter"),
    ("limit", "limit", "positive_int"),
    ("min_score", "min_score", "positive_int")
)
async def subreddit_pain_point_scanner(
    subreddit_names: List[str],
    time_filter: str = "week",
    limit: int = 100,
    pain_keywords: List[str] = None,
    min_score: int = 5,
    include_comments: bool = True,
    comment_depth: int = 3
) -> List[TextContent]:
    """Scans subreddits for posts/comments expressing problems or frustrations.

    Args:
        subreddit_names: List of subreddit names to scan
        time_filter: Time filter for posts ('hour', 'day', 'week', 'month', 'year', 'all')
        limit: Maximum number of posts to analyze per subreddit
        pain_keywords: Keywords indicating pain points
        min_score: Minimum score for posts/comments to consider
        include_comments: Whether to include comments in the scan
        comment_depth: Depth of comment threads to search
    """
    if pain_keywords is None:
        pain_keywords = ["frustrated", "annoying", "wish there was", "need help with", 
                        "struggling with", "pain point", "difficult", "tedious"]
    
    pain_points = []
    category_counts = defaultdict(int)
    
    for subreddit_name in subreddit_names:
        try:
            # Get subreddit using RedditClient
            reddit = await RedditClient.get()
            subreddit = await RedditClient.execute(
                lambda: reddit.subreddit(subreddit_name)
            )
            
            # Get posts using RedditClient
            posts = await RedditClient.execute(
                lambda: list(subreddit.top(time_filter=time_filter, limit=limit))
            )
            
            for post in posts:
                if post.score < min_score:
                    continue
                    
                post_text = f"{post.title} {post.selftext}".lower()
                found_keywords = [kw for kw in pain_keywords if kw in post_text]
                
                if found_keywords:
                    # Extract problem description
                    problem = '. '.join([
                        s for s in post.selftext.split('.') 
                        if any(kw in s.lower() for kw in found_keywords)
                    ][:3])
                    
                    # Use safe serialization
                    pain_point = SafeJSONEncoder.serialize_post(post)
                    pain_point.update({
                        "type": "post",
                        "keywords": found_keywords,
                        "problem": problem
                    })
                    pain_points.append(pain_point)
                    
                    for kw in found_keywords:
                        category_counts[kw] += 1
                
                # Process comments if requested
                if include_comments:
                    try:
                        await RedditClient.execute(
                            lambda: post.comments.replace_more(limit=0)
                        )
                        comments = await RedditClient.execute(
                            lambda: post.comments.list()[:comment_depth]
                        )
                        
                        for comment in comments:
                            if (hasattr(comment, 'score') and comment.score >= min_score and 
                                hasattr(comment, 'body') and 
                                any(kw in comment.body.lower() for kw in pain_keywords)):
                                
                                comment_data = SafeJSONEncoder.serialize_comment(comment)
                                comment_data["type"] = "comment"
                                pain_points.append(comment_data)
                                
                    except (prawcore.exceptions.Forbidden, 
                            prawcore.exceptions.NotFound) as e:
                        logger.warning(f"Cannot access comments for post {post.id}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing comments for post {post.id}: {e}")
                        
        except (prawcore.exceptions.Forbidden, 
                prawcore.exceptions.NotFound) as e:
            logger.warning(f"Cannot access subreddit {subreddit_name}: {e}")
        except (prawcore.exceptions.ServerError,
                prawcore.exceptions.TooManyRequests) as e:
            logger.error(f"Reddit API error for {subreddit_name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error scanning {subreddit_name}: {e}")
    
    # Sort by score and limit results
    pain_points.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    result = {
        "pain_points": pain_points[:50],
        "category_counts": dict(category_counts),
        "total_found": len(pain_points),
        "subreddits_scanned": len(subreddit_names)
    }
    
    return safe_json_response(result)

@mcp.tool()
async def solution_request_tracker(
    subreddit_names: List[str],
    request_patterns: List[str] = None,
    exclude_solved: bool = True,
    min_engagement: int = 5,
    time_window: str = "month",
    category_keywords: Dict[str, List[str]] = None
) -> List[TextContent]:
    """Identifies posts asking for software/tool recommendations.

    Args:
        subreddit_names: Target subreddits
        request_patterns: Patterns for requests (e.g., "looking for")
        exclude_solved: Filter out solved posts
        min_engagement: Minimum comments for relevance
        time_window: Time period ('hour', 'day', 'week', 'month', 'year', 'all')
        category_keywords: Keywords to categorize requests
    """
    if request_patterns is None:
        request_patterns = ["looking for.*app", "any.*tools", "recommend.*software"]
    if category_keywords is None:
        category_keywords = {"automation": ["automate", "workflow"], "productivity": ["efficiency", "time"]}
    requests = []
    
    for subreddit_name in subreddit_names:
        try:
            reddit = await RedditClient.get()
            subreddit = await RedditClient.execute(
                lambda: reddit.subreddit(subreddit_name)
            )
            posts = await RedditClient.execute(
                lambda: list(subreddit.top(time_filter=time_window, limit=100))
            )
            
            for post in posts:
                if not hasattr(post, 'num_comments') or post.num_comments < min_engagement:
                    continue
                post_text = f"{post.title} {post.selftext}" if hasattr(post, 'selftext') else post.title
                if any(re.search(p, post_text, re.IGNORECASE) for p in request_patterns):
                    if exclude_solved and ("[solved]" in post_text.lower() or "solved" in post_text.lower()):
                        continue
                    category = next((cat for cat, kws in category_keywords.items() if any(kw in post_text.lower() for kw in kws)), "general")
                    requirements = extract_patterns(post_text, [r"need.*?(?=\.)", r"must.*?(?=\.)"])
                    
                    request_data = SafeJSONEncoder.serialize_post(post)
                    request_data.update({
                        "category": category, 
                        "requirements": requirements[:3]
                    })
                    requests.append(request_data)
        except Exception as e:
            logger.error(f"Error tracking {subreddit_name}: {e}")
    result = {"requests": requests[:50], "total": len(requests)}
    return safe_json_response(result)

@mcp.tool()
async def user_workflow_analyzer(
    subreddit_names: List[str],
    workflow_indicators: List[str] = None,
    min_steps: int = 3,
    efficiency_keywords: List[str] = None
) -> List[TextContent]:
    """Analyzes user workflows for automation opportunities.

    Args:
        subreddit_names: Subreddits to target
        workflow_indicators: Keywords like “process”, “steps”
        min_steps: Minimum steps for complexity
        efficiency_keywords: Words indicating inefficiency
    """
    if workflow_indicators is None:
        workflow_indicators = ["process", "steps", "routine", "manually"]
    if efficiency_keywords is None:
        efficiency_keywords = ["time-consuming", "repetitive"]
    workflows = []
    
    for subreddit_name in subreddit_names:
        try:
            reddit = await RedditClient.get()
            subreddit = await RedditClient.execute(
                lambda: reddit.subreddit(subreddit_name)
            )
            posts = await RedditClient.execute(
                lambda: list(subreddit.search(" ".join(workflow_indicators), limit=100))
            )
            
            for post in posts:
                if hasattr(post, 'selftext') and post.selftext:
                    steps = extract_patterns(post.selftext, [r"(?:step\s*)?(\d+)[\.\)]\s*([^\n]+)", r"(?:then|next)\s*([^\n]+)"])
                    if len(steps) >= min_steps:
                        issues = [kw for kw in efficiency_keywords if kw in post.selftext.lower()]
                        
                        workflow_data = SafeJSONEncoder.serialize_post(post)
                        workflow_data.update({
                            "steps": steps[:5], 
                            "issues": issues,
                            "complexity_score": len(steps)
                        })
                        workflows.append(workflow_data)
        except Exception as e:
            logger.error(f"Error analyzing {subreddit_name}: {e}")
    result = {"workflows": workflows[:50], "total": len(workflows)}
    return safe_json_response(result)

@mcp.tool()
async def competitor_mention_monitor(
    competitor_names: List[str],
    sentiment_threshold: float = 0.0,
    limitation_keywords: List[str] = None
) -> List[TextContent]:
    """Tracks competitor mentions and limitations.

    Args:
        competitor_names: Software names to monitor
        sentiment_threshold: Sentiment score threshold (-1 to 1)
        limitation_keywords: Keywords indicating limitations
    """
    if limitation_keywords is None:
        limitation_keywords = ["but", "however", "missing"]
    pos_keywords = {"great": 0.8, "love": 1.0}
    neg_keywords = {"hate": -1.0, "terrible": -1.0}
    mentions = []
    
    for name in competitor_names:
        try:
            reddit = await RedditClient.get()
            posts = await RedditClient.execute(
                lambda: list(reddit.subreddit("all").search(name, limit=100))
            )
            
            for post in posts:
                if hasattr(post, 'title') and hasattr(post, 'selftext'):
                    text = f"{post.title} {post.selftext}".lower()
                    sentiment = calculate_sentiment(text, pos_keywords, neg_keywords)
                    if sentiment < sentiment_threshold:
                        limits = [kw for kw in limitation_keywords if kw in text]
                        if limits:
                            mention_data = SafeJSONEncoder.serialize_post(post)
                            mention_data.update({
                                "competitor": name,
                                "sentiment": round(sentiment, 3),
                                "limitations": limits
                            })
                            mentions.append(mention_data)
        except Exception as e:
            logger.error(f"Error monitoring {name}: {e}")
    
    mentions.sort(key=lambda x: x.get("sentiment", 0))  # Sort by sentiment (most negative first)
    result = {"mentions": mentions[:50], "total": len(mentions)}
    return safe_json_response(result)

@mcp.tool()
@input_validator(
    ("post_ids", "post_ids", "post_id_list")
)
async def subreddit_engagement_analyzer(
    post_ids: List[str],
    engagement_metrics: List[str] = None,
    time_decay: bool = True
) -> List[TextContent]:
    """Analyzes engagement patterns to validate problem severity.

    Args:
        post_ids: Post IDs to analyze
        engagement_metrics: Metrics ('upvote_ratio', 'comment_rate')
        time_decay: Apply time decay to engagement
    """
    if engagement_metrics is None:
        engagement_metrics = ["upvote_ratio", "comment_rate"]
    
    analyzed = []
    failed_posts = []
    
    for post_id in post_ids:
        try:
            reddit = await RedditClient.get()
            post = await RedditClient.execute(
                lambda: reddit.submission(id=post_id)
            )
            
            # Verify post exists and is accessible
            if not hasattr(post, 'title') or not hasattr(post, 'created_utc'):
                logger.warning(f"Post {post_id} missing required attributes")
                failed_posts.append({"post_id": post_id, "error": "Missing required attributes"})
                continue
            
            metrics = {}
            
            # Calculate upvote ratio if requested
            if "upvote_ratio" in engagement_metrics and hasattr(post, 'upvote_ratio'):
                metrics["upvote_ratio"] = post.upvote_ratio
            
            # Calculate comment rate if requested
            if "comment_rate" in engagement_metrics:
                if hasattr(post, 'num_comments') and hasattr(post, 'created_utc'):
                    # Use timezone-aware datetime calculation
                    post_time = datetime.fromtimestamp(post.created_utc)
                    current_time = datetime.now()
                    hours = (current_time - post_time).total_seconds() / 3600
                    metrics["comment_rate"] = post.num_comments / max(hours, 1)
                else:
                    metrics["comment_rate"] = 0
            
            # Use safe serialization and add engagement metrics
            post_data = SafeJSONEncoder.serialize_post(post)
            post_data.update({
                "engagement": normalize_engagement(post, time_decay),
                "metrics": metrics,
                "analysis_timestamp": datetime.now().isoformat()
            })
            
            analyzed.append(post_data)
            
        except (prawcore.exceptions.NotFound, 
                prawcore.exceptions.Forbidden) as e:
            logger.warning(f"Cannot access post {post_id}: {e}")
            failed_posts.append({"post_id": post_id, "error": str(e)})
        except Exception as e:
            logger.error(f"Error analyzing post {post_id}: {e}")
            failed_posts.append({"post_id": post_id, "error": str(e)})
    
    # Calculate average engagement
    avg_engagement = 0
    if analyzed:
        total_engagement = sum(p.get("engagement", 0) for p in analyzed)
        avg_engagement = total_engagement / len(analyzed)
    
    result = {
        "analyzed": analyzed,
        "failed_posts": failed_posts,
        "avg_engagement": avg_engagement,
        "total_posts": len(post_ids),
        "successful_analyses": len(analyzed),
        "metrics_calculated": engagement_metrics
    }
    
    return safe_json_response(result)

@mcp.tool()
async def niche_community_discoverer(
    topic_keywords: List[str],
    min_subscribers: int = 5000,
    max_subscribers: int = 200000,
    max_communities: int = 50,
    spider_sidebar: bool = True,
    batch_delay: float = 1.5
) -> List[TextContent]:
    """Discovers niche communities using keyword search and sidebar spidering.

    Finds subreddits related to specified topics by searching Reddit and
    optionally crawling sidebar links to discover related communities.

    Args:
        topic_keywords: Intent-based search terms (e.g., ["python automation", "productivity tools"])
        min_subscribers: Minimum subscriber count to include (default: 5000)
        max_subscribers: Maximum subscriber count to include (default: 200000)
        max_communities: Hard cap on total communities to return (default: 50)
        spider_sidebar: Enable/disable sidebar crawl for related communities (default: True)
        batch_delay: Seconds between API batches for rate limiting (default: 1.5)

    Returns:
        ToolResponse envelope with discovered communities, each containing:
        - name: Subreddit display name
        - subscribers: Subscriber count
        - source: "search" or "sidebar of {parent}"
        - description: Public description
    """
    # Create response envelope
    response = ToolResponse(tool_name="niche_community_discoverer")

    # Input validation
    if not validate_keyword_list(topic_keywords):
        response.add_error("topic_keywords", "Must be a non-empty list of valid keywords (2-50 chars each)")
        response.extra_metadata["keywords_searched"] = []
        return response.to_response()

    if not validate_batch_delay(batch_delay):
        response.add_error("batch_delay", "Must be between 0.5 and 10.0 seconds")
        response.extra_metadata["keywords_searched"] = topic_keywords
        return response.to_response()

    # Validate subscriber range
    if not isinstance(min_subscribers, int) or min_subscribers < 0:
        response.add_error("min_subscribers", "Must be a non-negative integer")
        return response.to_response()
    if not isinstance(max_subscribers, int) or max_subscribers < 0:
        response.add_error("max_subscribers", "Must be a non-negative integer")
        return response.to_response()
    if min_subscribers > max_subscribers:
        response.add_error("subscriber_range", "min_subscribers must be <= max_subscribers")
        return response.to_response()

    # Validate max_communities
    if not isinstance(max_communities, int) or max_communities < 1:
        response.add_error("max_communities", "Must be a positive integer")
        return response.to_response()

    # Set up rate limiting
    config = RateLimitConfig(batch_delay=batch_delay, request_budget=100)
    executor = RateLimitedExecutor(config)

    # Track discovered communities by name for deduplication
    discovered: Dict[str, Dict[str, Any]] = {}

    # Regex pattern to extract subreddit links from sidebar
    subreddit_pattern = re.compile(r'/r/([A-Za-z0-9_]+)', re.IGNORECASE)

    def is_valid_community(subreddit) -> bool:
        """Check if subreddit meets subscriber criteria and is public."""
        try:
            if not hasattr(subreddit, 'subscribers'):
                return False
            if not (min_subscribers <= subreddit.subscribers <= max_subscribers):
                return False
            # Check if public
            sub_type = getattr(subreddit, 'subreddit_type', 'public')
            if sub_type not in ('public', 'restricted'):
                return False
            return True
        except Exception:
            return False

    def extract_community_data(subreddit, source: str) -> Dict[str, Any]:
        """Extract standardized community data from subreddit object."""
        return {
            "name": subreddit.display_name,
            "subscribers": subreddit.subscribers,
            "source": source,
            "description": (subreddit.public_description[:200]
                          if hasattr(subreddit, 'public_description') and subreddit.public_description
                          else "No description available")
        }

    # Phase 1: Primary Search
    logger.info(f"Starting community discovery with keywords: {topic_keywords}")

    for keyword in topic_keywords:
        if len(discovered) >= max_communities:
            logger.info(f"Reached max_communities cap ({max_communities}), stopping search")
            break

        try:
            reddit = await RedditClient.get()

            # Search for subreddits matching keyword
            search_results = await executor.execute(
                lambda kw=keyword: list(reddit.subreddits.search(kw, limit=20))
            )

            for subreddit in search_results:
                if len(discovered) >= max_communities:
                    break

                # Skip if already discovered
                name_lower = subreddit.display_name.lower()
                if name_lower in discovered:
                    continue

                # Validate subscriber range
                if is_valid_community(subreddit):
                    discovered[name_lower] = extract_community_data(subreddit, "search")
                    logger.debug(f"Discovered via search: {subreddit.display_name}")

        except BudgetExhaustedError:
            response.add_error(keyword, "Request budget exhausted")
            response.partial = True
            break
        except Exception as e:
            logger.error(f"Error searching keyword '{keyword}': {e}")
            response.add_error(keyword, str(e))

    # Phase 2: Sidebar Spider (if enabled and under cap)
    if spider_sidebar and len(discovered) < max_communities:
        logger.info("Starting sidebar spider phase")

        # Get list of communities to spider (copy to avoid modification during iteration)
        communities_to_spider = list(discovered.values())

        for community in communities_to_spider:
            if len(discovered) >= max_communities:
                logger.info(f"Reached max_communities cap ({max_communities}), stopping spider")
                break

            try:
                reddit = await RedditClient.get()

                # Fetch the subreddit to get sidebar description
                subreddit = await executor.execute(
                    lambda name=community["name"]: reddit.subreddit(name)
                )

                # Get sidebar text (description field contains sidebar in old reddit)
                sidebar_text = getattr(subreddit, 'description', '') or ''

                # Extract r/SubredditName patterns
                matches = subreddit_pattern.findall(sidebar_text)

                for match in matches:
                    if len(discovered) >= max_communities:
                        break

                    # Skip if already discovered
                    match_lower = match.lower()
                    if match_lower in discovered:
                        continue

                    try:
                        # Validate the linked subreddit
                        linked_sub = await executor.execute(
                            lambda m=match: reddit.subreddit(m)
                        )

                        if is_valid_community(linked_sub):
                            discovered[match_lower] = extract_community_data(
                                linked_sub,
                                f"sidebar of {community['name']}"
                            )
                            logger.debug(f"Discovered via sidebar of {community['name']}: {match}")

                    except BudgetExhaustedError:
                        response.add_error(f"sidebar:{match}", "Request budget exhausted")
                        response.partial = True
                        break
                    except (prawcore.exceptions.NotFound, prawcore.exceptions.Forbidden):
                        # Subreddit doesn't exist or is private
                        logger.debug(f"Skipping inaccessible sidebar link: r/{match}")
                        continue
                    except Exception as e:
                        logger.debug(f"Error validating sidebar link r/{match}: {e}")
                        continue

            except BudgetExhaustedError:
                response.add_error(f"spider:{community['name']}", "Request budget exhausted")
                response.partial = True
                break
            except Exception as e:
                logger.debug(f"Error spidering sidebar of {community['name']}: {e}")
                continue

    # Build response
    response.results = list(discovered.values())
    response.stats = {
        **executor.get_stats(),
        "communities_from_search": len([c for c in discovered.values() if c["source"] == "search"]),
        "communities_from_sidebar": len([c for c in discovered.values() if c["source"].startswith("sidebar of")])
    }

    # Add keywords_searched to metadata at top level (per API spec)
    response.extra_metadata["keywords_searched"] = topic_keywords

    logger.info(f"Community discovery complete: {len(discovered)} communities found")

    return response.to_response()

@mcp.tool()
@input_validator(
    ("subreddit_names", "subreddit_names", "subreddit_list")
)
async def temporal_trend_analyzer(
    subreddit_names: List[str],
    time_periods: List[str] = None,
    trend_keywords: List[str] = None,
    growth_threshold: float = 0.5
) -> List[TextContent]:
    """Tracks problem evolution over time.

    Args:
        subreddit_names: Subreddits to monitor
        time_periods: Time intervals ('day', 'week', 'month')
        trend_keywords: Keywords to track
        growth_threshold: Minimum growth rate
    """
    if time_periods is None:
        time_periods = ["day", "week", "month"]
    if trend_keywords is None:
        trend_keywords = ["automation", "productivity", "software", "tool", "app"]
    
    trends = defaultdict(lambda: defaultdict(int))
    failed_subreddits = []
    
    for subreddit_name in subreddit_names:
        try:
            reddit = await RedditClient.get()
            subreddit = await RedditClient.execute(
                lambda: reddit.subreddit(subreddit_name)
            )
            
            for period in time_periods:
                try:
                    posts = await RedditClient.execute(
                        lambda: list(subreddit.top(time_filter=period, limit=50))
                    )
                    
                    for post in posts:
                        if hasattr(post, 'title') and hasattr(post, 'selftext'):
                            text = f"{post.title} {post.selftext}".lower()
                            for keyword in trend_keywords:
                                if keyword.lower() in text:
                                    trends[keyword][period] += 1
                                    
                except (prawcore.exceptions.Forbidden, 
                        prawcore.exceptions.NotFound) as e:
                    logger.warning(f"Cannot access {subreddit_name} for period {period}: {e}")
                    
        except Exception as e:
            logger.error(f"Error analyzing {subreddit_name}: {e}")
            failed_subreddits.append({"subreddit": subreddit_name, "error": str(e)})
    
    # Calculate emerging trends
    emerging_trends = []
    for keyword, period_data in trends.items():
        if "week" in period_data and "month" in period_data and period_data["month"] > 0:
            weekly_avg = period_data["month"] / 4
            if weekly_avg > 0:
                growth_rate = (period_data["week"] - weekly_avg) / weekly_avg
                if growth_rate > growth_threshold:
                    emerging_trends.append({
                        "keyword": keyword,
                        "growth_rate": round(growth_rate, 3),
                        "weekly_mentions": period_data["week"],
                        "monthly_mentions": period_data["month"]
                    })
    
    emerging_trends.sort(key=lambda x: x["growth_rate"], reverse=True)
    
    result = {
        "emerging_trends": emerging_trends[:10],
        "trend_data": dict(trends),
        "failed_subreddits": failed_subreddits,
        "keywords_tracked": trend_keywords,
        "time_periods": time_periods
    }
    
    return safe_json_response(result)

@mcp.tool()
@input_validator(
    ("user_sample_size", "user_sample_size", "positive_int"),
    ("activity_depth", "activity_depth", "positive_int")
)
async def user_persona_extractor(
    user_sample_size: int = 100,
    activity_depth: int = 50,
    need_categories: List[str] = None
) -> List[TextContent]:
    """Builds user personas from Reddit activity.

    Args:
        user_sample_size: Number of users to analyze
        activity_depth: Posts/comments per user
        need_categories: Categories like 'productivity'
    """
    if need_categories is None:
        need_categories = ["productivity", "automation"]
    
    personas = []
    users_processed = set()
    seed_subreddits = ["technology", "software"]
    
    for sub_name in seed_subreddits:
        try:
            reddit = await RedditClient.get()
            subreddit = await RedditClient.execute(
                lambda: reddit.subreddit(sub_name)
            )
            
            # Get hot posts
            posts = await RedditClient.execute(
                lambda: list(subreddit.hot(limit=50))
            )
            
            for post in posts:
                if len(users_processed) >= user_sample_size:
                    break
                    
                # Check if post has author and we haven't processed them
                if (post.author and 
                    hasattr(post.author, 'name') and 
                    post.author.name not in users_processed):
                    
                    try:
                        users_processed.add(post.author.name)
                        
                        # Safely get user submissions
                        user_submissions = await RedditClient.execute(
                            lambda: list(post.author.submissions.new(limit=activity_depth))
                        )
                        
                        # Analyze user needs
                        needs = defaultdict(int)
                        for submission in user_submissions:
                            if hasattr(submission, 'title'):
                                text = submission.title.lower()
                                for category in need_categories:
                                    if category in text:
                                        needs[category] += 1
                        
                        # Try to get user subreddits (this may fail for privacy reasons)
                        user_subreddits = []
                        try:
                            subreddits = await RedditClient.execute(
                                lambda: list(post.author.subreddits(limit=5))
                            )
                            user_subreddits = [s.display_name for s in subreddits if hasattr(s, 'display_name')]
                        except (prawcore.exceptions.Forbidden, 
                                prawcore.exceptions.NotFound) as e:
                            logger.debug(f"Cannot access subreddits for user {post.author.name}: {e}")
                        
                        persona = {
                            "user": post.author.name,
                            "subreddits": user_subreddits,
                            "needs": dict(needs),
                            "submission_count": len(user_submissions)
                        }
                        personas.append(persona)
                        
                    except (prawcore.exceptions.Forbidden, 
                            prawcore.exceptions.NotFound) as e:
                        logger.warning(f"Cannot access user data for {post.author.name}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing user {post.author.name}: {e}")
                        
        except (prawcore.exceptions.Forbidden, 
                prawcore.exceptions.NotFound) as e:
            logger.warning(f"Cannot access subreddit {sub_name}: {e}")
        except Exception as e:
            logger.error(f"Error extracting from {sub_name}: {e}")
    
    result = {
        "personas": personas,
        "total": len(personas),
        "categories_analyzed": need_categories,
        "seed_subreddits": seed_subreddits
    }
    
    return safe_json_response(result)

@mcp.tool()
async def idea_validation_scorer(
    pain_point_data: Dict[str, Any],
    market_size_weight: float = 0.3,
    severity_weight: float = 0.3,
    competition_weight: float = 0.2,
    technical_feasibility: Dict[str, float] = None
) -> List[TextContent]:
    """Scores software opportunities.

    Args:
        pain_point_data: Pain point info (e.g., {'pain': 'slow', 'count': 10})
        market_size_weight: Weight for market size
        severity_weight: Weight for severity
        competition_weight: Weight for competition
        technical_feasibility: Feasibility scores
    """
    if technical_feasibility is None:
        technical_feasibility = {"general": 0.7}
    scores = []
    
    for pain, data in pain_point_data.items():
        market = min(1.0, data.get("count", 0) / 100)
        severity = min(1.0, data.get("score", 0) / 100)
        comp = 1 - min(1.0, data.get("competition", 0) / 10)
        feas = technical_feasibility.get(pain, 0.5)
        total = (market * market_size_weight + severity * severity_weight + comp * competition_weight + feas * 0.2)
        scores.append({"pain": pain, "total": total, "market": market, "severity": severity, "comp": comp, "feas": feas})
    scores.sort(key=lambda x: x["total"], reverse=True)
    result = {"scores": scores[:10], "total": len(scores)}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# Default workflow signals for workflow_thread_inspector
DEFAULT_WORKFLOW_SIGNALS = [
    "step", "process", "export", "import", "csv", "manual", "copy", "paste",
    "click", "download", "upload", "workaround", "hack", "tedious"
]


@mcp.tool()
async def workflow_thread_inspector(
    post_ids: List[str],
    workflow_signals: List[str] = None,
    comment_limit: int = 100,
    expand_depth: int = 5,
    min_score: int = 1,
    batch_delay: float = 1.5
) -> List[TextContent]:
    """Expands nested comment trees to find workflow details buried in replies.

    Analyzes Reddit post comment threads to discover detailed workflow information,
    step-by-step processes, and manual work patterns that users describe in comments.

    Args:
        post_ids: Reddit post IDs to analyze (e.g., ["abc123", "def456"])
        workflow_signals: LLM-generated contextual keywords to match. If None,
            uses default signals like "step", "process", "export", "manual", etc.
        comment_limit: Maximum comments to return per post (default: 100)
        expand_depth: How many "load more" expansions to perform (default: 5)
        min_score: Minimum comment score to include (default: 1)
        batch_delay: Seconds between API requests for rate limiting (default: 1.5)

    Returns:
        ToolResponse envelope with workflow comments per post, each containing:
        - post_id: The Reddit post ID
        - post_title: Title of the post
        - post_url: URL to the post
        - workflow_comments: List of comments matching workflow signals, each with:
            - author: Username (or None if deleted)
            - body: Comment text (truncated to 1000 chars)
            - score: Comment score
            - depth: Nesting depth in thread
            - url: Permalink to comment
    """
    # Create response envelope
    response = ToolResponse(tool_name="workflow_thread_inspector")

    # Apply default workflow signals if None provided
    signals_used = workflow_signals if workflow_signals is not None else DEFAULT_WORKFLOW_SIGNALS
    response.extra_metadata["signals_used"] = signals_used

    # Input validation: post_ids
    if not isinstance(post_ids, list) or not post_ids:
        response.add_error("post_ids", "Must be a non-empty list of post IDs")
        return response.to_response()

    for post_id in post_ids:
        if not validate_post_id(post_id):
            response.add_error("post_ids", f"Invalid post ID format: {post_id}")
            return response.to_response()

    # Input validation: batch_delay
    if not validate_batch_delay(batch_delay):
        response.add_error("batch_delay", "Must be between 0.5 and 10.0 seconds")
        return response.to_response()

    # Set up rate limiting
    config = RateLimitConfig(batch_delay=batch_delay, request_budget=100)
    executor = RateLimitedExecutor(config)

    def truncate_body(body: str, max_length: int = 1000) -> str:
        """Truncate body to max_length chars with '...' suffix."""
        if len(body) <= max_length:
            return body
        return body[:max_length] + "..."

    def comment_matches_signals(comment_body: str, signals: List[str]) -> bool:
        """Check if comment body contains any of the signal keywords."""
        body_lower = comment_body.lower()
        return any(signal.lower() in body_lower for signal in signals)

    def extract_comment_data(comment, signals: List[str]) -> Optional[Dict[str, Any]]:
        """Extract standardized comment data if it passes filters."""
        # Check score
        if not hasattr(comment, 'score') or comment.score < min_score:
            return None

        # Check body exists
        if not hasattr(comment, 'body') or not comment.body:
            return None

        # Check for signal keywords
        if not comment_matches_signals(comment.body, signals):
            return None

        # Extract author name (handle deleted users)
        author_name = None
        if hasattr(comment, 'author') and comment.author is not None:
            author_name = getattr(comment.author, 'name', None)

        return {
            "author": author_name,
            "body": truncate_body(comment.body),
            "score": comment.score,
            "depth": getattr(comment, 'depth', 0),
            "url": f"https://reddit.com{comment.permalink}" if hasattr(comment, 'permalink') else None
        }

    # Process each post
    logger.info(f"Starting workflow thread inspection for {len(post_ids)} posts")

    for post_id in post_ids:
        try:
            reddit = await RedditClient.get()

            # Fetch submission
            submission = await executor.execute(
                lambda pid=post_id: reddit.submission(id=pid)
            )

            # Expand comment tree
            await executor.execute(
                lambda sub=submission: sub.comments.replace_more(limit=expand_depth)
            )

            # Get flattened comment list
            comments = await executor.execute(
                lambda sub=submission: list(sub.comments.list())
            )

            # Filter and extract workflow comments
            workflow_comments = []
            for comment in comments:
                comment_data = extract_comment_data(comment, signals_used)
                if comment_data is not None:
                    workflow_comments.append(comment_data)

                    # Respect comment_limit
                    if len(workflow_comments) >= comment_limit:
                        break

            # Build result for this post
            post_result = {
                "post_id": submission.id,
                "post_title": submission.title if hasattr(submission, 'title') else "",
                "post_url": submission.url if hasattr(submission, 'url') else f"https://reddit.com/comments/{submission.id}",
                "workflow_comments": workflow_comments
            }
            response.add_result(post_result)

            logger.debug(f"Processed post {post_id}: found {len(workflow_comments)} workflow comments")

        except BudgetExhaustedError:
            response.add_error(post_id, "Request budget exhausted")
            break
        except (prawcore.exceptions.NotFound, prawcore.exceptions.Forbidden) as e:
            logger.warning(f"Cannot access post {post_id}: {e}")
            response.add_error(post_id, f"Post inaccessible: {type(e).__name__}")
        except Exception as e:
            logger.error(f"Error processing post {post_id}: {e}")
            response.add_error(post_id, str(e))

    # Add executor stats to response
    response.stats = executor.get_stats()

    logger.info(f"Workflow thread inspection complete: {len(response.results)} posts processed, {len(response.errors)} errors")

    return response.to_response()


# Default page keywords for wiki_tool_extractor
DEFAULT_PAGE_KEYWORDS = [
    "tools", "software", "resources", "guide", "faq", "index", "wiki", "recommended"
]


def normalize_url(url: str) -> str:
    """Normalize URL for deduplication by removing trailing slashes and query params.

    Args:
        url: The URL to normalize

    Returns:
        Normalized URL string
    """
    # Remove trailing slash
    url = url.rstrip('/')
    # Remove query parameters
    if '?' in url:
        url = url.split('?')[0]
    # Remove fragment
    if '#' in url:
        url = url.split('#')[0]
    return url.lower()


def extract_tool_name_from_domain(url: str) -> str:
    """Extract a tool name from a URL's domain.

    Args:
        url: The URL to extract domain from

    Returns:
        Capitalized tool name derived from domain
    """
    try:
        # Extract domain from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]

        # Remove common prefixes and suffixes
        domain = domain.replace('www.', '')

        # Get the main part of the domain (before the TLD)
        parts = domain.split('.')
        if len(parts) >= 2:
            # Use the main domain name, not the TLD
            name = parts[0]
        else:
            name = domain

        # Capitalize first letter
        return name.capitalize()
    except Exception:
        return "Unknown"


@mcp.tool()
async def wiki_tool_extractor(
    subreddit_names: List[str],
    scan_sidebar: bool = True,
    scan_wiki: bool = True,
    page_keywords: List[str] = None,
    batch_delay: float = 1.5
) -> List[TextContent]:
    """Scans subreddit wikis and sidebars for mentioned tools and software.

    Extracts links from subreddit sidebars and wiki pages to discover tools,
    software, and resources mentioned by the community.

    Args:
        subreddit_names: List of subreddit names to scan (e.g., ["productivity", "selfhosted"])
        scan_sidebar: Whether to scan sidebar/description for links (default: True)
        scan_wiki: Whether to scan wiki pages for links (default: True)
        page_keywords: Wiki page name filters. If None, uses defaults like
            "tools", "software", "resources", "guide", "faq", etc.
        batch_delay: Seconds between API requests for rate limiting (default: 1.5)

    Returns:
        ToolResponse envelope with extracted tools per subreddit, each containing:
        - subreddit: The subreddit name
        - tools: List of tools found, each with:
            - name: Tool name (from link text or inferred from domain)
            - url: The tool's URL
            - sources: List of sources where the tool was found (e.g., ["sidebar", "wiki/tools"])
    """
    # Create response envelope
    response = ToolResponse(tool_name="wiki_tool_extractor")
    pages_scanned = 0

    # Apply default page keywords if None provided
    keywords_used = page_keywords if page_keywords is not None else DEFAULT_PAGE_KEYWORDS

    # Input validation: subreddit_names
    if not isinstance(subreddit_names, list) or not subreddit_names:
        response.add_error("subreddit_names", "Must be a non-empty list of subreddit names")
        response.extra_metadata["pages_scanned"] = pages_scanned
        return response.to_response()

    for name in subreddit_names:
        if not validate_subreddit_name(name):
            response.add_error("subreddit_names", f"Invalid subreddit name format: {name}")
            response.extra_metadata["pages_scanned"] = pages_scanned
            return response.to_response()

    # Input validation: batch_delay
    if not validate_batch_delay(batch_delay):
        response.add_error("batch_delay", "Must be between 0.5 and 10.0 seconds")
        response.extra_metadata["pages_scanned"] = pages_scanned
        return response.to_response()

    # Set up rate limiting
    config = RateLimitConfig(batch_delay=batch_delay, request_budget=100)
    executor = RateLimitedExecutor(config)

    # Regex patterns for link extraction
    markdown_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    bare_url_pattern = re.compile(r'https?://[^\s\)\]<>]+')

    def extract_links_from_text(text: str, source: str) -> List[Dict[str, Any]]:
        """Extract links from text and return tool info with source."""
        tools = []
        seen_urls = set()

        # Extract markdown links [text](url)
        for match in markdown_link_pattern.finditer(text):
            link_text = match.group(1).strip()
            url = match.group(2).strip()

            # Skip non-http links and Reddit internal links
            if not url.startswith(('http://', 'https://')):
                continue
            if 'reddit.com' in url.lower():
                continue

            normalized = normalize_url(url)
            if normalized not in seen_urls:
                seen_urls.add(normalized)
                tools.append({
                    "name": link_text,
                    "url": url.rstrip('/'),
                    "source": source
                })

        # Extract bare URLs
        for match in bare_url_pattern.finditer(text):
            url = match.group(0).strip()

            # Skip Reddit internal links
            if 'reddit.com' in url.lower():
                continue

            # Check if this URL was already captured as a markdown link
            normalized = normalize_url(url)
            if normalized not in seen_urls:
                seen_urls.add(normalized)
                tools.append({
                    "name": extract_tool_name_from_domain(url),
                    "url": url.rstrip('/'),
                    "source": source
                })

        return tools

    def page_matches_keywords(page_name: str, keywords: List[str]) -> bool:
        """Check if wiki page name matches any keyword."""
        page_lower = page_name.lower()
        return any(kw.lower() in page_lower for kw in keywords)

    def merge_tools(tools_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge tools by normalized URL, combining sources."""
        merged = {}

        for tool in tools_list:
            normalized = normalize_url(tool["url"])

            if normalized in merged:
                # Add source to existing tool
                if tool["source"] not in merged[normalized]["sources"]:
                    merged[normalized]["sources"].append(tool["source"])
                # Keep the first name (from link text if available)
            else:
                merged[normalized] = {
                    "name": tool["name"],
                    "url": tool["url"],
                    "sources": [tool["source"]]
                }

        return list(merged.values())

    logger.info(f"Starting wiki tool extraction for {len(subreddit_names)} subreddits")

    for subreddit_name in subreddit_names:
        subreddit_tools = []

        try:
            reddit = await RedditClient.get()

            # Fetch subreddit
            subreddit = await executor.execute(
                lambda name=subreddit_name: reddit.subreddit(name)
            )

            # Scan sidebar if enabled
            if scan_sidebar:
                sidebar_text = ""

                # Get description (sidebar markdown)
                if hasattr(subreddit, 'description') and subreddit.description:
                    sidebar_text += subreddit.description + "\n"

                # Get public description
                if hasattr(subreddit, 'public_description') and subreddit.public_description:
                    sidebar_text += subreddit.public_description + "\n"

                if sidebar_text:
                    sidebar_tools = extract_links_from_text(sidebar_text, "sidebar")
                    subreddit_tools.extend(sidebar_tools)

            # Scan wiki if enabled
            if scan_wiki:
                try:
                    # List wiki pages
                    wiki_pages = await executor.execute(
                        lambda sub=subreddit: list(sub.wiki)
                    )

                    for page in wiki_pages:
                        page_name = getattr(page, 'name', str(page))

                        # Filter by keywords
                        if not page_matches_keywords(page_name, keywords_used):
                            continue

                        try:
                            # Fetch page content
                            page_content = await executor.execute(
                                lambda p=page: p.content_md if hasattr(p, 'content_md') else ""
                            )

                            pages_scanned += 1

                            if page_content:
                                wiki_tools = extract_links_from_text(
                                    page_content,
                                    f"wiki/{page_name}"
                                )
                                subreddit_tools.extend(wiki_tools)

                        except (prawcore.exceptions.NotFound,
                                prawcore.exceptions.Forbidden) as e:
                            logger.debug(f"Cannot access wiki page {page_name} in {subreddit_name}: {e}")
                        except Exception as e:
                            logger.debug(f"Error reading wiki page {page_name}: {e}")

                except (prawcore.exceptions.NotFound,
                        prawcore.exceptions.Forbidden) as e:
                    logger.debug(f"Cannot access wiki for {subreddit_name}: {e}")
                except Exception as e:
                    logger.debug(f"Error listing wiki pages for {subreddit_name}: {e}")

            # Merge and deduplicate tools for this subreddit
            merged_tools = merge_tools(subreddit_tools)

            # Build result for this subreddit
            result = {
                "subreddit": subreddit_name,
                "tools": merged_tools
            }
            response.add_result(result)

            logger.debug(f"Processed {subreddit_name}: found {len(merged_tools)} tools")

        except BudgetExhaustedError:
            response.add_error(subreddit_name, "Request budget exhausted")
            break
        except (prawcore.exceptions.NotFound, prawcore.exceptions.Forbidden) as e:
            logger.warning(f"Cannot access subreddit {subreddit_name}: {e}")
            response.add_error(subreddit_name, f"Subreddit inaccessible: {type(e).__name__}")
        except Exception as e:
            logger.error(f"Error processing subreddit {subreddit_name}: {e}")
            response.add_error(subreddit_name, str(e))

    # Add executor stats and pages_scanned to response
    response.stats = executor.get_stats()
    response.extra_metadata["pages_scanned"] = pages_scanned

    logger.info(f"Wiki tool extraction complete: {len(response.results)} subreddits processed, "
                f"{pages_scanned} pages scanned, {len(response.errors)} errors")

    return response.to_response()


if __name__ == "__main__":
    logger.info("Starting Reddit Scanner MCP Server...")
    logger.info(f"Server name: reddit_opportunity_finder_enhanced")
    logger.info(f"Transport: stdio")
    logger.info(f"Reddit credentials: {'Configured' if os.getenv('REDDIT_CLIENT_ID') else 'Not configured'}")
    logger.info("Server is ready and waiting for MCP commands...")
    mcp.run(transport="stdio")