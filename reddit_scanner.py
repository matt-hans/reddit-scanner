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
@input_validator(
    ("seed_subreddits", "seed_subreddits", "subreddit_list"),
    ("min_subscribers", "min_subscribers", "positive_int"),
    ("max_subscribers", "max_subscribers", "positive_int")
)
async def niche_community_discoverer(
    seed_subreddits: List[str],
    min_subscribers: int = 1000,
    max_subscribers: int = 100000,
    activity_threshold: float = 0.3,
    related_depth: int = 1
) -> List[TextContent]:
    """Finds niche subreddits with unmet software needs.

    Args:
        seed_subreddits: Starting subreddits
        min_subscribers: Minimum size
        max_subscribers: Maximum size
        activity_threshold: Posts per day threshold
        related_depth: Depth of related subreddit exploration
    """
    # Use provided seeds or defaults
    if not seed_subreddits:
        seed_subreddits = ["learnprogramming", "webdev", "entrepreneur", "smallbusiness"]
        logger.info(f"Using default seed subreddits: {seed_subreddits}")
    
    discovered_communities = []
    failed_subreddits = []
    
    # Simplified approach: analyze seed subreddits directly
    for subreddit_name in seed_subreddits:
        try:
            community_data = await _analyze_community_safely(
                subreddit_name, min_subscribers, max_subscribers, activity_threshold
            )
            
            if community_data:
                discovered_communities.append(community_data)
                logger.info(f"Discovered community: {subreddit_name}")
            else:
                failed_subreddits.append({
                    "name": subreddit_name, 
                    "reason": "Did not meet criteria"
                })
                
        except Exception as e:
            logger.error(f"Error analyzing {subreddit_name}: {e}")
            failed_subreddits.append({
                "name": subreddit_name,
                "reason": str(e)
            })
    
    # Sort by score (subscriber count * activity rate)
    discovered_communities.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    result = {
        "communities": discovered_communities[:20],  # Limit results
        "failed_subreddits": failed_subreddits,
        "total_discovered": len(discovered_communities),
        "seed_subreddits": seed_subreddits,
        "criteria": {
            "min_subscribers": min_subscribers,
            "max_subscribers": max_subscribers,
            "activity_threshold": activity_threshold
        }
    }
    
    return safe_json_response(result)

async def _analyze_community_safely(
    subreddit_name: str, 
    min_subscribers: int,
    max_subscribers: int, 
    activity_threshold: float
) -> Optional[Dict[str, Any]]:
    """Safely analyze a subreddit community."""
    try:
        reddit = await RedditClient.get()
        subreddit = await RedditClient.execute(
            lambda: reddit.subreddit(subreddit_name)
        )
        
        # Check if we can access basic subreddit info
        if not hasattr(subreddit, 'subscribers'):
            return None
            
        # Check subscriber count
        if not (min_subscribers <= subreddit.subscribers <= max_subscribers):
            logger.debug(f"Skipping {subreddit_name}: {subreddit.subscribers} subscribers outside range")
            return None
        
        # Get recent posts to calculate activity
        posts = await RedditClient.execute(
            lambda: list(subreddit.new(limit=20))
        )
        
        if not posts:
            logger.debug(f"Skipping {subreddit_name}: no recent posts")
            return None
        
        # Calculate activity rate (posts per day)
        if len(posts) > 1:
            newest_post = posts[0]
            oldest_post = posts[-1]
            time_span_hours = (newest_post.created_utc - oldest_post.created_utc) / 3600
            time_span_days = max(time_span_hours / 24, 1)  # At least 1 day
            activity_rate = len(posts) / time_span_days
        else:
            activity_rate = 0.1  # Very low activity
        
        if activity_rate < activity_threshold:
            logger.debug(f"Skipping {subreddit_name}: activity rate {activity_rate:.2f} below threshold")
            return None
        
        # Calculate community score
        score = (subreddit.subscribers / 10000) * activity_rate
        
        return {
            "name": subreddit_name,
            "subscribers": subreddit.subscribers,
            "activity_rate": round(activity_rate, 2),
            "score": round(score, 2),
            "description": (subreddit.public_description[:200] 
                          if hasattr(subreddit, 'public_description') and subreddit.public_description 
                          else "No description available"),
            "url": f"https://reddit.com/r/{subreddit_name}"
        }
        
    except (prawcore.exceptions.Forbidden, 
            prawcore.exceptions.NotFound) as e:
        logger.warning(f"Cannot access subreddit {subreddit_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error analyzing {subreddit_name}: {e}")
        return None

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

if __name__ == "__main__":
    logger.info("Starting Reddit Scanner MCP Server...")
    logger.info(f"Server name: reddit_opportunity_finder_enhanced")
    logger.info(f"Transport: stdio")
    logger.info(f"Reddit credentials: {'Configured' if os.getenv('REDDIT_CLIENT_ID') else 'Not configured'}")
    logger.info("Server is ready and waiting for MCP commands...")
    mcp.run(transport="stdio")