import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter, defaultdict
import logging

import praw
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Reddit instance
reddit = None
def get_reddit():
    global reddit
    if reddit is None:
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            raise ValueError(
                "Reddit API credentials not found. Please set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET environment variables in your .env file."
            )
        
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=os.getenv("REDDIT_USER_AGENT", "MCP Reddit Analyzer 1.0"),
        )
    return reddit

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
    if time_decay:
        post_age_days = (datetime.utcnow() - datetime.utcfromtimestamp(post.created_utc)).days
        decay_factor = 1.0 / (1 + post_age_days * 0.1)
        engagement *= decay_factor
    return engagement

# Tools
@mcp.tool()
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
        pain_keywords = ["frustrated", "annoying", "wish there was", "need help with", "struggling with", "pain point", "difficult", "tedious"]
    pain_points, category_counts = [], defaultdict(int)
    
    for subreddit_name in subreddit_names:
        try:
            subreddit = get_reddit().subreddit(subreddit_name)
            posts = subreddit.top(time_filter=time_filter, limit=limit)
            for post in posts:
                if post.score < min_score:
                    continue
                post_text = f"{post.title} {post.selftext}".lower()
                found_keywords = [kw for kw in pain_keywords if kw in post_text]
                if found_keywords:
                    problem = '. '.join([s for s in post.selftext.split('.') if any(kw in s.lower() for kw in found_keywords)][:3])
                    pain_points.append({"type": "post", "title": post.title, "url": post.permalink, "score": post.score, "keywords": found_keywords, "problem": problem})
                    for kw in found_keywords:
                        category_counts[kw] += 1
                if include_comments:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:comment_depth]:
                        if comment.score >= min_score and any(kw in comment.body.lower() for kw in pain_keywords):
                            pain_points.append({"type": "comment", "body": comment.body[:500], "score": comment.score, "url": comment.permalink})
        except Exception as e:
            logger.error(f"Error scanning {subreddit_name}: {e}")
    pain_points.sort(key=lambda x: x["score"], reverse=True)
    result = {"pain_points": pain_points[:50], "category_counts": dict(category_counts), "total_found": len(pain_points)}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

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
            subreddit = get_reddit().subreddit(subreddit_name)
            posts = subreddit.top(time_filter=time_window, limit=100)
            for post in posts:
                if post.num_comments < min_engagement:
                    continue
                post_text = f"{post.title} {post.selftext}"
                if any(re.search(p, post_text, re.IGNORECASE) for p in request_patterns):
                    if exclude_solved and ("[solved]" in post.title.lower() or "solved" in post.selftext.lower()):
                        continue
                    category = next((cat for cat, kws in category_keywords.items() if any(kw in post_text.lower() for kw in kws)), "general")
                    requirements = extract_patterns(post_text, [r"need.*?(?=\.)", r"must.*?(?=\.)"])
                    requests.append({"title": post.title, "url": post.permalink, "category": category, "requirements": requirements[:3]})
        except Exception as e:
            logger.error(f"Error tracking {subreddit_name}: {e}")
    result = {"requests": requests[:50], "total": len(requests)}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

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
            subreddit = get_reddit().subreddit(subreddit_name)
            posts = subreddit.search(" ".join(workflow_indicators), limit=100)
            for post in posts:
                steps = extract_patterns(post.selftext, [r"(?:step\s*)?(\d+)[\.\)]\s*([^\n]+)", r"(?:then|next)\s*([^\n]+)"])
                if len(steps) >= min_steps:
                    issues = [kw for kw in efficiency_keywords if kw in post.selftext.lower()]
                    workflows.append({"title": post.title, "url": post.permalink, "steps": steps[:5], "issues": issues})
        except Exception as e:
            logger.error(f"Error analyzing {subreddit_name}: {e}")
    result = {"workflows": workflows[:50], "total": len(workflows)}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

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
            posts = get_reddit().subreddit("all").search(name, limit=100)
            for post in posts:
                text = f"{post.title} {post.selftext}".lower()
                sentiment = calculate_sentiment(text, pos_keywords, neg_keywords)
                if sentiment < sentiment_threshold:
                    limits = [kw for kw in limitation_keywords if kw in text]
                    if limits:
                        mentions.append({"competitor": name, "title": post.title, "url": post.permalink, "sentiment": sentiment, "limits": limits})
        except Exception as e:
            logger.error(f"Error monitoring {name}: {e}")
    result = {"mentions": mentions[:50], "total": len(mentions)}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

@mcp.tool()
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
    
    for post_id in post_ids:
        try:
            post = get_reddit().submission(id=post_id)
            metrics = {}
            if "upvote_ratio" in engagement_metrics:
                metrics["upvote_ratio"] = post.upvote_ratio
            if "comment_rate" in engagement_metrics:
                hours = (datetime.utcnow() - datetime.utcfromtimestamp(post.created_utc)).total_seconds() / 3600
                metrics["comment_rate"] = post.num_comments / max(hours, 1)
            analyzed.append({"post_id": post_id, "title": post.title, "engagement": normalize_engagement(post, time_decay), "metrics": metrics})
        except Exception as e:
            logger.error(f"Error analyzing {post_id}: {e}")
    result = {"analyzed": analyzed, "avg_engagement": sum(p["engagement"] for p in analyzed) / len(analyzed) if analyzed else 0}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

@mcp.tool()
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
    # Handle empty seed_subreddits
    if not seed_subreddits:
        default_seeds = ["learnprogramming", "webdev", "entrepreneur", "smallbusiness", "sideproject", "startups"]
        logger.warning(f"No seed subreddits provided. Using defaults: {default_seeds}")
        seed_subreddits = default_seeds
    
    discovered, visited = [], set()
    to_visit = seed_subreddits.copy()
    depth = 0
    
    # Helper function to analyze a subreddit
    def analyze_subreddit(name: str) -> Dict[str, Any]:
        try:
            subreddit = get_reddit().subreddit(name)
            
            # Check subscriber count
            if not (min_subscribers <= subreddit.subscribers <= max_subscribers):
                logger.debug(f"Skipping {name}: {subreddit.subscribers} subscribers outside range")
                return None
                
            # Calculate activity rate
            posts = list(subreddit.new(limit=30))
            if not posts:
                logger.debug(f"Skipping {name}: no recent posts")
                return None
                
            # Calculate posts per day rate
            days = max(1, (datetime.utcnow() - datetime.utcfromtimestamp(posts[-1].created_utc)).days)
            rate = len(posts) / days
            
            if rate < activity_threshold:
                logger.debug(f"Skipping {name}: activity rate {rate:.2f} below threshold")
                return None
                
            score = (subreddit.subscribers / 100000) * rate
            return {
                "name": name,
                "subscribers": subreddit.subscribers,
                "rate": rate,
                "score": score,
                "description": subreddit.public_description[:200] if subreddit.public_description else ""
            }
        except Exception as e:
            logger.error(f"Error analyzing {name}: {e}")
            return None
    
    # Helper function to find related subreddits
    def find_related_subreddits(subreddit_name: str, visited: set) -> List[str]:
        related = []
        try:
            subreddit = get_reddit().subreddit(subreddit_name)
            
            # Method 0: Use PRAW's built-in recommended method
            try:
                recommended = get_reddit().subreddits.recommended(
                    subreddits=[subreddit_name],
                    omit_subreddits=list(visited)
                )
                for rec in recommended[:10]:
                    if rec.display_name not in visited and rec.display_name not in related:
                        related.append(rec.display_name)
                        logger.debug(f"Found via PRAW recommended: {rec.display_name}")
            except Exception as e:
                logger.debug(f"PRAW recommended method failed for {subreddit_name}: {e}")
            
            # Method 1: Try widgets (if available)
            try:
                widgets = subreddit.widgets
                for widget in widgets.sidebar:
                    if hasattr(widget, 'data') and isinstance(widget.data, dict) and 'subreddits' in widget.data:
                        for sub in widget.data['subreddits']:
                            if sub.display_name not in visited:
                                related.append(sub.display_name)
                                logger.debug(f"Found related via widget: {sub.display_name}")
            except Exception as e:
                logger.debug(f"Widget method failed for {subreddit_name}: {e}")
            
            # Method 2: Extract subreddit mentions from top posts
            try:
                for post in subreddit.hot(limit=10):
                    # Look for r/subreddit mentions in title and selftext
                    text = f"{post.title} {post.selftext}"
                    mentions = re.findall(r'r/([a-zA-Z0-9_]+)', text)
                    for mention in mentions:
                        if mention not in visited and mention not in related:
                            related.append(mention)
                            logger.debug(f"Found related via mention: {mention}")
            except Exception as e:
                logger.debug(f"Mention extraction failed for {subreddit_name}: {e}")
                
            # Method 3: Search for similar subreddits by keywords
            try:
                if subreddit.public_description:
                    # Extract keywords from description
                    keywords = re.findall(r'\b[a-zA-Z]{4,}\b', subreddit.public_description.lower())
                    keywords = [kw for kw in keywords if kw not in ['that', 'this', 'with', 'from', 'have']][:3]
                    
                    if keywords:
                        search_query = ' OR '.join(keywords)
                        for result in get_reddit().subreddits.search(search_query, limit=5):
                            if result.display_name not in visited and result.display_name not in related:
                                related.append(result.display_name)
                                logger.debug(f"Found related via search: {result.display_name}")
            except Exception as e:
                logger.debug(f"Keyword search failed for {subreddit_name}: {e}")
                
        except Exception as e:
            logger.error(f"Error finding related for {subreddit_name}: {e}")
            
        return related[:5]  # Limit to 5 related per subreddit
    
    # Main discovery loop
    while to_visit and depth < related_depth:
        next_batch = []
        logger.info(f"Processing depth {depth} with {len(to_visit)} subreddits")
        
        for name in to_visit:
            if name in visited:
                continue
                
            visited.add(name)
            
            # For seed subreddits (depth 0), always use them for discovery regardless of size
            if depth == 0:
                # Find related subreddits from seeds without filtering
                related = find_related_subreddits(name, visited)
                next_batch.extend(related)
                logger.info(f"Seed {name}: found {len(related)} related subreddits")
                
                # Optionally analyze the seed itself if it meets criteria
                analysis = analyze_subreddit(name)
                if analysis:
                    discovered.append(analysis)
                    logger.info(f"Seed {name} also meets criteria (subscribers: {analysis['subscribers']}, rate: {analysis['rate']:.2f})")
            else:
                # For discovered subreddits, analyze first then find related if they meet criteria
                analysis = analyze_subreddit(name)
                if analysis:
                    discovered.append(analysis)
                    logger.info(f"Discovered: {name} (subscribers: {analysis['subscribers']}, rate: {analysis['rate']:.2f})")
                    
                    # Find related subreddits only if this one meets criteria and we're not at max depth
                    if depth < related_depth - 1:
                        related = find_related_subreddits(name, visited)
                        next_batch.extend(related)
                
        to_visit = list(set(next_batch))  # Remove duplicates
        depth += 1
    
    # Sort by score
    discovered.sort(key=lambda x: x["score"], reverse=True)
    
    logger.info(f"Discovery complete. Found {len(discovered)} communities")
    result = {"communities": discovered[:30], "total": len(discovered)}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

@mcp.tool()
async def temporal_trend_analyzer(
    subreddit_names: List[str],
    time_periods: List[str] = None,
    trend_keywords: List[str] = None,
    growth_threshold: float = 0.5
) -> List[TextContent]:
    """Tracks problem evolution over time.

    Args不喜欢: 
        subreddit_names: Subreddits to monitor
        time_periods: Time intervals ('day', 'week', 'month')
        trend_keywords: Keywords to track
        growth_threshold: Minimum growth rate
    """
    if time_periods is None:
        time_periods = ["day", "week", "month"]
    if trend_keywords is None:
        trend_keywords = []
    trends = defaultdict(lambda: defaultdict(int))
    
    for name in subreddit_names:
        try:
            subreddit = get_reddit().subreddit(name)
            for period in time_periods:
                posts = subreddit.top(time_filter=period, limit=100)
                for post in posts:
                    text = f"{post.title} {post.selftext}".lower()
                    for kw in trend_keywords:
                        if kw.lower() in text:
                            trends[kw][period] += 1
        except Exception as e:
            logger.error(f"Error analyzing {name}: {e}")
    emerging = []
    for kw, periods in trends.items():
        if "week" in periods and "month" in periods:
            weekly_avg = periods["month"] / 4
            if weekly_avg and (growth := (periods["week"] - weekly_avg) / weekly_avg) > growth_threshold:
                emerging.append({"keyword": kw, "growth": growth, "mentions": periods["week"]})
    emerging.sort(key=lambda x: x["growth"], reverse=True)
    result = {"trends": emerging, "data": dict(trends)}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

@mcp.tool()
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
    personas, users = [], set()
    subs = ["technology", "software"]
    
    for sub in subs:
        try:
            subreddit = get_reddit().subreddit(sub)
            for post in subreddit.hot(limit=50):
                if post.author and post.author.name not in users and len(users) < user_sample_size:
                    users.add(post.author.name)
                    user = post.author
                    needs = defaultdict(int)
                    for item in user.submissions.new(limit=activity_depth):
                        text = item.title.lower()
                        for cat in need_categories:
                            if cat in text:
                                needs[cat] += 1
                    personas.append({"user": user.name, "subs": [s.display_name for s in user.subreddits(limit=5)], "needs": dict(needs)})
        except Exception as e:
            logger.error(f"Error extracting from {sub}: {e}")
    result = {"personas": personas, "total": len(personas)}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

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