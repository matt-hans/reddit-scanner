import os
import re
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging

import praw
import spacy
import numpy as np
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

# Initialize spaCy instance
nlp = None
def get_nlp():
    global nlp
    if nlp is None:
        try:
            # Try to load the large model first
            nlp = spacy.load("en_core_web_lg")
            logger.info("Loaded spaCy large model (en_core_web_lg)")
        except OSError:
            try:
                # Fallback to medium model
                nlp = spacy.load("en_core_web_md")
                logger.warning("Large model not found, using medium model (en_core_web_md)")
            except OSError:
                try:
                    # Fallback to small model
                    nlp = spacy.load("en_core_web_sm")
                    logger.warning("Medium model not found, using small model (en_core_web_sm)")
                except OSError:
                    logger.error("No spaCy model found. Please install a spaCy model: python -m spacy download en_core_web_lg")
                    nlp = None
    return nlp

# NLP Status Functions
def get_nlp_status() -> Dict[str, Any]:
    """Get comprehensive status of spaCy NLP availability and setup guidance.
    
    Returns:
        Dict containing status, available models, installation commands, and guidance
    """
    status = {
        "nlp_available": False,
        "current_model": None,
        "available_models": [],
        "missing_models": [],
        "installation_commands": [],
        "status_message": "",
        "setup_guidance": "",
        "performance_note": ""
    }
    
    # Test each model in order of preference
    models_to_test = [
        ("en_core_web_lg", "Large model (best accuracy, ~800MB)", "Recommended for production use"),
        ("en_core_web_md", "Medium model (good accuracy, ~50MB)", "Good balance of speed and accuracy"),
        ("en_core_web_sm", "Small model (basic accuracy, ~15MB)", "Fastest but limited features")
    ]
    
    for model_name, description, note in models_to_test:
        try:
            test_nlp = spacy.load(model_name)
            status["available_models"].append({
                "name": model_name,
                "description": description,
                "note": note,
                "has_vectors": test_nlp.vocab.vectors_length > 0,
                "vector_size": test_nlp.vocab.vectors_length
            })
            if not status["nlp_available"]:  # Use first available model
                status["nlp_available"] = True
                status["current_model"] = {
                    "name": model_name,
                    "description": description,
                    "has_vectors": test_nlp.vocab.vectors_length > 0,
                    "vector_size": test_nlp.vocab.vectors_length
                }
        except OSError:
            status["missing_models"].append({
                "name": model_name,
                "description": description,
                "install_command": f"python -m spacy download {model_name}",
                "note": note
            })
    
    # Generate user-friendly messages and commands
    if status["nlp_available"]:
        current = status["current_model"]
        status["status_message"] = f"✅ NLP Available: Using {current['name']} ({current['description']})"
        if current["has_vectors"]:
            status["performance_note"] = "Full NLP features available including semantic similarity and advanced clustering."
        else:
            status["performance_note"] = "Basic NLP features available. Install a larger model for semantic similarity."
            
        if len(status["missing_models"]) > 0:
            best_missing = status["missing_models"][0]  # First (best) missing model
            status["setup_guidance"] = f"💡 For better accuracy, install: {best_missing['install_command']}"
    else:
        status["status_message"] = "❌ NLP Unavailable: No spaCy models found"
        status["setup_guidance"] = "🔧 To enable advanced NLP features:"
        
        # Add installation commands for missing models
        for model in status["missing_models"]:
            status["installation_commands"].append(model["install_command"])
    
    return status

# Initialize FastMCP server
mcp = FastMCP("reddit_opportunity_finder_enhanced")

# Initialize ThreadPoolExecutor for async operations
executor = ThreadPoolExecutor(max_workers=10)

# Rate limiting semaphore to prevent API flooding
reddit_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent Reddit operations
nlp_semaphore = asyncio.Semaphore(3)     # Max 3 concurrent NLP operations

# Async wrapper functions for blocking operations
async def async_get_reddit():
    """Async wrapper for Reddit initialization"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_reddit)

async def async_get_nlp():
    """Async wrapper for NLP initialization"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_nlp)

async def async_reddit_operation(func, *args, **kwargs):
    """Wrapper for Reddit API operations with rate limiting"""
    async with reddit_semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args, **kwargs)

async def async_nlp_operation(func, *args, **kwargs):
    """Wrapper for NLP operations with rate limiting"""
    async with nlp_semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args, **kwargs)

async def batch_nlp_process(texts: List[str], operations: List[str]) -> List[Dict]:
    """Process texts in batches using spaCy pipe() for efficiency"""
    try:
        nlp_instance = await async_get_nlp()
        if not nlp_instance:
            return []
        
        # Process all texts through spaCy pipe for efficiency
        def process_batch():
            docs = list(nlp_instance.pipe(texts))
            results = []
            
            for doc, text in zip(docs, texts):
                result = {"text": text}
                
                for operation in operations:
                    if operation == "entities":
                        result["entities"] = {
                            "companies": [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON"]],
                            "products": [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT"]],
                            "money": [ent.text for ent in doc.ents if ent.label_ in ["MONEY"]]
                        }
                    elif operation == "sentiment":
                        # Simple sentiment using spaCy
                        result["sentiment"] = calculate_sentiment_nlp(text, doc)
                    elif operation == "pain_points":
                        result["pain_points"] = extract_pain_points_nlp(text, doc)
                
                results.append(result)
            
            return results
        
        return await async_nlp_operation(process_batch)
    
    except Exception as e:
        logger.error(f"Batch NLP processing failed: {e}")
        return []

async def with_timeout(coro, timeout_seconds=45):
    """Wrap a coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout_seconds} seconds")
        raise

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

# NLP Helper Functions
def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text using spaCy NER."""
    nlp_instance = get_nlp()
    if not nlp_instance:
        return {"companies": [], "technologies": [], "products": [], "money": []}
    
    try:
        doc = nlp_instance(text)
        entities = {
            "companies": [],
            "technologies": [],
            "products": [],
            "money": []
        }
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities["companies"].append(ent.text)
            elif ent.label_ == "PRODUCT":
                entities["products"].append(ent.text)
            elif ent.label_ == "MONEY":
                entities["money"].append(ent.text)
            elif ent.label_ in ["PERSON", "GPE"] and any(tech_word in ent.text.lower() for tech_word in ["api", "app", "software", "tool"]):
                entities["technologies"].append(ent.text)
        
        return entities
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return {"companies": [], "technologies": [], "products": [], "money": []}

def extract_pain_points_nlp(text: str) -> List[Dict[str, Any]]:
    """Extract pain points using dependency parsing and patterns."""
    nlp_instance = get_nlp()
    if not nlp_instance:
        return []
    
    try:
        doc = nlp_instance(text)
        pain_points = []
        
        for sent in doc.sents:
            for token in sent:
                # Pattern: "struggling with X", "frustrated by Y"
                if token.dep_ == "prep" and token.head.lemma_ in ["struggle", "frustrate", "annoy", "hate"]:
                    pain_object = " ".join([child.text for child in token.subtree])
                    pain_points.append({
                        "type": "verb_prep_pattern",
                        "verb": token.head.text,
                        "issue": pain_object,
                        "context": sent.text,
                        "confidence": 0.8
                    })
                
                # Pattern: "X is difficult/tedious/annoying"
                elif token.pos_ == "ADJ" and token.lemma_ in ["difficult", "tedious", "annoying", "slow", "broken"]:
                    subjects = [child for child in token.head.children if child.dep_ == "nsubj"]
                    if subjects:
                        pain_points.append({
                            "type": "adjective_pattern",
                            "subject": subjects[0].text,
                            "problem": token.text,
                            "context": sent.text,
                            "confidence": 0.7
                        })
        
        return pain_points
    except Exception as e:
        logger.error(f"Error extracting pain points: {e}")
        return []

def calculate_sentiment_nlp(text: str) -> float:
    """Calculate sentiment using spaCy word vectors and similarity."""
    nlp_instance = get_nlp()
    if not nlp_instance or not nlp_instance.vocab.vectors_length:
        # Fallback to keyword-based sentiment
        positive_keywords = {"great": 0.8, "love": 1.0, "excellent": 0.9, "amazing": 0.9}
        negative_keywords = {"hate": -1.0, "terrible": -1.0, "awful": -0.9, "frustrating": -0.8}
        return calculate_sentiment(text, positive_keywords, negative_keywords)
    
    try:
        doc = nlp_instance(text)
        
        # Define sentiment anchor words
        positive_words = ["great", "excellent", "love", "perfect", "amazing", "wonderful"]
        negative_words = ["hate", "terrible", "awful", "frustrating", "annoying", "broken"]
        
        sentiment_score = 0
        word_count = 0
        
        for token in doc:
            if token.has_vector and token.pos_ in ["ADJ", "VERB", "NOUN"]:
                # Calculate similarity to positive/negative anchors
                pos_similarities = []
                neg_similarities = []
                
                for pos_word in positive_words:
                    pos_doc = nlp_instance(pos_word)
                    if pos_doc.vector_norm > 0:
                        pos_similarities.append(token.similarity(pos_doc))
                
                for neg_word in negative_words:
                    neg_doc = nlp_instance(neg_word)
                    if neg_doc.vector_norm > 0:
                        neg_similarities.append(token.similarity(neg_doc))
                
                if pos_similarities and neg_similarities:
                    max_pos = max(pos_similarities)
                    max_neg = max(neg_similarities)
                    sentiment_score += (max_pos - max_neg)
                    word_count += 1
        
        return sentiment_score / max(word_count, 1) if word_count else 0.0
    except Exception as e:
        logger.error(f"Error calculating sentiment: {e}")
        return 0.0

def extract_workflow_steps_nlp(text: str) -> List[Dict[str, Any]]:
    """Extract workflow steps using sentence segmentation and dependency parsing."""
    nlp_instance = get_nlp()
    if not nlp_instance:
        return []
    
    try:
        doc = nlp_instance(text)
        steps = []
        
        for sent in doc.sents:
            # Look for imperative sentences or numbered steps
            root_tokens = [token for token in sent if token.dep_ == "ROOT"]
            if root_tokens:
                root = root_tokens[0]
                if root.pos_ == "VERB":
                    # Extract action and objects
                    action = root.lemma_
                    objects = []
                    
                    for child in root.children:
                        if child.dep_ in ["dobj", "pobj", "attr"]:
                            objects.append(child.text)
                        elif child.dep_ == "prep":
                            # Get objects of preposition
                            for prep_child in child.children:
                                if prep_child.dep_ == "pobj":
                                    objects.append(f"{child.text} {prep_child.text}")
                    
                    if action and (objects or len(sent.text.split()) > 3):
                        steps.append({
                            "action": action,
                            "objects": objects,
                            "full_step": sent.text.strip(),
                            "complexity": len(objects) + (1 if len(sent.text.split()) > 10 else 0)
                        })
        
        return steps
    except Exception as e:
        logger.error(f"Error extracting workflow steps: {e}")
        return []

def classify_request_type_nlp(text: str) -> Dict[str, Any]:
    """Classify request type using spaCy patterns and entity recognition."""
    nlp_instance = get_nlp()
    if not nlp_instance:
        return {"type": "general", "confidence": 0.5, "indicators": []}
    
    try:
        doc = nlp_instance(text.lower())
        
        request_patterns = {
            "tool_request": {
                "patterns": ["looking for", "need", "recommend", "suggestion", "any.*tools"],
                "pos_patterns": [("VERB", "looking"), ("VERB", "need"), ("VERB", "recommend")]
            },
            "how_to": {
                "patterns": ["how to", "how do", "how can", "how should"],
                "pos_patterns": [("ADV", "how")]
            },
            "comparison": {
                "patterns": ["vs", "versus", "better than", "compare", "which is"],
                "pos_patterns": [("ADP", "than")]
            },
            "troubleshooting": {
                "patterns": ["error", "not working", "issue", "problem", "broken", "fails"],
                "pos_patterns": [("NOUN", "error"), ("NOUN", "problem")]
            }
        }
        
        scores = defaultdict(float)
        indicators = defaultdict(list)
        
        text_lower = text.lower()
        
        for category, config in request_patterns.items():
            # Pattern matching
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    scores[category] += 1.0
                    indicators[category].append(f"pattern: {pattern}")
            
            # POS pattern matching
            for pos, word in config["pos_patterns"]:
                for token in doc:
                    if token.pos_ == pos and word in token.lemma_:
                        scores[category] += 0.5
                        indicators[category].append(f"pos: {pos}={word}")
        
        if not scores:
            return {"type": "general", "confidence": 0.5, "indicators": []}
        
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 3.0, 1.0)  # Normalize to 0-1
        
        return {
            "type": best_type,
            "confidence": confidence,
            "indicators": indicators[best_type],
            "all_scores": dict(scores)
        }
    except Exception as e:
        logger.error(f"Error classifying request: {e}")
        return {"type": "general", "confidence": 0.5, "indicators": []}

def cluster_similar_texts(texts: List[str], threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Cluster similar texts using spaCy document similarity."""
    nlp_instance = get_nlp()
    if not nlp_instance or not nlp_instance.vocab.vectors_length:
        # Fallback: simple keyword-based clustering
        return [{"representative_text": text, "similar_texts": [text], "size": 1} for text in texts[:10]]
    
    try:
        # Process all texts
        docs = list(nlp_instance.pipe(texts))
        clusters = []
        used_indices = set()
        
        for i, doc1 in enumerate(docs):
            if i in used_indices or not doc1.vector_norm:
                continue
            
            cluster = {
                "representative_text": texts[i],
                "similar_texts": [texts[i]],
                "indices": [i],
                "size": 1
            }
            used_indices.add(i)
            
            # Find similar documents
            for j, doc2 in enumerate(docs):
                if j in used_indices or not doc2.vector_norm or i == j:
                    continue
                
                similarity = doc1.similarity(doc2)
                if similarity > threshold:
                    cluster["similar_texts"].append(texts[j])
                    cluster["indices"].append(j)
                    cluster["size"] += 1
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        # Sort clusters by size (largest first)
        clusters.sort(key=lambda x: x["size"], reverse=True)
        return clusters
    except Exception as e:
        logger.error(f"Error clustering texts: {e}")
        return [{"representative_text": text, "similar_texts": [text], "size": 1} for text in texts[:10]]

# Tools
async def process_subreddit_concurrent(subreddit_name: str, time_filter: str, limit: int, min_score: int, include_comments: bool, comment_depth: int, use_nlp: bool, pain_keywords: List[str]) -> Dict:
    """Process a single subreddit concurrently"""
    try:
        reddit = await async_get_reddit()
        subreddit = await async_reddit_operation(reddit.subreddit, subreddit_name)
        
        # Initialize aggregation variables
        pain_points = []
        nlp_pain_points = []
        category_counts = defaultdict(int)
        all_texts_for_clustering = []
        
        # Use iterator for lazy evaluation and early filtering
        posts_iter = subreddit.top(time_filter=time_filter, limit=limit)
        posts_list = list(posts_iter)
        logger.info(f"Fetched {len(posts_list)} posts from r/{subreddit_name}")
        
        # Process posts concurrently with progressive filtering
        post_tasks = []
        processed_count = 0
        
        for post in posts_list:
            if processed_count >= limit:
                break
                
            if post.score >= min_score:
                post_tasks.append(process_post_concurrent(post, include_comments, comment_depth, use_nlp, pain_keywords, min_score))
                processed_count += 1
                
                # Process in batches to avoid memory issues
                if len(post_tasks) >= 20:
                    batch_results = await asyncio.gather(*post_tasks, return_exceptions=True)
                    for result in batch_results:
                        if isinstance(result, Exception):
                            logger.warning(f"Post processing failed: {result}")
                            continue
                        
                        pain_points.extend(result['pain_points'])
                        nlp_pain_points.extend(result['nlp_pain_points'])
                        all_texts_for_clustering.extend(result['clustering_texts'])
                        for k, v in result['category_counts'].items():
                            category_counts[k] += v
                    
                    post_tasks = []  # Reset for next batch
        
        # Process remaining posts
        if post_tasks:
            batch_results = await asyncio.gather(*post_tasks, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.warning(f"Post processing failed: {result}")
                    continue
                
                pain_points.extend(result['pain_points'])
                nlp_pain_points.extend(result['nlp_pain_points'])
                all_texts_for_clustering.extend(result['clustering_texts'])
                for k, v in result['category_counts'].items():
                    category_counts[k] += v
        
        logger.info(f"r/{subreddit_name} results: {len(pain_points)} keyword matches, {len(nlp_pain_points)} NLP matches")
        
        return {
            'pain_points': pain_points,
            'nlp_pain_points': nlp_pain_points,
            'category_counts': category_counts,
            'clustering_texts': all_texts_for_clustering
        }
        
    except Exception as e:
        logger.error(f"Error scanning {subreddit_name}: {e}")
        return {
            'pain_points': [],
            'nlp_pain_points': [],
            'category_counts': defaultdict(int),
            'clustering_texts': []
        }

async def process_post_concurrent(post, include_comments: bool, comment_depth: int, use_nlp: bool, pain_keywords: List[str], min_score: int) -> Dict:
    """Process a single post concurrently"""
    pain_points = []
    nlp_pain_points = []
    category_counts = defaultdict(int)
    all_texts_for_clustering = []
    
    # Combine title and text
    full_text = f"{post.title} {post.selftext}"
    post_text_lower = full_text.lower()
    
    # Traditional keyword-based detection with improved phrase matching
    found_keywords = []
    for keyword in pain_keywords:
        # Handle both single words and phrases with proper matching
        if keyword.lower() in post_text_lower:
            found_keywords.append(keyword)
    
    if found_keywords:
        problem = '. '.join([s for s in post.selftext.split('.') if any(kw in s.lower() for kw in found_keywords)][:3])
        pain_point = {
            "type": "post", 
            "title": post.title, 
            "url": post.permalink, 
            "score": post.score, 
            "keywords": found_keywords, 
            "problem": problem,
            "method": "keyword"
        }
        pain_points.append(pain_point)
        for kw in found_keywords:
            category_counts[kw] += 1
    
    # Enhanced NLP-based detection
    nlp_instance = await async_get_nlp()
    if use_nlp and nlp_instance:
        try:
            # Batch NLP operations for efficiency
            nlp_tasks = [
                async_nlp_operation(extract_entities_from_text, full_text),
                async_nlp_operation(extract_pain_points_nlp, full_text),
                async_nlp_operation(calculate_sentiment_nlp, full_text)
            ]
            
            entities, nlp_extracted, sentiment = await asyncio.gather(*nlp_tasks)
            
            if nlp_extracted or sentiment < -0.3 or (sentiment < -0.1 and entities["companies"]):
                nlp_pain_point = {
                    "type": "post",
                    "title": post.title,
                    "url": post.permalink,
                    "score": post.score,
                    "method": "nlp",
                    "nlp_patterns": nlp_extracted,
                    "entities": entities,
                    "sentiment": sentiment,
                    "text_snippet": full_text[:300] + "..." if len(full_text) > 300 else full_text
                }
                nlp_pain_points.append(nlp_pain_point)
                all_texts_for_clustering.append(full_text)
                
                # Count NLP-detected pain point types
                for pattern in nlp_extracted:
                    category_counts[f"nlp_{pattern['type']}"] += 1
        
        except Exception as e:
            logger.warning(f"NLP processing failed for post {post.id}: {e}")
    
    # Process comments concurrently
    if include_comments:
        try:
            await async_reddit_operation(lambda: post.comments.replace_more(limit=0))
            comments_list = await async_reddit_operation(lambda: post.comments.list()[:comment_depth])
            
            # Process comments in batches
            comment_tasks = []
            for comment in comments_list:
                if comment.score >= min_score:
                    comment_tasks.append(process_comment_concurrent(comment, use_nlp, pain_keywords))
            
            if comment_tasks:
                comment_results = await asyncio.gather(*comment_tasks, return_exceptions=True)
                for result in comment_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Comment processing failed: {result}")
                        continue
                    pain_points.extend(result['pain_points'])
                    nlp_pain_points.extend(result['nlp_pain_points'])
                    all_texts_for_clustering.extend(result['clustering_texts'])
                    
        except Exception as e:
            logger.warning(f"Error processing comments for post {post.id}: {e}")
    
    return {
        'pain_points': pain_points,
        'nlp_pain_points': nlp_pain_points,
        'category_counts': category_counts,
        'clustering_texts': all_texts_for_clustering
    }

async def process_comment_concurrent(comment, use_nlp: bool, pain_keywords: List[str]) -> Dict:
    """Process a single comment concurrently"""
    pain_points = []
    nlp_pain_points = []
    clustering_texts = []
    
    # Traditional keyword detection
    if any(kw in comment.body.lower() for kw in pain_keywords):
        pain_points.append({
            "type": "comment", 
            "body": comment.body[:500], 
            "score": comment.score, 
            "url": comment.permalink,
            "method": "keyword"
        })
    
    # NLP detection
    nlp_instance = await async_get_nlp()
    if use_nlp and nlp_instance:
        try:
            nlp_comment_patterns, comment_sentiment = await asyncio.gather(
                async_nlp_operation(extract_pain_points_nlp, comment.body),
                async_nlp_operation(calculate_sentiment_nlp, comment.body)
            )
            
            if nlp_comment_patterns or comment_sentiment < -0.3:
                nlp_pain_points.append({
                    "type": "comment",
                    "body": comment.body[:500],
                    "score": comment.score,
                    "url": comment.permalink,
                    "method": "nlp",
                    "nlp_patterns": nlp_comment_patterns,
                    "sentiment": comment_sentiment
                })
                clustering_texts.append(comment.body)
        except Exception as e:
            logger.warning(f"NLP processing failed for comment: {e}")
    
    return {
        'pain_points': pain_points,
        'nlp_pain_points': nlp_pain_points,
        'clustering_texts': clustering_texts
    }

@mcp.tool()
async def subreddit_pain_point_scanner(
    subreddit_names: List[str],
    time_filter: str = "week",
    limit: int = 50,  # Reduced for performance
    pain_keywords: List[str] = None,
    min_score: int = 2,  # Lowered for better coverage
    include_comments: bool = False,  # Disabled by default for performance
    comment_depth: int = 2,  # Reduced for performance
    use_nlp: bool = True,
    cluster_similar: bool = True
) -> List[TextContent]:
    """Enhanced pain point scanner using spaCy NLP for better accuracy and insights.

    Args:
        subreddit_names: List of subreddit names to scan
        time_filter: Time filter for posts ('hour', 'day', 'week', 'month', 'year', 'all')
        limit: Maximum number of posts to analyze per subreddit
        pain_keywords: Keywords indicating pain points (used as fallback if NLP fails)
        min_score: Minimum score for posts/comments to consider
        include_comments: Whether to include comments in the scan
        comment_depth: Depth of comment threads to search
        use_nlp: Whether to use spaCy NLP for enhanced pain point detection
        cluster_similar: Whether to cluster similar pain points using spaCy similarity
    """
    if pain_keywords is None:
        pain_keywords = [
            "frustrated", "annoying", "wish there was", "need help with", "struggling with", 
            "pain point", "difficult", "tedious", "broken", "doesn't work", "hate when", 
            "takes forever", "buggy", "slow", "crash", "error", "problem", "issue",
            "terrible", "awful", "hate", "worst", "useless", "waste", "confusing",
            "complicated", "limitation", "missing", "lacking", "inadequate", "poor", "bad",
            "not working", "failed", "failing", "glitchy", "unstable"
        ]
    
    pain_points = []
    category_counts = defaultdict(int)
    nlp_pain_points = []
    all_texts_for_clustering = []
    
    # Process subreddits concurrently
    subreddit_tasks = []
    for subreddit_name in subreddit_names:
        subreddit_tasks.append(
            process_subreddit_concurrent(subreddit_name, time_filter, limit, min_score, include_comments, comment_depth, use_nlp, pain_keywords)
        )
    
    # Execute all subreddit processing concurrently with timeout
    try:
        subreddit_results = await with_timeout(asyncio.gather(*subreddit_tasks, return_exceptions=True), timeout_seconds=45)
    except asyncio.TimeoutError:
        logger.error("Subreddit processing timed out - returning partial results")
        return [TextContent(type="text", text=json.dumps({"error": "Processing timed out", "timeout_seconds": 45}, indent=2))]
    
    # Aggregate results from all subreddits
    for result in subreddit_results:
        if isinstance(result, Exception):
            logger.warning(f"Subreddit processing failed: {result}")
            continue
        
        pain_points.extend(result['pain_points'])
        nlp_pain_points.extend(result['nlp_pain_points'])
        all_texts_for_clustering.extend(result['clustering_texts'])
        for k, v in result['category_counts'].items():
            category_counts[k] += v
    
    # Combine and sort results
    all_pain_points = pain_points + nlp_pain_points
    all_pain_points.sort(key=lambda x: x["score"], reverse=True)
    
    # Cluster similar pain points if requested
    clusters = []
    nlp_instance = await async_get_nlp()
    if cluster_similar and all_texts_for_clustering and nlp_instance:
        try:
            clusters = await async_nlp_operation(cluster_similar_texts, all_texts_for_clustering, 0.6)
            logger.info(f"Created {len(clusters)} clusters from {len(all_texts_for_clustering)} texts")
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
    
    # Get comprehensive NLP status for user guidance
    nlp_status = get_nlp_status()
    
    # Prepare final result
    result = {
        "pain_points": all_pain_points[:50],
        "category_counts": dict(category_counts),
        "total_found": len(all_pain_points),
        "nlp_enhanced": use_nlp and nlp_status["nlp_available"],
        "nlp_found": len(nlp_pain_points),
        "keyword_found": len(pain_points),
        "nlp_status": {
            "available": nlp_status["nlp_available"],
            "status_message": nlp_status["status_message"],
            "performance_note": nlp_status["performance_note"],
            "setup_guidance": nlp_status["setup_guidance"],
            "installation_commands": nlp_status["installation_commands"] if not nlp_status["nlp_available"] else [],
            "current_model": nlp_status["current_model"]["name"] if nlp_status["current_model"] else None
        }
    }
    
    if clusters:
        result["clusters"] = [
            {
                "size": cluster["size"],
                "representative": cluster["representative_text"][:200] + "..." if len(cluster["representative_text"]) > 200 else cluster["representative_text"],
                "sample_texts": [text[:100] + "..." if len(text) > 100 else text for text in cluster["similar_texts"][:3]]
            }
            for cluster in clusters[:10]
        ]
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

@mcp.tool()
async def solution_request_tracker(
    subreddit_names: List[str],
    request_patterns: List[str] = None,
    exclude_solved: bool = True,
    min_engagement: int = 5,
    time_window: str = "month",
    category_keywords: Dict[str, List[str]] = None,
    use_nlp: bool = True,
    confidence_threshold: float = 0.6
) -> List[TextContent]:
    """Enhanced solution request tracker using spaCy NLP for accurate request classification.

    Args:
        subreddit_names: Target subreddits
        request_patterns: Patterns for requests (used as fallback)
        exclude_solved: Filter out solved posts
        min_engagement: Minimum comments for relevance
        time_window: Time period ('hour', 'day', 'week', 'month', 'year', 'all')
        category_keywords: Keywords to categorize requests
        use_nlp: Whether to use spaCy NLP for enhanced request classification
        confidence_threshold: Minimum confidence for NLP classification
    """
    if request_patterns is None:
        request_patterns = ["looking for.*app", "any.*tools", "recommend.*software", "need.*tool", "suggest.*solution"]
    if category_keywords is None:
        category_keywords = {
            "automation": ["automate", "workflow", "script", "batch"],
            "productivity": ["efficiency", "time", "organize", "task"],
            "development": ["code", "programming", "debug", "IDE"],
            "design": ["design", "graphics", "visual", "UI"],
            "analytics": ["data", "analytics", "report", "dashboard"],
            "communication": ["chat", "meeting", "collaboration", "team"]
        }
    
    requests = []
    nlp_requests = []
    request_type_counts = Counter()
    budget_mentions = []
    
    for subreddit_name in subreddit_names:
        try:
            reddit = await async_get_reddit()
            subreddit = await async_reddit_operation(reddit.subreddit, subreddit_name)
            posts_list = await async_reddit_operation(lambda: list(subreddit.top(time_filter=time_window, limit=100)))
            
            for post in posts_list:
                if post.num_comments < min_engagement:
                    continue
                
                full_text = f"{post.title} {post.selftext}"
                
                # Skip solved posts if requested
                if exclude_solved and ("[solved]" in post.title.lower() or "solved" in full_text.lower() or "update:" in post.title.lower()):
                    continue
                
                # Traditional pattern matching (keep as fallback)
                traditional_match = any(re.search(p, full_text, re.IGNORECASE) for p in request_patterns)
                if traditional_match:
                    category = next((cat for cat, kws in category_keywords.items() if any(kw in full_text.lower() for kw in kws)), "general")
                    requirements = extract_patterns(full_text, [r"need.*?(?=\.)", r"must.*?(?=\.)"])
                    
                    request = {
                        "title": post.title,
                        "url": post.permalink,
                        "score": post.score,
                        "comments": post.num_comments,
                        "category": category,
                        "requirements": requirements[:3],
                        "method": "traditional"
                    }
                    requests.append(request)
                
                # Enhanced NLP-based request classification
                nlp_instance = await async_get_nlp()
                if use_nlp and nlp_instance:
                    try:
                        # Classify request type using NLP
                        classification = await async_nlp_operation(classify_request_type_nlp, full_text)
                        
                        if classification["confidence"] >= confidence_threshold:
                            # Extract entities for budget/technology insights
                            entities = await async_nlp_operation(extract_entities_from_text, full_text)
                            
                            # Extract more sophisticated requirements using dependency parsing
                            pain_points = await async_nlp_operation(extract_pain_points_nlp, full_text)
                            
                            # Determine category using NLP + traditional keywords
                            nlp_category = "general"
                            max_category_score = 0
                            
                            for cat, kws in category_keywords.items():
                                score = 0
                                for kw in kws:
                                    if kw in full_text.lower():
                                        score += 1
                                if score > max_category_score:
                                    max_category_score = score
                                    nlp_category = cat
                            
                            # Extract budget information
                            budget_info = []
                            if entities["money"]:
                                budget_info = entities["money"]
                                budget_mentions.extend(entities["money"])
                            
                            # Extract detailed requirements
                            requirement_patterns = [
                                r"(?:need|want|require|looking for|must have)\s+(.{1,50}?)(?:\.|,|\n|$)",
                                r"(?:should|has to|needs to)\s+(.{1,50}?)(?:\.|,|\n|$)"
                            ]
                            detailed_requirements = []
                            for pattern in requirement_patterns:
                                matches = re.findall(pattern, full_text, re.IGNORECASE)
                                detailed_requirements.extend([match.strip() for match in matches])
                            
                            nlp_request = {
                                "title": post.title,
                                "url": post.permalink,
                                "score": post.score,
                                "comments": post.num_comments,
                                "method": "nlp",
                                "request_type": classification["type"],
                                "classification_confidence": classification["confidence"],
                                "classification_indicators": classification["indicators"],
                                "category": nlp_category,
                                "entities": entities,
                                "budget_mentions": budget_info,
                                "detailed_requirements": detailed_requirements[:5],
                                "pain_points": pain_points,
                                "text_snippet": full_text[:350] + "..." if len(full_text) > 350 else full_text
                            }
                            nlp_requests.append(nlp_request)
                            request_type_counts[classification["type"]] += 1
                    
                    except Exception as e:
                        logger.warning(f"NLP processing failed for request tracking: {e}")
        
        except Exception as e:
            logger.error(f"Error tracking {subreddit_name}: {e}")
    
    # Combine results and sort by engagement
    all_requests = requests + nlp_requests
    all_requests.sort(key=lambda x: x["score"] + x["comments"], reverse=True)
    
    # Analyze patterns across requests
    common_categories = Counter()
    budget_ranges = Counter()
    urgent_requests = []
    
    for request in all_requests:
        common_categories[request["category"]] += 1
        
        # Check for urgency indicators
        urgency_words = ["urgent", "asap", "immediately", "quickly", "deadline"]
        if any(word in request["title"].lower() for word in urgency_words):
            urgent_requests.append(request["title"])
    
    # Analyze budget mentions
    for budget in budget_mentions:
        # Categorize budget ranges
        if any(indicator in budget.lower() for indicator in ["free", "$0", "no cost"]):
            budget_ranges["free"] += 1
        elif any(indicator in budget for indicator in ["$", "€", "£"]):
            budget_ranges["paid"] += 1
    
    # Get comprehensive NLP status for user guidance
    nlp_status = get_nlp_status()
    
    # Prepare comprehensive result
    result = {
        "requests": all_requests[:50],
        "total_found": len(all_requests),
        "nlp_enhanced": use_nlp and nlp_status["nlp_available"],
        "request_types": dict(request_type_counts.most_common()),
        "common_categories": dict(common_categories.most_common()),
        "budget_analysis": {
            "ranges": dict(budget_ranges),
            "mentions": budget_mentions[:10]
        },
        "urgent_requests": len(urgent_requests),
        "urgent_samples": urgent_requests[:5],
        "engagement_stats": {
            "avg_score": round(sum(r["score"] for r in all_requests) / len(all_requests), 1) if all_requests else 0,
            "avg_comments": round(sum(r["comments"] for r in all_requests) / len(all_requests), 1) if all_requests else 0,
            "high_engagement": len([r for r in all_requests if r["score"] > 50 or r["comments"] > 20])
        },
        "nlp_status": {
            "available": nlp_status["nlp_available"],
            "status_message": nlp_status["status_message"],
            "performance_note": nlp_status["performance_note"],
            "setup_guidance": nlp_status["setup_guidance"],
            "installation_commands": nlp_status["installation_commands"] if not nlp_status["nlp_available"] else [],
            "current_model": nlp_status["current_model"]["name"] if nlp_status["current_model"] else None
        }
    }
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

@mcp.tool()
async def user_workflow_analyzer(
    subreddit_names: List[str],
    workflow_indicators: List[str] = None,
    min_steps: int = 3,
    efficiency_keywords: List[str] = None,
    use_nlp: bool = True,
    automation_score_threshold: float = 0.5
) -> List[TextContent]:
    """Enhanced workflow analyzer using spaCy sentence segmentation and dependency parsing.

    Args:
        subreddit_names: Subreddits to target
        workflow_indicators: Keywords like "process", "steps"
        min_steps: Minimum steps for complexity
        efficiency_keywords: Words indicating inefficiency
        use_nlp: Whether to use spaCy NLP for enhanced workflow detection
        automation_score_threshold: Minimum automation potential score (0-1)
    """
    if workflow_indicators is None:
        workflow_indicators = ["process", "steps", "routine", "manually", "workflow", "procedure"]
    if efficiency_keywords is None:
        efficiency_keywords = ["time-consuming", "repetitive", "tedious", "manual", "slow", "inefficient"]
    
    workflows = []
    nlp_workflows = []
    
    for subreddit_name in subreddit_names:
        try:
            reddit = await async_get_reddit()
            subreddit = await async_reddit_operation(reddit.subreddit, subreddit_name)
            posts_list = await async_reddit_operation(lambda: list(subreddit.search(" ".join(workflow_indicators), limit=100)))
            
            for post in posts_list:
                full_text = f"{post.title} {post.selftext}"
                
                # Traditional regex-based step extraction (keep as fallback)
                traditional_steps = extract_patterns(post.selftext, [r"(?:step\s*)?(\d+)[\.\)]\s*([^\n]+)", r"(?:then|next)\s*([^\n]+)"])
                traditional_issues = [kw for kw in efficiency_keywords if kw in post.selftext.lower()]
                
                if len(traditional_steps) >= min_steps:
                    workflow = {
                        "title": post.title,
                        "url": post.permalink,
                        "steps": traditional_steps[:5],
                        "issues": traditional_issues,
                        "method": "traditional"
                    }
                    workflows.append(workflow)
                
                # Enhanced NLP-based workflow analysis
                nlp_instance = await async_get_nlp()
                if use_nlp and nlp_instance:
                    try:
                        # Extract workflow steps using NLP
                        nlp_steps = await async_nlp_operation(extract_workflow_steps_nlp, full_text)
                        
                        # Calculate automation potential score
                        automation_score = 0.0
                        
                        # Score based on number of steps
                        if len(nlp_steps) >= min_steps:
                            automation_score += min(len(nlp_steps) / 10.0, 0.4)  # Max 0.4 for step count
                        
                        # Score based on efficiency issues
                        efficiency_mentions = sum(1 for kw in efficiency_keywords if kw in full_text.lower())
                        automation_score += min(efficiency_mentions / 5.0, 0.3)  # Max 0.3 for efficiency issues
                        
                        # Score based on manual/repetitive work indicators
                        manual_indicators = ["manual", "manually", "by hand", "copy", "paste", "repeat", "same thing"]
                        manual_mentions = sum(1 for indicator in manual_indicators if indicator in full_text.lower())
                        automation_score += min(manual_mentions / 5.0, 0.3)  # Max 0.3 for manual work
                        
                        if automation_score >= automation_score_threshold and len(nlp_steps) >= min_steps:
                            # Extract entities that might be relevant tools/technologies
                            entities = await async_nlp_operation(extract_entities_from_text, full_text)
                            
                            # Calculate complexity score
                            complexity_score = 0
                            for step in nlp_steps:
                                complexity_score += step.get("complexity", 0)
                            avg_complexity = complexity_score / len(nlp_steps) if nlp_steps else 0
                            
                            # Identify time-related mentions
                            time_patterns = [r"(\d+)\s*(?:minutes?|hours?|days?)", r"takes?\s*(\w+)\s*(?:minutes?|hours?)"]
                            time_mentions = []
                            for pattern in time_patterns:
                                time_mentions.extend(re.findall(pattern, full_text, re.IGNORECASE))
                            
                            nlp_workflow = {
                                "title": post.title,
                                "url": post.permalink,
                                "score": post.score,
                                "method": "nlp",
                                "nlp_steps": nlp_steps,
                                "automation_score": round(automation_score, 2),
                                "complexity_score": round(avg_complexity, 2),
                                "entities": entities,
                                "time_mentions": time_mentions,
                                "efficiency_issues": [kw for kw in efficiency_keywords if kw in full_text.lower()],
                                "text_snippet": full_text[:400] + "..." if len(full_text) > 400 else full_text
                            }
                            nlp_workflows.append(nlp_workflow)
                    
                    except Exception as e:
                        logger.warning(f"NLP processing failed for workflow analysis: {e}")
        
        except Exception as e:
            logger.error(f"Error analyzing {subreddit_name}: {e}")
    
    # Combine and sort results
    all_workflows = workflows + nlp_workflows
    all_workflows.sort(key=lambda x: x.get("automation_score", 0) if "automation_score" in x else len(x.get("steps", [])), reverse=True)
    
    # Aggregate insights
    automation_opportunities = []
    common_pain_points = Counter()
    tools_mentioned = Counter()
    
    for workflow in nlp_workflows:
        if workflow["automation_score"] > 0.7:
            automation_opportunities.append({
                "title": workflow["title"],
                "score": workflow["automation_score"],
                "step_count": len(workflow["nlp_steps"]),
                "complexity": workflow["complexity_score"]
            })
        
        # Count pain points and tools
        common_pain_points.update(workflow["efficiency_issues"])
        if workflow["entities"]["technologies"]:
            tools_mentioned.update(workflow["entities"]["technologies"])
        if workflow["entities"]["products"]:
            tools_mentioned.update(workflow["entities"]["products"])
    
    # Get comprehensive NLP status for user guidance
    nlp_status = get_nlp_status()
    
    # Prepare final result
    result = {
        "workflows": all_workflows[:50],
        "total_found": len(all_workflows),
        "nlp_enhanced": use_nlp and nlp_status["nlp_available"],
        "automation_opportunities": automation_opportunities[:10],
        "common_pain_points": common_pain_points.most_common(10),
        "tools_mentioned": tools_mentioned.most_common(10),
        "high_automation_potential": len([w for w in nlp_workflows if w["automation_score"] > 0.7]),
        "average_automation_score": round(sum(w["automation_score"] for w in nlp_workflows) / len(nlp_workflows), 2) if nlp_workflows else 0,
        "nlp_status": {
            "available": nlp_status["nlp_available"],
            "status_message": nlp_status["status_message"],
            "performance_note": nlp_status["performance_note"],
            "setup_guidance": nlp_status["setup_guidance"],
            "installation_commands": nlp_status["installation_commands"] if not nlp_status["nlp_available"] else [],
            "current_model": nlp_status["current_model"]["name"] if nlp_status["current_model"] else None
        }
    }
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

@mcp.tool()
async def competitor_mention_monitor(
    competitor_names: List[str],
    sentiment_threshold: float = 0.0,
    limitation_keywords: List[str] = None,
    use_nlp: bool = True,
    auto_discover_competitors: bool = False,
    subreddit_names: List[str] = None
) -> List[TextContent]:
    """Enhanced competitor analysis using NER and advanced sentiment analysis.

    Args:
        competitor_names: Software names to monitor
        sentiment_threshold: Sentiment score threshold (-1 to 1)
        limitation_keywords: Keywords indicating limitations
        use_nlp: Whether to use spaCy NLP for enhanced analysis
        auto_discover_competitors: Whether to auto-discover competitors using NER
        subreddit_names: Specific subreddits to search (defaults to 'all')
    """
    if limitation_keywords is None:
        limitation_keywords = ["but", "however", "missing", "lacks", "doesn't have", "wish it had", "limitation", "downside"]
    
    # Fallback sentiment keywords
    pos_keywords = {"great": 0.8, "love": 1.0, "excellent": 0.9, "amazing": 0.9, "perfect": 1.0}
    neg_keywords = {"hate": -1.0, "terrible": -1.0, "awful": -0.9, "frustrating": -0.8, "broken": -0.9}
    
    mentions = defaultdict(lambda: {"positive": [], "negative": [], "features": [], "missing_features": [], "neutral": []})
    discovered_competitors = set(competitor_names)
    
    # Define search targets
    search_targets = subreddit_names if subreddit_names else ["all"]
    
    for target in search_targets:
        try:
            reddit = await async_get_reddit()
            if target == "all":
                subreddit = await async_reddit_operation(reddit.subreddit, "all")
            else:
                subreddit = await async_reddit_operation(reddit.subreddit, target)
            
            # Search for each known competitor
            for name in competitor_names:
                try:
                    posts_list = await async_reddit_operation(lambda: list(subreddit.search(name, limit=100)))
                    for post in posts_list:
                        full_text = f"{post.title} {post.selftext}"
                        text_lower = full_text.lower()
                        
                        # Traditional sentiment analysis
                        traditional_sentiment = calculate_sentiment(text_lower, pos_keywords, neg_keywords)
                        
                        # Enhanced NLP analysis
                        nlp_instance = await async_get_nlp()
                        if use_nlp and nlp_instance:
                            try:
                                # Extract entities to find other tools/companies mentioned
                                entities = await async_nlp_operation(extract_entities_from_text, full_text)
                                
                                # Advanced sentiment analysis
                                nlp_sentiment = await async_nlp_operation(calculate_sentiment_nlp, full_text)
                                
                                # Use NLP sentiment if available, fallback to traditional
                                sentiment = nlp_sentiment if nlp_sentiment != 0.0 else traditional_sentiment
                                
                                # Extract pain points that might indicate missing features
                                pain_points = await async_nlp_operation(extract_pain_points_nlp, full_text)
                                
                                # Auto-discover competitors from entities
                                if auto_discover_competitors:
                                    for company in entities["companies"]:
                                        if company.lower() not in [c.lower() for c in discovered_competitors]:
                                            # Check if this looks like a software/tool company
                                            if any(tech_word in full_text.lower() for tech_word in ["software", "tool", "app", "platform", "service"]):
                                                discovered_competitors.add(company)
                                                logger.info(f"Auto-discovered potential competitor: {company}")
                                
                                # Extract features mentioned with the competitor
                                features_mentioned = []
                                if entities["products"] or entities["technologies"]:
                                    features_mentioned.extend(entities["products"])
                                    features_mentioned.extend(entities["technologies"])
                                
                                # Look for missing features using pain points and limitation keywords
                                missing_features = []
                                for pain_point in pain_points:
                                    if any(limit_kw in pain_point["context"].lower() for limit_kw in limitation_keywords):
                                        missing_features.append(pain_point["issue"])
                                
                                # Also check for traditional limitation keywords
                                traditional_limits = [kw for kw in limitation_keywords if kw in text_lower]
                                
                                # Categorize the mention
                                mention_data = {
                                    "title": post.title,
                                    "url": post.permalink,
                                    "score": post.score,
                                    "sentiment": sentiment,
                                    "traditional_sentiment": traditional_sentiment,
                                    "limitation_keywords": traditional_limits,
                                    "entities": entities,
                                    "features_mentioned": features_mentioned,
                                    "missing_features": missing_features,
                                    "pain_points": pain_points,
                                    "text_snippet": full_text[:300] + "..." if len(full_text) > 300 else full_text
                                }
                                
                                # Categorize based on sentiment
                                if sentiment > 0.2:
                                    mentions[name]["positive"].append(mention_data)
                                elif sentiment < -0.2:
                                    mentions[name]["negative"].append(mention_data)
                                else:
                                    mentions[name]["neutral"].append(mention_data)
                                
                                # Add to features and missing features lists
                                mentions[name]["features"].extend(features_mentioned)
                                mentions[name]["missing_features"].extend(missing_features)
                                
                            except Exception as e:
                                logger.warning(f"NLP processing failed for {name} mention: {e}")
                                # Fallback to traditional analysis
                                if traditional_sentiment < sentiment_threshold:
                                    limits = [kw for kw in limitation_keywords if kw in text_lower]
                                    if limits:
                                        mention_data = {
                                            "title": post.title,
                                            "url": post.permalink,
                                            "score": post.score,
                                            "sentiment": traditional_sentiment,
                                            "limitation_keywords": limits,
                                            "method": "traditional_fallback"
                                        }
                                        mentions[name]["negative"].append(mention_data)
                        else:
                            # Traditional analysis only
                            if traditional_sentiment < sentiment_threshold:
                                limits = [kw for kw in limitation_keywords if kw in text_lower]
                                if limits:
                                    mention_data = {
                                        "title": post.title,
                                        "url": post.permalink,
                                        "score": post.score,
                                        "sentiment": traditional_sentiment,
                                        "limitation_keywords": limits,
                                        "method": "traditional"
                                    }
                                    mentions[name]["negative"].append(mention_data)
                except Exception as e:
                    logger.error(f"Error searching for {name}: {e}")
        except Exception as e:
            logger.error(f"Error accessing target {target}: {e}")
    
    # Process and aggregate results
    final_mentions = {}
    top_missing_features = Counter()
    top_mentioned_features = Counter()
    
    for competitor, data in mentions.items():
        # Count feature mentions
        top_mentioned_features.update(data["features"])
        top_missing_features.update(data["missing_features"])
        
        # Sort mentions by score
        for category in ["positive", "negative", "neutral"]:
            data[category].sort(key=lambda x: x["score"], reverse=True)
        
        final_mentions[competitor] = {
            "positive_mentions": len(data["positive"]),
            "negative_mentions": len(data["negative"]),
            "neutral_mentions": len(data["neutral"]),
            "top_positive": data["positive"][:5],
            "top_negative": data["negative"][:5],
            "common_features": Counter(data["features"]).most_common(5),
            "missing_features": Counter(data["missing_features"]).most_common(5)
        }
    
    # Get comprehensive NLP status for user guidance
    nlp_status = get_nlp_status()
    
    # Prepare final result
    result = {
        "competitors_analyzed": dict(final_mentions),
        "discovered_competitors": list(discovered_competitors) if auto_discover_competitors else None,
        "top_missing_features_overall": top_missing_features.most_common(10),
        "top_mentioned_features_overall": top_mentioned_features.most_common(10),
        "analysis_enhanced": use_nlp and nlp_status["nlp_available"],
        "total_mentions": sum(len(data["positive"]) + len(data["negative"]) + len(data["neutral"]) for data in mentions.values()),
        "nlp_status": {
            "available": nlp_status["nlp_available"],
            "status_message": nlp_status["status_message"],
            "performance_note": nlp_status["performance_note"],
            "setup_guidance": nlp_status["setup_guidance"],
            "installation_commands": nlp_status["installation_commands"] if not nlp_status["nlp_available"] else [],
            "current_model": nlp_status["current_model"]["name"] if nlp_status["current_model"] else None
        }
    }
    
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
            reddit = await async_get_reddit()
            post = await async_reddit_operation(reddit.submission, id=post_id)
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
    async def analyze_subreddit(name: str) -> Dict[str, Any]:
        try:
            reddit = await async_get_reddit()
            subreddit = await async_reddit_operation(reddit.subreddit, name)
            
            # Check subscriber count
            subscribers = await async_reddit_operation(lambda: subreddit.subscribers)
            if not (min_subscribers <= subscribers <= max_subscribers):
                logger.debug(f"Skipping {name}: {subscribers} subscribers outside range")
                return None
                
            # Calculate activity rate
            posts = await async_reddit_operation(lambda: list(subreddit.new(limit=30)))
            if not posts:
                logger.debug(f"Skipping {name}: no recent posts")
                return None
                
            # Calculate posts per day rate
            days = max(1, (datetime.utcnow() - datetime.utcfromtimestamp(posts[-1].created_utc)).days)
            rate = len(posts) / days
            
            if rate < activity_threshold:
                logger.debug(f"Skipping {name}: activity rate {rate:.2f} below threshold")
                return None
                
            public_desc = await async_reddit_operation(lambda: subreddit.public_description)
            score = (subscribers / 100000) * rate
            return {
                "name": name,
                "subscribers": subscribers,
                "rate": rate,
                "score": score,
                "description": public_desc[:200] if public_desc else ""
            }
        except Exception as e:
            logger.error(f"Error analyzing {name}: {e}")
            return None
    
    # Helper function to find related subreddits
    async def find_related_subreddits(subreddit_name: str, visited: set) -> List[str]:
        related = []
        try:
            reddit = await async_get_reddit()
            subreddit = await async_reddit_operation(reddit.subreddit, subreddit_name)
            
            # Method 0: Use PRAW's built-in recommended method
            try:
                recommended = await async_reddit_operation(
                    reddit.subreddits.recommended,
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
                widgets = await async_reddit_operation(lambda: subreddit.widgets)
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
                posts_list = await async_reddit_operation(lambda: list(subreddit.hot(limit=10)))
                for post in posts_list:
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
                        search_results = await async_reddit_operation(lambda: list(reddit.subreddits.search(search_query, limit=5)))
                        for result in search_results:
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
                related = await find_related_subreddits(name, visited)
                next_batch.extend(related)
                logger.info(f"Seed {name}: found {len(related)} related subreddits")
                
                # Optionally analyze the seed itself if it meets criteria
                analysis = await analyze_subreddit(name)
                if analysis:
                    discovered.append(analysis)
                    logger.info(f"Seed {name} also meets criteria (subscribers: {analysis['subscribers']}, rate: {analysis['rate']:.2f})")
            else:
                # For discovered subreddits, analyze first then find related if they meet criteria
                analysis = await analyze_subreddit(name)
                if analysis:
                    discovered.append(analysis)
                    logger.info(f"Discovered: {name} (subscribers: {analysis['subscribers']}, rate: {analysis['rate']:.2f})")
                    
                    # Find related subreddits only if this one meets criteria and we're not at max depth
                    if depth < related_depth - 1:
                        related = await find_related_subreddits(name, visited)
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
            reddit = await async_get_reddit()
            subreddit = await async_reddit_operation(reddit.subreddit, name)
            for period in time_periods:
                posts_list = await async_reddit_operation(lambda: list(subreddit.top(time_filter=period, limit=100)))
                for post in posts_list:
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
            reddit = await async_get_reddit()
            subreddit = await async_reddit_operation(reddit.subreddit, sub)
            posts_list = await async_reddit_operation(lambda: list(subreddit.hot(limit=50)))
            for post in posts_list:
                if post.author and post.author.name not in users and len(users) < user_sample_size:
                    users.add(post.author.name)
                    user = post.author
                    needs = defaultdict(int)
                    submissions = await async_reddit_operation(lambda: list(user.submissions.new(limit=activity_depth)))
                    for item in submissions:
                        text = item.title.lower()
                        for cat in need_categories:
                            if cat in text:
                                needs[cat] += 1
                    user_subs = await async_reddit_operation(lambda: [s.display_name for s in user.subreddits(limit=5)])
                    personas.append({"user": user.name, "subs": user_subs, "needs": dict(needs)})
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

@mcp.tool()
async def nlp_content_clusterer(
    subreddit_names: List[str],
    content_types: List[str] = None,
    similarity_threshold: float = 0.7,
    min_cluster_size: int = 2,
    max_clusters: int = 20,
    time_filter: str = "week",
    min_score: int = 5,
    extract_insights: bool = True
) -> List[TextContent]:
    """Cluster similar Reddit content using spaCy semantic similarity to identify trending topics and patterns.

    Args:
        subreddit_names: List of subreddit names to analyze
        content_types: Types of content to cluster ('posts', 'comments', 'both') 
        similarity_threshold: Minimum similarity score for clustering (0-1)
        min_cluster_size: Minimum number of items per cluster
        max_clusters: Maximum number of clusters to return
        time_filter: Time filter for content ('hour', 'day', 'week', 'month', 'year', 'all')
        min_score: Minimum score for content to consider
        extract_insights: Whether to extract insights from clusters
    """
    if content_types is None:
        content_types = ["posts"]
    
    if not get_nlp():
        nlp_status = get_nlp_status()
        return [TextContent(type="text", text=json.dumps({
            "error": "spaCy NLP not available for content clustering", 
            "clusters": [], 
            "fallback_used": True,
            "nlp_status": {
                "available": False,
                "status_message": nlp_status["status_message"],
                "setup_guidance": nlp_status["setup_guidance"],
                "installation_commands": nlp_status["installation_commands"],
                "note": "Content clustering requires spaCy models for semantic similarity analysis"
            }
        }, indent=2))]
    
    all_content = []
    content_metadata = []
    
    for subreddit_name in subreddit_names:
        try:
            subreddit = get_reddit().subreddit(subreddit_name)
            posts = subreddit.top(time_filter=time_filter, limit=100)
            
            for post in posts:
                if post.score < min_score:
                    continue
                
                # Collect posts
                if "posts" in content_types or "both" in content_types:
                    post_text = f"{post.title} {post.selftext}".strip()
                    if len(post_text) > 50:  # Skip very short posts
                        all_content.append(post_text)
                        content_metadata.append({
                            "type": "post",
                            "title": post.title,
                            "url": post.permalink,
                            "score": post.score,
                            "subreddit": subreddit_name,
                            "num_comments": post.num_comments
                        })
                
                # Collect comments
                if "comments" in content_types or "both" in content_types:
                    try:
                        await async_reddit_operation(lambda: post.comments.replace_more(limit=0))
                        comments_list = await async_reddit_operation(lambda: post.comments.list()[:20])
                        for comment in comments_list:  # Limit comments per post
                            if comment.score >= min_score and len(comment.body) > 50:
                                all_content.append(comment.body)
                                content_metadata.append({
                                    "type": "comment",
                                    "body": comment.body[:200] + "..." if len(comment.body) > 200 else comment.body,
                                    "url": comment.permalink,
                                    "score": comment.score,
                                    "subreddit": subreddit_name,
                                    "parent_title": post.title
                                })
                    except Exception as e:
                        logger.warning(f"Error processing comments for post {post.id}: {e}")
        
        except Exception as e:
            logger.error(f"Error clustering content from {subreddit_name}: {e}")
    
    if not all_content:
        return [TextContent(type="text", text=json.dumps({
            "message": "No content found matching criteria",
            "clusters": [],
            "total_content": 0
        }, indent=2))]
    
    # Cluster the content using spaCy similarity
    try:
        clusters = await async_nlp_operation(cluster_similar_texts, all_content, threshold=similarity_threshold)
        
        # Filter clusters by minimum size and limit to max_clusters
        filtered_clusters = [c for c in clusters if c["size"] >= min_cluster_size][:max_clusters]
        
        # Enhance clusters with metadata and insights
        enhanced_clusters = []
        for i, cluster in enumerate(filtered_clusters):
            cluster_metadata = []
            total_score = 0
            subreddit_distribution = Counter()
            content_type_distribution = Counter()
            
            # Collect metadata for items in this cluster
            for idx in cluster["indices"]:
                if idx < len(content_metadata):
                    meta = content_metadata[idx]
                    cluster_metadata.append(meta)
                    total_score += meta["score"]
                    subreddit_distribution[meta["subreddit"]] += 1
                    content_type_distribution[meta["type"]] += 1
            
            # Extract insights if requested
            insights = {}
            if extract_insights and get_nlp():
                try:
                    # Get representative text for analysis
                    representative_text = cluster["representative_text"]
                    
                    # Extract entities
                    entities = extract_entities_from_text(representative_text)
                    
                    # Classify the topic type
                    classification = classify_request_type_nlp(representative_text)
                    
                    # Extract pain points
                    pain_points = extract_pain_points_nlp(representative_text)
                    
                    insights = {
                        "entities": entities,
                        "topic_classification": classification,
                        "pain_points": pain_points,
                        "sentiment": calculate_sentiment_nlp(representative_text)
                    }
                except Exception as e:
                    logger.warning(f"Error extracting insights for cluster {i}: {e}")
            
            enhanced_cluster = {
                "cluster_id": i + 1,
                "size": cluster["size"],
                "similarity_threshold_used": similarity_threshold,
                "representative_sample": cluster["representative_text"][:300] + "..." if len(cluster["representative_text"]) > 300 else cluster["representative_text"],
                "sample_texts": [text[:150] + "..." if len(text) > 150 else text for text in cluster["similar_texts"][:5]],
                "metadata": {
                    "total_engagement_score": total_score,
                    "avg_score": round(total_score / len(cluster_metadata), 1) if cluster_metadata else 0,
                    "subreddit_distribution": dict(subreddit_distribution.most_common()),
                    "content_types": dict(content_type_distribution),
                    "sample_items": cluster_metadata[:3]
                },
                "insights": insights
            }
            enhanced_clusters.append(enhanced_cluster)
        
        # Generate overall analysis
        total_items_clustered = sum(cluster["size"] for cluster in enhanced_clusters)
        unclustered_items = len(all_content) - total_items_clustered
        
        # Find trending topics
        trending_topics = []
        entity_mentions = Counter()
        
        for cluster in enhanced_clusters:
            if cluster["insights"] and "entities" in cluster["insights"]:
                entities = cluster["insights"]["entities"]
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        entity_mentions[entity] += cluster["size"]
        
        # Create trending topics from most mentioned entities
        for entity, mention_count in entity_mentions.most_common(10):
            trending_topics.append({
                "topic": entity,
                "mention_clusters": mention_count,
                "trend_strength": min(mention_count / len(enhanced_clusters), 1.0)
            })
        
        # Get comprehensive NLP status for user guidance
        nlp_status = get_nlp_status()
        
        # Prepare final result
        result = {
            "clusters": enhanced_clusters,
            "analysis": {
                "total_content_analyzed": len(all_content),
                "items_clustered": total_items_clustered,
                "unclustered_items": unclustered_items,
                "clustering_efficiency": round(total_items_clustered / len(all_content), 2) if all_content else 0,
                "avg_cluster_size": round(sum(c["size"] for c in enhanced_clusters) / len(enhanced_clusters), 1) if enhanced_clusters else 0
            },
            "trending_topics": trending_topics,
            "parameters": {
                "similarity_threshold": similarity_threshold,
                "min_cluster_size": min_cluster_size,
                "time_filter": time_filter,
                "subreddits_analyzed": subreddit_names
            },
            "nlp_enhanced": True,
            "nlp_status": {
                "available": nlp_status["nlp_available"],
                "status_message": nlp_status["status_message"],
                "performance_note": nlp_status["performance_note"],
                "current_model": nlp_status["current_model"]["name"] if nlp_status["current_model"] else None,
                "note": "Successfully used semantic similarity for content clustering"
            }
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    except Exception as e:
        logger.error(f"Error during content clustering: {e}")
        return [TextContent(type="text", text=json.dumps({
            "error": f"Clustering failed: {str(e)}",
            "total_content_collected": len(all_content),
            "fallback_summary": {
                "subreddits": subreddit_names,
                "content_count": len(all_content)
            }
        }, indent=2))]

# Simplified "Smart" Tools with Intelligent Defaults
@mcp.tool()
async def smart_pain_point_scanner(
    subreddit_names: List[str],
    time_filter: str = "month",
    focus: str = "all"
) -> List[TextContent]:
    """Simplified pain point scanner with intelligent defaults and automatic NLP configuration.
    
    Args:
        subreddit_names: List of subreddit names to scan
        time_filter: Time period ('day', 'week', 'month', 'year') - defaults to 'month'
        focus: Analysis focus ('pain_points', 'solutions', 'workflows', 'all') - defaults to 'all'
    """
    nlp_status = get_nlp_status()
    use_nlp = nlp_status["nlp_available"]
    
    # Configure settings based on focus
    if focus == "pain_points":
        pain_keywords = ["frustrated", "annoying", "difficult", "struggling with", "pain point", "broken", "terrible"]
        limit = 150
        min_score = 3
    elif focus == "solutions":
        pain_keywords = ["looking for", "need help", "recommend", "suggestion", "any tools"]
        limit = 100  
        min_score = 5
    elif focus == "workflows":
        pain_keywords = ["process", "steps", "workflow", "manually", "time-consuming", "repetitive"]
        limit = 100
        min_score = 5
    else:  # "all"
        pain_keywords = None  # Use defaults
        limit = 200
        min_score = 3
    
    # Call the full scanner with optimized settings
    result = await subreddit_pain_point_scanner(
        subreddit_names=subreddit_names,
        time_filter=time_filter,
        limit=limit,
        pain_keywords=pain_keywords,
        min_score=min_score,
        include_comments=True,
        comment_depth=5,
        use_nlp=use_nlp,
        cluster_similar=use_nlp  # Only cluster if NLP available
    )
    
    return result

@mcp.tool()
async def smart_solution_finder(
    subreddit_names: List[str],
    time_window: str = "month"
) -> List[TextContent]:
    """Simplified solution finder that automatically tracks requests and provides recommendations.
    
    Args:
        subreddit_names: Target subreddits to search
        time_window: Time period ('week', 'month', 'year') - defaults to 'month'
    """
    nlp_status = get_nlp_status()
    use_nlp = nlp_status["nlp_available"]
    
    # Call solution request tracker with smart defaults
    result = await solution_request_tracker(
        subreddit_names=subreddit_names,
        request_patterns=None,  # Use defaults
        exclude_solved=True,
        min_engagement=3,  # Lower threshold for more results
        time_window=time_window,
        category_keywords=None,  # Use defaults
        use_nlp=use_nlp,
        confidence_threshold=0.5  # Balanced threshold
    )
    
    return result

@mcp.tool()
async def quick_market_analysis(
    subreddit_names: List[str],
    analysis_type: str = "comprehensive"
) -> List[TextContent]:
    """One-click market analysis combining multiple tools for comprehensive insights.
    
    Args:
        subreddit_names: Subreddits to analyze
        analysis_type: 'quick', 'comprehensive', 'competitive' - defaults to 'comprehensive'
    """
    results = {}
    nlp_status = get_nlp_status()
    
    # Always include NLP status in final response
    results["nlp_status"] = {
        "available": nlp_status["nlp_available"],
        "status_message": nlp_status["status_message"],
        "setup_guidance": nlp_status["setup_guidance"] if not nlp_status["nlp_available"] else None
    }
    
    try:
        if analysis_type in ["quick", "comprehensive"]:
            # Get pain points
            pain_result = await smart_pain_point_scanner(subreddit_names, focus="pain_points")
            results["pain_points"] = json.loads(pain_result[0].text)
            
            # Get solution requests  
            solution_result = await smart_solution_finder(subreddit_names)
            results["solution_requests"] = json.loads(solution_result[0].text)
        
        if analysis_type == "comprehensive":
            # Add workflow analysis
            workflow_result = await user_workflow_analyzer(
                subreddit_names=subreddit_names,
                use_nlp=nlp_status["nlp_available"],
                automation_score_threshold=0.3
            )
            results["workflows"] = json.loads(workflow_result[0].text)
            
            # Add content clustering if NLP available
            if nlp_status["nlp_available"]:
                cluster_result = await nlp_content_clusterer(
                    subreddit_names=subreddit_names,
                    similarity_threshold=0.6,
                    min_cluster_size=2
                )
                results["content_clusters"] = json.loads(cluster_result[0].text)
    
    except Exception as e:
        results["error"] = f"Analysis failed: {str(e)}"
        logger.error(f"Quick market analysis failed: {e}")
    
    # Generate summary insights
    total_pain_points = results.get("pain_points", {}).get("total_found", 0)
    total_requests = results.get("solution_requests", {}).get("total_found", 0)
    total_workflows = results.get("workflows", {}).get("total_found", 0)
    
    results["summary"] = {
        "analysis_type": analysis_type,
        "subreddits_analyzed": subreddit_names,
        "total_pain_points": total_pain_points,
        "total_solution_requests": total_requests,
        "total_workflows": total_workflows,
        "nlp_enhanced": nlp_status["nlp_available"],
        "key_insight": f"Found {total_pain_points} pain points and {total_requests} solution requests across {len(subreddit_names)} subreddits"
    }
    
    return [TextContent(type="text", text=json.dumps(results, indent=2))]

@mcp.tool()
async def nlp_health_check() -> List[TextContent]:
    """Diagnostic tool to check NLP setup and provide troubleshooting guidance."""
    status = get_nlp_status()
    
    # Test basic functionality if NLP is available
    if status["nlp_available"]:
        try:
            # Test entity extraction
            test_entities = extract_entities_from_text("TestCorp released a new software tool for $100")
            
            # Test sentiment analysis
            test_sentiment = calculate_sentiment_nlp("This is a great tool that I love using")
            
            # Test clustering
            test_clustering = cluster_similar_texts(["software development", "app development", "mobile apps"])
            
            status["functionality_tests"] = {
                "entity_extraction": "✅ Working" if test_entities else "❌ Failed",
                "sentiment_analysis": "✅ Working" if isinstance(test_sentiment, float) else "❌ Failed", 
                "text_clustering": "✅ Working" if test_clustering else "❌ Failed"
            }
        except Exception as e:
            status["functionality_tests"] = {"error": f"Testing failed: {str(e)}"}
    
    return [TextContent(type="text", text=json.dumps(status, indent=2))]

if __name__ == "__main__":
    logger.info("Starting Reddit Scanner MCP Server...")
    logger.info(f"Server name: reddit_opportunity_finder_enhanced")
    logger.info(f"Transport: stdio")
    logger.info(f"Reddit credentials: {'Configured' if os.getenv('REDDIT_CLIENT_ID') else 'Not configured'}")
    
    # Display comprehensive NLP status at startup
    nlp_status = get_nlp_status()
    logger.info("=== NLP SETUP STATUS ===")
    logger.info(f"Status: {nlp_status['status_message']}")
    
    if nlp_status['nlp_available']:
        logger.info(f"Performance: {nlp_status['performance_note']}")
        if nlp_status['setup_guidance']:
            logger.info(f"Tip: {nlp_status['setup_guidance']}")
    else:
        logger.warning("⚠️  NLP FEATURES DISABLED - spaCy models not found")
        logger.warning(f"Guidance: {nlp_status['setup_guidance']}")
        for cmd in nlp_status['installation_commands']:
            logger.warning(f"Run: {cmd}")
        logger.warning("After installing models, restart the server to enable NLP features.")
    
    logger.info("========================")
    logger.info("Server is ready and waiting for MCP commands...")
    mcp.run(transport="stdio")