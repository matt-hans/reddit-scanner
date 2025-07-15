#!/usr/bin/env python3
"""
Comprehensive test suite for Reddit Scanner MCP tools with spaCy NLP enhancements.

This test suite covers:
- Authentication and initialization
- spaCy model loading with fallback chain
- NLP helper functions 
- Enhanced MCP tools with NLP integration
- Error handling and edge cases
- Backward compatibility
"""

import pytest
import asyncio
import json
import os
import types
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from collections import defaultdict, Counter
from datetime import datetime
import logging

# Import the module under test
import reddit_scanner as rs


# Test Fixtures and Helpers
class MockRedditPost:
    """Mock Reddit post object for testing"""
    def __init__(self, title="Test Post", selftext="Test content", score=10, 
                 num_comments=5, upvote_ratio=0.8, created_utc=None, 
                 permalink="/r/test/test_post", post_id="abc123"):
        self.title = title
        self.selftext = selftext
        self.score = score
        self.num_comments = num_comments
        self.upvote_ratio = upvote_ratio
        self.created_utc = created_utc or datetime.utcnow().timestamp()
        self.permalink = permalink
        self.id = post_id
        self.comments = MockCommentForest()

class MockComment:
    """Mock Reddit comment object"""
    def __init__(self, body="Test comment", score=5, permalink="/r/test/comment"):
        self.body = body
        self.score = score
        self.permalink = permalink

class MockCommentForest:
    """Mock Reddit comment forest"""
    def __init__(self, comments=None):
        self._comments = comments or [MockComment()]
    
    def replace_more(self, limit=0):
        pass
    
    def list(self):
        return self._comments[:20]  # Simulate limiting

class MockSubreddit:
    """Mock Reddit subreddit object"""
    def __init__(self, name="test", subscribers=10000):
        self.display_name = name
        self.subscribers = subscribers
        self.public_description = f"A test subreddit about {name}"
    
    def top(self, time_filter="week", limit=100):
        return [MockRedditPost() for _ in range(min(limit, 10))]
    
    def search(self, query, limit=100):
        return [MockRedditPost(title=f"Search result for {query}") for _ in range(min(limit, 5))]
    
    def hot(self, limit=100):
        return [MockRedditPost() for _ in range(min(limit, 10))]
    
    def new(self, limit=100):
        return [MockRedditPost() for _ in range(min(limit, 10))]

class MockReddit:
    """Mock Reddit instance"""
    def subreddit(self, name):
        return MockSubreddit(name)
    
    def submission(self, id):
        return MockRedditPost(post_id=id)

class MockSpacyDoc:
    """Mock spaCy document"""
    def __init__(self, text, entities=None, sents=None, vector_norm=1.0):
        self.text = text
        self.ents = entities or []
        self.sents = sents or [MockSpacySent(text)]
        self.vector_norm = vector_norm
        self.vector = [0.1] * 300  # Mock vector
    
    def similarity(self, other):
        return 0.7  # Mock similarity score
    
    def __iter__(self):
        # Mock tokens
        words = self.text.split()
        return iter([MockSpacyToken(word) for word in words])

class MockSpacyToken:
    """Mock spaCy token"""
    def __init__(self, text, pos_="NOUN", dep_="ROOT", lemma_=None):
        self.text = text
        self.pos_ = pos_
        self.dep_ = dep_
        self.lemma_ = lemma_ or text.lower()
        self.has_vector = True
        self.vector_norm = 1.0
        self.head = self
        self.children = []
        self.subtree = [self]
    
    def similarity(self, other):
        return 0.6

class MockSpacySent:
    """Mock spaCy sentence"""
    def __init__(self, text):
        self.text = text
        words = text.split()
        self.tokens = [MockSpacyToken(word) for word in words]
    
    def __iter__(self):
        return iter(self.tokens)

class MockSpacyEntity:
    """Mock spaCy named entity"""
    def __init__(self, text, label):
        self.text = text
        self.label_ = label

class MockSpacyVocab:
    """Mock spaCy vocabulary"""
    def __init__(self, vectors_length=300):
        self.vectors_length = vectors_length

class MockSpacyNLP:
    """Mock spaCy NLP model"""
    def __init__(self, model_name="en_core_web_lg", vectors_length=300):
        self.model_name = model_name
        self.vocab = MockSpacyVocab(vectors_length)
        self.vectors_length = vectors_length
    
    def __call__(self, text):
        # Return mock document with some entities
        entities = []
        if "company" in text.lower():
            entities.append(MockSpacyEntity("TestCorp", "ORG"))
        if "$" in text:
            entities.append(MockSpacyEntity("$100", "MONEY"))
        
        return MockSpacyDoc(text, entities)
    
    def pipe(self, texts):
        return [self(text) for text in texts]


# Pytest Fixtures
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables"""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test_client_secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "Test Agent 1.0")

@pytest.fixture
def mock_reddit():
    """Provide mock Reddit instance"""
    return MockReddit()

@pytest.fixture
def mock_spacy_nlp():
    """Provide mock spaCy NLP instance"""
    return MockSpacyNLP()

@pytest.fixture
def reset_globals():
    """Reset global instances before each test"""
    rs.reddit = None
    rs.nlp = None
    yield
    rs.reddit = None
    rs.nlp = None


# ==========================================
# AUTHENTICATION & INITIALIZATION TESTS
# ==========================================

class TestAuthentication:
    """Test Reddit authentication and initialization"""
    
    def test_get_reddit_missing_credentials(self, monkeypatch, reset_globals):
        """Test get_reddit() fails when credentials are missing"""
        # Clear environment variables
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        
        with pytest.raises(ValueError, match="Reddit API credentials not found"):
            rs.get_reddit()
    
    def test_get_reddit_missing_client_id(self, monkeypatch, reset_globals):
        """Test get_reddit() fails when only client_id is missing"""
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test_secret")
        
        with pytest.raises(ValueError):
            rs.get_reddit()
    
    def test_get_reddit_missing_client_secret(self, monkeypatch, reset_globals):
        """Test get_reddit() fails when only client_secret is missing"""
        monkeypatch.setenv("REDDIT_CLIENT_ID", "test_id")
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        
        with pytest.raises(ValueError):
            rs.get_reddit()
    
    @patch('reddit_scanner.praw.Reddit')
    def test_get_reddit_success(self, mock_praw, mock_env_vars, reset_globals):
        """Test successful Reddit initialization"""
        mock_reddit_instance = Mock()
        mock_praw.return_value = mock_reddit_instance
        
        result = rs.get_reddit()
        
        assert result == mock_reddit_instance
        mock_praw.assert_called_once_with(
            client_id="test_client_id",
            client_secret="test_client_secret",
            user_agent="Test Agent 1.0"
        )
    
    @patch('reddit_scanner.praw.Reddit')
    def test_get_reddit_default_user_agent(self, mock_praw, monkeypatch, reset_globals):
        """Test Reddit initialization with default user agent"""
        monkeypatch.setenv("REDDIT_CLIENT_ID", "test_id")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test_secret")
        monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
        
        mock_reddit_instance = Mock()
        mock_praw.return_value = mock_reddit_instance
        
        rs.get_reddit()
        
        mock_praw.assert_called_once_with(
            client_id="test_id",
            client_secret="test_secret",
            user_agent="MCP Reddit Analyzer 1.0"
        )
    
    @patch('reddit_scanner.praw.Reddit')
    def test_get_reddit_singleton_behavior(self, mock_praw, mock_env_vars, reset_globals):
        """Test that get_reddit() returns the same instance on multiple calls"""
        mock_reddit_instance = Mock()
        mock_praw.return_value = mock_reddit_instance
        
        reddit1 = rs.get_reddit()
        reddit2 = rs.get_reddit()
        
        assert reddit1 == reddit2
        assert mock_praw.call_count == 1  # Only called once


# ==========================================
# SPACY MODEL LOADING TESTS  
# ==========================================

class TestSpacyInitialization:
    """Test spaCy model loading with fallback chain"""
    
    def test_get_nlp_full_fallback_to_none(self, monkeypatch, reset_globals):
        """Test all three spaCy model loads fail → get_nlp() returns None"""
        call_count = {"num": 0}
        
        def fake_spacy_load(model_name):
            call_count["num"] += 1
            raise OSError("model not installed")
        
        mock_spacy = types.SimpleNamespace(load=fake_spacy_load)
        monkeypatch.setattr(rs, "spacy", mock_spacy)
        
        assert rs.get_nlp() is None
        assert call_count["num"] == 3  # Tried all three models
    
    def test_get_nlp_large_model_success(self, monkeypatch, reset_globals):
        """Test successful loading of large model"""
        mock_nlp = MockSpacyNLP("en_core_web_lg")
        
        def fake_spacy_load(model_name):
            if model_name == "en_core_web_lg":
                return mock_nlp
            raise OSError("model not found")
        
        mock_spacy = types.SimpleNamespace(load=fake_spacy_load)
        monkeypatch.setattr(rs, "spacy", mock_spacy)
        
        result = rs.get_nlp()
        assert result == mock_nlp
        assert result.model_name == "en_core_web_lg"
    
    def test_get_nlp_medium_fallback_success(self, monkeypatch, reset_globals):
        """Test fallback to medium model when large fails"""
        mock_nlp = MockSpacyNLP("en_core_web_md")
        
        def fake_spacy_load(model_name):
            if model_name == "en_core_web_lg":
                raise OSError("lg model not found")
            elif model_name == "en_core_web_md":
                return mock_nlp
            raise OSError("model not found")
        
        mock_spacy = types.SimpleNamespace(load=fake_spacy_load)
        monkeypatch.setattr(rs, "spacy", mock_spacy)
        
        result = rs.get_nlp()
        assert result == mock_nlp
        assert result.model_name == "en_core_web_md"
    
    def test_get_nlp_small_fallback_success(self, monkeypatch, reset_globals):
        """Test fallback to small model when lg and md fail"""
        mock_nlp = MockSpacyNLP("en_core_web_sm")
        
        def fake_spacy_load(model_name):
            if model_name in ["en_core_web_lg", "en_core_web_md"]:
                raise OSError("model not found")
            elif model_name == "en_core_web_sm":
                return mock_nlp
            raise OSError("model not found")
        
        mock_spacy = types.SimpleNamespace(load=fake_spacy_load)
        monkeypatch.setattr(rs, "spacy", mock_spacy)
        
        result = rs.get_nlp()
        assert result == mock_nlp
        assert result.model_name == "en_core_web_sm"
    
    def test_get_nlp_singleton_behavior(self, monkeypatch, reset_globals):
        """Test that get_nlp() returns the same instance on multiple calls"""
        mock_nlp = MockSpacyNLP()
        
        def fake_spacy_load(model_name):
            return mock_nlp
        
        mock_spacy = types.SimpleNamespace(load=fake_spacy_load)
        monkeypatch.setattr(rs, "spacy", mock_spacy)
        
        nlp1 = rs.get_nlp()
        nlp2 = rs.get_nlp()
        
        assert nlp1 == nlp2


# ==========================================
# HELPER FUNCTION TESTS
# ==========================================

class TestHelperFunctions:
    """Test utility helper functions"""
    
    def test_calculate_sentiment_weighted(self):
        """Test sentiment calculation with mixed positive/negative keywords"""
        pos_keywords = {"love": 1.0, "great": 0.8}
        neg_keywords = {"hate": -1.0, "terrible": -0.9}
        
        # Text with both positive and negative: love(+1.0) + hate(-1.0) / 2 = 0.0
        score = rs.calculate_sentiment("I love and hate this tool", pos_keywords, neg_keywords)
        assert score == pytest.approx(0.0, abs=1e-6)
    
    def test_calculate_sentiment_only_positive(self):
        """Test sentiment with only positive keywords"""
        pos_keywords = {"excellent": 0.9, "amazing": 0.8}
        neg_keywords = {"terrible": -1.0}
        
        score = rs.calculate_sentiment("This is excellent and amazing", pos_keywords, neg_keywords)
        assert score == pytest.approx(0.85, abs=0.01)  # (0.9 + 0.8) / 2
    
    def test_calculate_sentiment_no_matches(self):
        """Test sentiment with no keyword matches"""
        pos_keywords = {"great": 1.0}
        neg_keywords = {"bad": -1.0}
        
        score = rs.calculate_sentiment("This is a neutral statement", pos_keywords, neg_keywords)
        assert score == 0.0
    
    def test_extract_patterns_valid_regex(self):
        """Test pattern extraction with valid regex"""
        patterns = [r"test (\w+)", r"number (\d+)"]
        text = "test pattern and number 123"
        
        matches = rs.extract_patterns(text, patterns)
        assert "pattern" in matches
        assert "123" in matches
    
    def test_extract_patterns_invalid_regex(self, caplog):
        """Test pattern extraction with invalid regex"""
        caplog.set_level(logging.ERROR)
        
        patterns = [r"valid (\w+)", r"[", r"good"]  # Middle pattern is invalid
        text = "valid match and another good one"
        
        matches = rs.extract_patterns(text, patterns)
        
        # Should get matches from valid patterns
        assert "match" in matches
        assert "good" in matches
        # Should log error for invalid regex
        assert any("Invalid regex pattern" in rec.message for rec in caplog.records)
    
    def test_normalize_engagement_with_decay(self):
        """Test engagement normalization with time decay"""
        post = MockRedditPost(score=100, num_comments=50, upvote_ratio=0.8)
        
        engagement = rs.normalize_engagement(post, time_decay=True)
        
        # Should be positive but less than without decay due to time factor
        assert engagement > 0
        assert engagement < (100 * 0.4 + 50 * 0.4 + 0.8 * 100 * 0.2)
    
    def test_normalize_engagement_without_decay(self):
        """Test engagement normalization without time decay"""
        post = MockRedditPost(score=100, num_comments=50, upvote_ratio=0.8)
        
        engagement = rs.normalize_engagement(post, time_decay=False)
        
        expected = 100 * 0.4 + 50 * 0.4 + 0.8 * 100 * 0.2
        assert engagement == pytest.approx(expected, abs=0.01)


# ==========================================
# NLP HELPER FUNCTION TESTS
# ==========================================

class TestNLPHelperFunctions:
    """Test NLP-specific helper functions"""
    
    def test_extract_entities_from_text_with_nlp(self, monkeypatch, reset_globals):
        """Test entity extraction with spaCy NLP available"""
        mock_nlp = MockSpacyNLP()
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        text = "TestCorp company released a new product for $100"
        entities = rs.extract_entities_from_text(text)
        
        assert isinstance(entities, dict)
        assert "companies" in entities
        assert "products" in entities
        assert "money" in entities
        assert "technologies" in entities
    
    def test_extract_entities_from_text_without_nlp(self, monkeypatch, reset_globals):
        """Test entity extraction fallback when spaCy unavailable"""
        monkeypatch.setattr(rs, "get_nlp", lambda: None)
        
        entities = rs.extract_entities_from_text("Some text")
        
        assert entities == {"companies": [], "technologies": [], "products": [], "money": []}
    
    def test_extract_pain_points_nlp_with_nlp(self, monkeypatch, reset_globals):
        """Test pain point extraction with spaCy available"""
        mock_nlp = MockSpacyNLP()
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        text = "I am struggling with this difficult tool"
        pain_points = rs.extract_pain_points_nlp(text)
        
        assert isinstance(pain_points, list)
        # Mock implementation returns empty list, but structure is correct
    
    def test_extract_pain_points_nlp_without_nlp(self, monkeypatch, reset_globals):
        """Test pain point extraction fallback when spaCy unavailable"""
        monkeypatch.setattr(rs, "get_nlp", lambda: None)
        
        pain_points = rs.extract_pain_points_nlp("Some text")
        assert pain_points == []
    
    def test_calculate_sentiment_nlp_with_vectors(self, monkeypatch, reset_globals):
        """Test NLP sentiment calculation with word vectors"""
        mock_nlp = MockSpacyNLP(vectors_length=300)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        sentiment = rs.calculate_sentiment_nlp("This is great software")
        
        assert isinstance(sentiment, float)
        assert -1.0 <= sentiment <= 1.0
    
    def test_calculate_sentiment_nlp_without_vectors(self, monkeypatch, reset_globals):
        """Test NLP sentiment fallback to keyword-based when no vectors"""
        mock_nlp = MockSpacyNLP(vectors_length=0)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        sentiment = rs.calculate_sentiment_nlp("This is great software")
        
        assert isinstance(sentiment, float)
        # Should use keyword-based fallback
    
    def test_calculate_sentiment_nlp_without_nlp(self, monkeypatch, reset_globals):
        """Test sentiment calculation fallback when spaCy unavailable"""
        monkeypatch.setattr(rs, "get_nlp", lambda: None)
        
        sentiment = rs.calculate_sentiment_nlp("This is great software")
        
        assert isinstance(sentiment, float)
        # Should use keyword-based fallback
    
    def test_cluster_similar_texts_vectorless_fallback(self, monkeypatch, reset_globals):
        """Test clustering fallback when spaCy model has no vectors"""
        mock_nlp = MockSpacyNLP(vectors_length=0)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        texts = ["first text", "second text", "third text"]
        clusters = rs.cluster_similar_texts(texts, threshold=0.9)
        
        # Fallback creates one cluster per input with size 1
        assert len(clusters) == len(texts)
        assert all(c["size"] == 1 for c in clusters)
        # Representative texts should match originals
        reps = [c["representative_text"] for c in clusters]
        assert reps == texts
    
    def test_cluster_similar_texts_with_vectors(self, monkeypatch, reset_globals):
        """Test clustering with spaCy vectors available"""
        mock_nlp = MockSpacyNLP(vectors_length=300)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        texts = ["first text", "second text"]
        clusters = rs.cluster_similar_texts(texts, threshold=0.5)
        
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        assert all("representative_text" in cluster for cluster in clusters)
    
    def test_cluster_similar_texts_without_nlp(self, monkeypatch, reset_globals):
        """Test clustering fallback when spaCy unavailable"""
        monkeypatch.setattr(rs, "get_nlp", lambda: None)
        
        texts = ["first text", "second text", "third text"]
        clusters = rs.cluster_similar_texts(texts)
        
        # Should return fallback clusters (first 10 items, one per cluster)
        assert len(clusters) == 3
        assert all(c["size"] == 1 for c in clusters)


# ==========================================
# MCP TOOL TESTS
# ==========================================

class TestMCPTools:
    """Test MCP tool functions"""
    
    @pytest.mark.asyncio
    async def test_subreddit_pain_point_scanner_basic(self, monkeypatch, reset_globals):
        """Test basic pain point scanner functionality"""
        mock_reddit = MockReddit()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: None)  # Disable NLP for basic test
        
        result = await rs.subreddit_pain_point_scanner(
            subreddit_names=["test"],
            use_nlp=False
        )
        
        assert len(result) == 1
        assert result[0].type == "text"
        data = json.loads(result[0].text)
        assert "pain_points" in data
        assert "total_found" in data
        assert "nlp_enhanced" in data
        assert data["nlp_enhanced"] is False
    
    @pytest.mark.asyncio
    async def test_subreddit_pain_point_scanner_with_nlp(self, monkeypatch, reset_globals):
        """Test pain point scanner with NLP enabled"""
        mock_reddit = MockReddit()
        mock_nlp = MockSpacyNLP()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        result = await rs.subreddit_pain_point_scanner(
            subreddit_names=["test"],
            use_nlp=True
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["nlp_enhanced"] is True
        assert "nlp_found" in data
        assert "keyword_found" in data
    
    @pytest.mark.asyncio
    async def test_solution_request_tracker_basic(self, monkeypatch, reset_globals):
        """Test basic solution request tracker functionality"""
        mock_reddit = MockReddit()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: None)
        
        result = await rs.solution_request_tracker(
            subreddit_names=["test"],
            use_nlp=False
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "requests" in data
        assert "total_found" in data
        assert "nlp_enhanced" in data
        assert data["nlp_enhanced"] is False
    
    @pytest.mark.asyncio
    async def test_solution_request_tracker_with_nlp(self, monkeypatch, reset_globals):
        """Test solution request tracker with NLP enabled"""
        mock_reddit = MockReddit()
        mock_nlp = MockSpacyNLP()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        result = await rs.solution_request_tracker(
            subreddit_names=["test"],
            use_nlp=True,
            confidence_threshold=0.5
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["nlp_enhanced"] is True
        assert "request_types" in data
    
    @pytest.mark.asyncio
    async def test_user_workflow_analyzer_basic(self, monkeypatch, reset_globals):
        """Test basic workflow analyzer functionality"""
        mock_reddit = MockReddit()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: None)
        
        result = await rs.user_workflow_analyzer(
            subreddit_names=["test"],
            use_nlp=False
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "workflows" in data
        assert "total_found" in data
        assert "nlp_enhanced" in data
        assert data["nlp_enhanced"] is False
    
    @pytest.mark.asyncio
    async def test_user_workflow_analyzer_with_nlp(self, monkeypatch, reset_globals):
        """Test workflow analyzer with NLP enabled"""
        mock_reddit = MockReddit()
        mock_nlp = MockSpacyNLP()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        result = await rs.user_workflow_analyzer(
            subreddit_names=["test"],
            use_nlp=True,
            automation_score_threshold=0.3
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["nlp_enhanced"] is True
        assert "automation_opportunities" in data
    
    @pytest.mark.asyncio
    async def test_competitor_mention_monitor_basic(self, monkeypatch, reset_globals):
        """Test basic competitor mention monitor functionality"""
        mock_reddit = MockReddit()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: None)
        
        result = await rs.competitor_mention_monitor(
            competitor_names=["TestTool", "CompetitorApp"],
            use_nlp=False
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "competitors_analyzed" in data
        assert "analysis_enhanced" in data
        assert data["analysis_enhanced"] is False
    
    @pytest.mark.asyncio
    async def test_competitor_mention_monitor_with_nlp(self, monkeypatch, reset_globals):
        """Test competitor mention monitor with NLP enabled"""
        mock_reddit = MockReddit()
        mock_nlp = MockSpacyNLP()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        result = await rs.competitor_mention_monitor(
            competitor_names=["TestTool"],
            use_nlp=True,
            auto_discover_competitors=True
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["analysis_enhanced"] is True
        assert "discovered_competitors" in data
    
    @pytest.mark.asyncio
    async def test_nlp_content_clusterer_without_nlp(self, monkeypatch, reset_globals):
        """Test content clusterer when NLP unavailable"""
        mock_reddit = MockReddit()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: None)
        
        result = await rs.nlp_content_clusterer(
            subreddit_names=["test"]
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data
        assert "spaCy NLP not available" in data["error"]
    
    @pytest.mark.asyncio
    async def test_nlp_content_clusterer_with_nlp(self, monkeypatch, reset_globals):
        """Test content clusterer with NLP available"""
        mock_reddit = MockReddit()
        mock_nlp = MockSpacyNLP()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        result = await rs.nlp_content_clusterer(
            subreddit_names=["test"],
            similarity_threshold=0.7,
            min_cluster_size=2
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "clusters" in data
        assert "analysis" in data
        assert "nlp_enhanced" in data
        assert data["nlp_enhanced"] is True
    
    @pytest.mark.asyncio
    async def test_subreddit_engagement_analyzer(self, monkeypatch, reset_globals):
        """Test engagement analyzer functionality"""
        mock_reddit = MockReddit()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        
        result = await rs.subreddit_engagement_analyzer(
            post_ids=["test123", "test456"]
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "analyzed" in data
        assert "avg_engagement" in data


# ==========================================
# ERROR HANDLING TESTS
# ==========================================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_tool_with_reddit_api_error(self, monkeypatch, reset_globals):
        """Test tool behavior when Reddit API fails"""
        def failing_reddit():
            raise Exception("Reddit API Error")
        
        monkeypatch.setattr(rs, "get_reddit", failing_reddit)
        
        result = await rs.subreddit_pain_point_scanner(
            subreddit_names=["test"],
            use_nlp=False
        )
        
        # Should not crash, should return valid JSON response
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "pain_points" in data
        assert data["total_found"] == 0  # No results due to error
    
    def test_extract_entities_error_handling(self, monkeypatch, reset_globals):
        """Test entity extraction error handling"""
        def failing_nlp(text):
            raise Exception("NLP processing failed")
        
        mock_nlp = Mock()
        mock_nlp.side_effect = failing_nlp
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        # Should not crash, should return empty results
        entities = rs.extract_entities_from_text("test text")
        assert entities == {"companies": [], "technologies": [], "products": [], "money": []}
    
    @pytest.mark.asyncio
    async def test_empty_subreddit_list(self, monkeypatch, reset_globals):
        """Test tool behavior with empty subreddit list"""
        mock_reddit = MockReddit()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        
        result = await rs.subreddit_pain_point_scanner(
            subreddit_names=[],  # Empty list
            use_nlp=False
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["total_found"] == 0
    
    @pytest.mark.asyncio
    async def test_invalid_subreddit_name(self, monkeypatch, reset_globals):
        """Test tool behavior with invalid subreddit names"""
        def failing_subreddit(name):
            if name == "invalid_subreddit":
                raise Exception("Subreddit not found")
            return MockSubreddit(name)
        
        mock_reddit = Mock()
        mock_reddit.subreddit = failing_subreddit
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        
        result = await rs.subreddit_pain_point_scanner(
            subreddit_names=["invalid_subreddit"],
            use_nlp=False
        )
        
        # Should handle error gracefully
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["total_found"] == 0


# ==========================================
# INTEGRATION TESTS
# ==========================================

class TestIntegration:
    """Test integration scenarios and backward compatibility"""
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_use_nlp_false(self, monkeypatch, reset_globals):
        """Test that use_nlp=False maintains backward compatibility"""
        mock_reddit = MockReddit()
        mock_nlp = MockSpacyNLP()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        # Even with NLP available, use_nlp=False should use traditional methods
        result = await rs.subreddit_pain_point_scanner(
            subreddit_names=["test"],
            use_nlp=False
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["nlp_enhanced"] is False
        assert data["nlp_found"] == 0  # No NLP processing should occur
    
    @pytest.mark.asyncio
    async def test_combined_traditional_and_nlp_results(self, monkeypatch, reset_globals):
        """Test that tools combine traditional and NLP results when both available"""
        mock_reddit = MockReddit()
        mock_nlp = MockSpacyNLP()
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", lambda: mock_nlp)
        
        result = await rs.subreddit_pain_point_scanner(
            subreddit_names=["test"],
            use_nlp=True
        )
        
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["nlp_enhanced"] is True
        # Should have both traditional and NLP results
        assert "keyword_found" in data
        assert "nlp_found" in data
    
    @pytest.mark.asyncio
    async def test_graceful_nlp_failure_during_processing(self, monkeypatch, reset_globals):
        """Test graceful handling when NLP fails mid-processing"""
        mock_reddit = MockReddit()
        
        # Mock NLP that fails on certain calls
        call_count = {"count": 0}
        def failing_nlp_sometimes():
            call_count["count"] += 1
            if call_count["count"] > 2:  # Fail after a few successful calls
                return None
            return MockSpacyNLP()
        
        monkeypatch.setattr(rs, "get_reddit", lambda: mock_reddit)
        monkeypatch.setattr(rs, "get_nlp", failing_nlp_sometimes)
        
        result = await rs.subreddit_pain_point_scanner(
            subreddit_names=["test"],
            use_nlp=True
        )
        
        # Should still return valid results even with partial NLP failure
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "pain_points" in data
        assert data["total_found"] >= 0


if __name__ == "__main__":
    # Add pytest dependencies to project for running tests
    print("Run tests with: python -m pytest test_reddit_scanner.py -v")
    print("For coverage: python -m pytest test_reddit_scanner.py --cov=reddit_scanner")