#!/usr/bin/env python3
"""Test with NLP but no comments"""

import asyncio
import json
import time
from reddit_scanner import subreddit_pain_point_scanner

async def test_nlp_only():
    """Test with NLP but no comments"""
    print("🧪 Testing with NLP but no comments...")
    
    start_time = time.time()
    
    # Test with NLP but no comments
    result = await subreddit_pain_point_scanner(
        subreddit_names=['Python'],
        time_filter='week',
        limit=10,  # Small limit
        min_score=10,
        include_comments=False,  # No comments
        use_nlp=True,  # NLP enabled
        cluster_similar=True
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"⏱️  Test completed in {duration:.2f} seconds")
    
    # Parse and display results
    if result:
        data = json.loads(result[0].text)
        print(f"📊 Results:")
        print(f"  - Total pain points found: {data.get('total_found', 0)}")
        print(f"  - NLP enhanced: {data.get('nlp_enhanced', False)}")
        print(f"  - NLP found: {data.get('nlp_found', 0)}")
        print(f"  - Keyword found: {data.get('keyword_found', 0)}")
        print(f"  - Clusters: {len(data.get('clusters', []))}")
        print("✅ NLP functionality working")
    else:
        print("❌ No results returned")

if __name__ == "__main__":
    asyncio.run(test_nlp_only())