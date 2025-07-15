#!/usr/bin/env python3
"""Test without comments processing"""

import asyncio
import json
import time
from reddit_scanner import subreddit_pain_point_scanner

async def test_no_comments():
    """Test without comments to isolate the issue"""
    print("🧪 Testing without comments...")
    
    start_time = time.time()
    
    # Test with no comments
    result = await subreddit_pain_point_scanner(
        subreddit_names=['Python'],
        time_filter='week',
        limit=10,  # Very small limit
        min_score=10,
        include_comments=False,  # No comments
        use_nlp=False,  # No NLP
        cluster_similar=False
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
        print(f"  - Keyword found: {data.get('keyword_found', 0)}")
        print("✅ Basic functionality working")
    else:
        print("❌ No results returned")

if __name__ == "__main__":
    asyncio.run(test_no_comments())