#!/usr/bin/env python3
"""Simple performance test for the Reddit Scanner"""

import asyncio
import json
import time
from reddit_scanner import subreddit_pain_point_scanner

async def test_performance():
    """Test the pain point scanner directly"""
    print("🧪 Testing pain point scanner performance...")
    
    start_time = time.time()
    
    # Test with smaller parameters
    result = await subreddit_pain_point_scanner(
        subreddit_names=['Python'],  # Only 1 subreddit
        time_filter='week',
        limit=20,  # Reduced from 100
        min_score=5,
        include_comments=True,
        comment_depth=2,  # Reduced from 5
        use_nlp=True,
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
        
        if duration < 30:
            print("✅ Performance test PASSED - completed in under 30 seconds")
        else:
            print("❌ Performance test FAILED - took too long")
    else:
        print("❌ No results returned")

if __name__ == "__main__":
    asyncio.run(test_performance())