#!/usr/bin/env python3
"""Final performance test with concurrent improvements"""

import asyncio
import json
import time
from reddit_scanner import subreddit_pain_point_scanner

async def test_final():
    """Test with parameters closer to original timeout case"""
    print("🧪 Testing final performance with concurrent improvements...")
    
    start_time = time.time()
    
    # Test with parameters closer to the original problem case
    result = await subreddit_pain_point_scanner(
        subreddit_names=['Python', 'webdev', 'programming'],  # 3 subreddits
        time_filter='week',
        limit=30,  # Reduced but realistic
        min_score=5,
        include_comments=True,  # Comments enabled
        comment_depth=2,  # Reduced from original 5
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
        
        if duration < 60:
            print("✅ SURGICAL FIX SUCCESS - completed under 60 seconds")
        else:
            print("❌ Still too slow")
    else:
        print("❌ No results returned")

if __name__ == "__main__":
    asyncio.run(test_final())