#!/usr/bin/env python3
"""Comprehensive test of the optimized Reddit Scanner"""

import asyncio
import json
import time
from reddit_scanner import subreddit_pain_point_scanner

async def test_comprehensive():
    """Test the fully optimized implementation"""
    print("🚀 Testing comprehensive optimized Reddit Scanner...")
    print("Configuration: 3 subreddits, 50 posts each, NLP enabled, no comments")
    
    start_time = time.time()
    
    # Test with optimized defaults
    result = await subreddit_pain_point_scanner(
        subreddit_names=['Python', 'webdev', 'programming'],
        time_filter='week',
        limit=50,  # Using new default
        min_score=5,
        include_comments=False,  # Using new default (disabled)
        comment_depth=2,  # Using new default
        use_nlp=True,
        cluster_similar=True
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"⏱️  Test completed in {duration:.2f} seconds")
    
    # Parse and display results
    if result:
        data = json.loads(result[0].text)
        
        if "error" in data:
            print(f"❌ Error: {data['error']}")
            return
        
        print(f"📊 Results Summary:")
        print(f"  • Total pain points found: {data.get('total_found', 0)}")
        print(f"  • NLP enhanced: {data.get('nlp_enhanced', False)}")
        print(f"  • NLP detected: {data.get('nlp_found', 0)}")
        print(f"  • Keyword detected: {data.get('keyword_found', 0)}")
        print(f"  • Clusters generated: {len(data.get('clusters', []))}")
        
        # Performance assessment
        if duration < 30:
            print("✅ EXCELLENT - Under 30 seconds")
        elif duration < 45:
            print("✅ GOOD - Under 45 seconds")
        elif duration < 60:
            print("⚠️  ACCEPTABLE - Under 60 seconds")
        else:
            print("❌ SLOW - Over 60 seconds")
            
        # Show sample results
        if data.get('pain_points'):
            print(f"\n🔍 Sample Pain Points (top 3):")
            for i, pp in enumerate(data['pain_points'][:3], 1):
                print(f"  {i}. {pp.get('title', pp.get('body', 'N/A'))[:60]}...")
                print(f"     Score: {pp.get('score', 0)}, Method: {pp.get('method', 'N/A')}")
                
    else:
        print("❌ No results returned")

if __name__ == "__main__":
    asyncio.run(test_comprehensive())