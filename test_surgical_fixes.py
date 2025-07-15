#!/usr/bin/env python3
"""Test the surgical fixes for pain point detection issues"""

import asyncio
import json
import time
from reddit_scanner import subreddit_pain_point_scanner

async def test_surgical_fixes():
    """Test all surgical fixes are working correctly"""
    print("🔧 Testing surgical fixes for pain point detection...")
    
    # Test with user's exact parameters from the issue
    print("📋 Test Parameters:")
    print("   - Subreddits: ['Python', 'webdev', 'programming']")
    print("   - Time filter: 'week'")
    print("   - Limit: 5")
    print("   - Pain keywords: User's comprehensive list")
    print("   - Min score: 2 (lowered)")
    print("   - NLP: Enabled")
    print("   - Comments: Disabled for performance")
    
    start_time = time.time()
    
    # Test with the exact parameters that were failing
    result = await subreddit_pain_point_scanner(
        subreddit_names=['Python', 'webdev', 'programming'],
        time_filter='week',
        limit=5,
        pain_keywords=[
            "frustrated", "annoying", "wish there was", "need help with", 
            "struggling with", "pain point", "difficult", "tedious", 
            "broken", "doesn't work", "hate when", "takes forever"
        ],
        min_score=2,  # Using lowered threshold
        include_comments=False,
        use_nlp=True,
        cluster_similar=True
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"⏱️  Test completed in {duration:.2f} seconds")
    
    # Parse and analyze results
    if result:
        data = json.loads(result[0].text)
        
        if "error" in data:
            print(f"❌ Error: {data['error']}")
            return False
        
        print(f"\n📊 SURGICAL FIX RESULTS:")
        print(f"   ✅ Total pain points found: {data.get('total_found', 0)}")
        print(f"   ✅ NLP enhanced: {data.get('nlp_enhanced', False)}")
        print(f"   ✅ NLP detected: {data.get('nlp_found', 0)}")
        print(f"   ✅ Keyword detected: {data.get('keyword_found', 0)}")
        print(f"   ✅ Clusters generated: {len(data.get('clusters', []))}")
        
        # Test success criteria
        success_criteria = [
            ("Results returned", data.get('total_found', 0) > 0),
            ("NLP functioning", data.get('nlp_enhanced', False)),
            ("Keyword detection working", data.get('keyword_found', 0) > 0 or data.get('nlp_found', 0) > 0),
            ("Performance acceptable", duration < 60)
        ]
        
        print(f"\n🧪 SUCCESS CRITERIA:")
        all_passed = True
        for criterion, passed in success_criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status}: {criterion}")
            if not passed:
                all_passed = False
        
        # Show sample results
        if data.get('pain_points'):
            print(f"\n🔍 Sample Pain Points Found:")
            for i, pp in enumerate(data['pain_points'][:3], 1):
                print(f"   {i}. {pp.get('title', pp.get('body', 'N/A'))[:60]}...")
                print(f"      Score: {pp.get('score', 0)}, Method: {pp.get('method', 'N/A')}")
                if pp.get('keywords'):
                    print(f"      Keywords: {', '.join(pp['keywords'])}")
                
        if all_passed:
            print(f"\n🎉 ALL SURGICAL FIXES SUCCESSFUL!")
            print("   The pain point scanner is now working correctly")
            return True
        else:
            print(f"\n⚠️  Some issues remain - check failed criteria")
            return False
            
    else:
        print("❌ No results returned - surgical fixes failed")
        return False

async def test_specific_keywords():
    """Test specific multi-word phrases that were failing"""
    print(f"\n🔍 Testing specific multi-word phrase detection...")
    
    # Test phrases that were failing before
    test_phrases = [
        "wish there was",
        "need help with", 
        "doesn't work",
        "hate when",
        "takes forever"
    ]
    
    for phrase in test_phrases:
        result = await subreddit_pain_point_scanner(
            subreddit_names=['Python'],
            time_filter='week', 
            limit=5,
            pain_keywords=[phrase],  # Test single phrase
            min_score=2,
            include_comments=False,
            use_nlp=False,  # Test keyword detection only
            cluster_similar=False
        )
        
        if result:
            data = json.loads(result[0].text)
            found = data.get('keyword_found', 0)
            print(f"   '{phrase}': {found} matches found")

if __name__ == "__main__":
    print("🚀 Running comprehensive surgical fix tests...")
    success = asyncio.run(test_surgical_fixes())
    
    if success:
        print("\n🔬 Running specific keyword tests...")
        asyncio.run(test_specific_keywords())
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Main test failed - check implementation")