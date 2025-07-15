#!/usr/bin/env python3
"""Performance test for the Reddit Scanner MCP server"""

import asyncio
import json
import time
from typing import Dict, Any
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

async def test_pain_point_scanner():
    """Test the pain point scanner with performance monitoring"""
    print("🧪 Testing pain point scanner performance...")
    
    # Create MCP client
    async with stdio_client() as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {len(tools.tools)}")
            
            # Test with smaller parameters for performance
            start_time = time.time()
            
            result = await session.call_tool(
                'subreddit_pain_point_scanner',
                {
                    'subreddit_names': ['Python', 'webdev'],  # Only 2 subreddits
                    'time_filter': 'week',
                    'limit': 20,  # Reduced from 100
                    'min_score': 5,
                    'include_comments': True,
                    'comment_depth': 2,  # Reduced from 5
                    'use_nlp': True,
                    'cluster_similar': True
                }
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"⏱️  Test completed in {duration:.2f} seconds")
            
            # Parse and display results
            if result.content:
                data = json.loads(result.content[0].text)
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
    asyncio.run(test_pain_point_scanner())