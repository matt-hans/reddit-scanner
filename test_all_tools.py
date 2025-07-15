#!/usr/bin/env python3
"""
Comprehensive MCP test client for Reddit Scanner
Tests request/response for all 10 tools with placeholder values and robust debugging
"""
import json
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional

class MCPTestClient:
    """MCP client for testing Reddit Scanner tools"""
    
    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        
    def start(self):
        """Start the MCP server"""
        print("🚀 Starting MCP server...")
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        time.sleep(2)  # Give server time to start
        
    def stop(self):
        """Stop the MCP server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("🛑 MCP server stopped")
            
    def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send JSON-RPC request and get response"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id
        }
        
        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()
        
        # Get response
        response_str = self.process.stdout.readline()
        try:
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse response: {e}")
            print(f"Raw response: {response_str}")
            return {"error": f"Parse error: {str(e)}"}
            
    def initialize(self) -> bool:
        """Initialize MCP connection"""
        print("\n📋 Initializing MCP connection...")
        response = self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "RedditScannerTestClient",
                "version": "1.0.0"
            }
        })
        
        if "result" in response:
            print("✅ MCP initialized successfully")
            
            # Send initialized notification as required by MCP spec
            # Notifications don't have ID and don't expect response
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            notification_str = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_str)
            self.process.stdin.flush()
            
            return True
        else:
            print(f"❌ Initialization failed: {response}")
            return False
            
    def list_tools(self) -> List[str]:
        """Get list of available tools"""
        print("\n📋 Listing available tools...")
        response = self.send_request("tools/list")
        
        if "result" in response and "tools" in response["result"]:
            tools = response["result"]["tools"]
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}")
            return [tool["name"] for tool in tools]
        else:
            print(f"❌ Failed to list tools: {response}")
            return []
            
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool with arguments"""
        print(f"\n🔧 Calling tool: {tool_name}")
        print(f"📥 Arguments: {json.dumps(arguments, indent=2)}")
        
        response = self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        return response

def print_tool_response(tool_name: str, response: Dict[str, Any], start_time: float):
    """Pretty print tool response with debugging info"""
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"🔧 TOOL: {tool_name}")
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"{'='*80}")
    
    if "error" in response:
        print(f"❌ ERROR: {response['error']}")
        if "data" in response["error"]:
            print(f"📊 Error Data: {json.dumps(response['error']['data'], indent=2)}")
    elif "result" in response:
        result = response["result"]
        if "content" in result and isinstance(result["content"], list):
            for content in result["content"]:
                if content.get("type") == "text":
                    try:
                        # Try to parse as JSON for pretty printing
                        parsed = json.loads(content["text"])
                        print(f"✅ SUCCESS - Response Data:")
                        print(json.dumps(parsed, indent=2))
                        
                        # Print summary statistics
                        if isinstance(parsed, dict):
                            print(f"\n📊 Response Summary:")
                            for key, value in parsed.items():
                                if isinstance(value, list):
                                    print(f"  - {key}: {len(value)} items")
                                elif isinstance(value, (int, float)):
                                    print(f"  - {key}: {value}")
                                elif isinstance(value, dict):
                                    print(f"  - {key}: {len(value)} entries")
                    except json.JSONDecodeError:
                        print(f"✅ SUCCESS - Raw Response:")
                        print(content["text"])
        else:
            print(f"✅ SUCCESS - Raw Result:")
            print(json.dumps(result, indent=2))
    else:
        print(f"⚠️  Unexpected response format:")
        print(json.dumps(response, indent=2))

def test_all_tools():
    """Test all Reddit Scanner tools with placeholder values"""
    
    # Create test client
    client = MCPTestClient(["python", "reddit_scanner.py"])
    
    try:
        # Start server and initialize
        client.start()
        if not client.initialize():
            print("❌ Failed to initialize MCP connection")
            return
            
        # List available tools
        tools = client.list_tools()
        if not tools:
            print("❌ No tools found")
            return
            
        print(f"\n{'='*80}")
        print("🧪 STARTING COMPREHENSIVE TOOL TESTS")
        print(f"{'='*80}")
        
        # Test configurations for each tool
        test_configs = [
            {
                "name": "subreddit_pain_point_scanner",
                "args": {
                    "subreddit_names": ["python", "learnprogramming", "webdev"],
                    "time_filter": "week",
                    "limit": 10,
                    "pain_keywords": ["frustrated", "annoying", "wish there was"],
                    "min_score": 5,
                    "include_comments": True,
                    "comment_depth": 3
                },
                "description": "Scanning for user pain points and frustrations"
            },
            {
                "name": "solution_request_tracker",
                "args": {
                    "subreddit_names": ["software", "productivity"],
                    "request_patterns": ["looking for.*app", "any.*tools", "recommend.*software"],
                    "exclude_solved": True,
                    "min_engagement": 5,
                    "time_window": "month",
                    "category_keywords": {
                        "automation": ["automate", "workflow"],
                        "productivity": ["efficiency", "time", "organize"]
                    }
                },
                "description": "Tracking software recommendation requests"
            },
            {
                "name": "user_workflow_analyzer",
                "args": {
                    "subreddit_names": ["productivity", "getdisciplined"],
                    "workflow_indicators": ["process", "steps", "routine", "manually"],
                    "min_steps": 3,
                    "efficiency_keywords": ["time-consuming", "repetitive", "tedious"]
                },
                "description": "Analyzing workflows for automation opportunities"
            },
            {
                "name": "competitor_mention_monitor",
                "args": {
                    "competitor_names": ["notion", "obsidian", "todoist"],
                    "sentiment_threshold": 0.0,
                    "limitation_keywords": ["but", "however", "missing", "wish it had"]
                },
                "description": "Monitoring competitor mentions and limitations"
            },
            {
                "name": "subreddit_engagement_analyzer",
                "args": {
                    "post_ids": ["1abc2d3", "4efg5h6", "7ijk8l9"],  # Valid format placeholder IDs
                    "engagement_metrics": ["upvote_ratio", "comment_rate"],
                    "time_decay": True
                },
                "description": "Analyzing post engagement patterns"
            },
            {
                "name": "niche_community_discoverer",
                "args": {
                    "seed_subreddits": ["startups", "solopreneur"],
                    "min_subscribers": 5000,
                    "max_subscribers": 50000,
                    "activity_threshold": 0.3,
                    "related_depth": 1
                },
                "description": "Discovering niche communities with unmet needs"
            },
            {
                "name": "temporal_trend_analyzer",
                "args": {
                    "subreddit_names": ["technology", "programming"],
                    "time_periods": ["day", "week", "month"],
                    "trend_keywords": ["AI", "automation", "no-code", "productivity"],
                    "growth_threshold": 0.5
                },
                "description": "Tracking emerging trends over time"
            },
            {
                "name": "user_persona_extractor",
                "args": {
                    "user_sample_size": 10,
                    "activity_depth": 20,
                    "need_categories": ["productivity", "automation", "organization"]
                },
                "description": "Building user personas from activity patterns"
            },
            {
                "name": "idea_validation_scorer",
                "args": {
                    "pain_point_data": {
                        "task_automation": {"count": 150, "score": 85, "competition": 3},
                        "note_taking": {"count": 200, "score": 70, "competition": 8},
                        "time_tracking": {"count": 100, "score": 90, "competition": 5}
                    },
                    "market_size_weight": 0.3,
                    "severity_weight": 0.3,
                    "competition_weight": 0.2,
                    "technical_feasibility": {
                        "task_automation": 0.8,
                        "note_taking": 0.6,
                        "time_tracking": 0.9
                    }
                },
                "description": "Scoring software opportunity ideas"
            }
        ]
        
        # Test each tool
        successful_tests = 0
        failed_tests = 0
        
        for config in test_configs:
            print(f"\n{'='*80}")
            print(f"🧪 TEST: {config['name']}")
            print(f"📝 Description: {config['description']}")
            print(f"{'='*80}")
            
            start_time = time.time()
            try:
                response = client.call_tool(config["name"], config["args"])
                print_tool_response(config["name"], response, start_time)
                
                if "error" not in response:
                    successful_tests += 1
                else:
                    failed_tests += 1
                    
            except Exception as e:
                print(f"❌ Exception during test: {str(e)}")
                failed_tests += 1
                
            # Add delay between tests to avoid rate limiting
            time.sleep(1)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("📊 TEST SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successful tests: {successful_tests}")
        print(f"❌ Failed tests: {failed_tests}")
        print(f"📈 Success rate: {(successful_tests/(successful_tests+failed_tests)*100):.1f}%")
        print(f"🔧 Total tools tested: {len(test_configs)}")
        
        # Check stderr for any warnings or errors
        stderr_output = client.process.stderr.read()
        if stderr_output:
            print(f"\n⚠️  Server stderr output:")
            print(stderr_output)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
    finally:
        client.stop()

if __name__ == "__main__":
    print("🧪 Reddit Scanner MCP Tool Tester")
    print("📝 This script tests all 10 tools with placeholder values")
    print("🔍 Provides detailed debugging output for each response")
    print("-" * 80)
    
    # Check for .env file
    import os
    if not os.path.exists(".env"):
        print("\n⚠️  WARNING: .env file not found!")
        print("Please create a .env file with your Reddit API credentials:")
        print("  REDDIT_CLIENT_ID=your_client_id")
        print("  REDDIT_CLIENT_SECRET=your_client_secret")
        print("  REDDIT_USER_AGENT=MCP Reddit Analyzer 1.0")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    test_all_tools()