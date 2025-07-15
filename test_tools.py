#!/usr/bin/env python3
"""
Simple test to verify MCP tools are working
"""
import json
import subprocess
import sys

def test_mcp_server():
    # Start the MCP server
    process = subprocess.Popen(
        [sys.executable, "reddit_scanner.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Test initialization
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            },
            "id": 1
        }
        
        # Send initialization
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Read response
        response = process.stdout.readline()
        init_result = json.loads(response)
        print("✅ Initialization successful:", init_result.get("result", {}).get("serverInfo", {}))
        
        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        }
        
        process.stdin.write(json.dumps(initialized_notification) + "\n")
        process.stdin.flush()
        
        # Test tools/list
        tools_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        
        process.stdin.write(json.dumps(tools_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        tools_result = json.loads(response)
        
        if "result" in tools_result:
            tools = tools_result["result"]["tools"]
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools[:3]:  # Show first 3 tools
                print(f"  - {tool['name']}: {tool['description'][:100]}...")
        else:
            print("❌ Tools list failed:", tools_result)
            
        # Test a simple tool call
        tool_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "subreddit_pain_point_scanner",
                "arguments": {
                    "subreddit_names": ["test"],
                    "limit": 1
                }
            },
            "id": 3
        }
        
        process.stdin.write(json.dumps(tool_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        tool_result = json.loads(response)
        
        if "result" in tool_result:
            print("✅ Tool call successful")
        else:
            print("❌ Tool call failed:", tool_result)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    test_mcp_server()