#!/usr/bin/env python3
"""
Test MCP tool invocation to verify async operations work correctly
"""
import json
import subprocess
import sys
import time

def send_request(proc, request):
    """Send a request and get response"""
    proc.stdin.write(json.dumps(request) + '\n')
    proc.stdin.flush()
    response = proc.stdout.readline()
    return json.loads(response)

def main():
    # Start the MCP server
    proc = subprocess.Popen(
        [sys.executable, 'reddit_scanner.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Give server time to start
        time.sleep(2)
        
        # Send initialization
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {}
            },
            "id": 1
        }
        
        print("Sending init request...")
        response = send_request(proc, init_request)
        print(f"Init response: {response.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')}")
        
        # Test a simple tool - nlp_health_check
        health_check = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "nlp_health_check",
                "arguments": {}
            },
            "id": 2
        }
        
        print("\nTesting nlp_health_check tool...")
        response = send_request(proc, health_check)
        
        if "result" in response:
            print("✅ Tool responded successfully!")
            content = response["result"]["content"]
            if content:
                print(f"Response preview: {content[0]['text'][:200]}...")
        else:
            print("❌ Tool failed:", response.get("error", "Unknown error"))
        
        # Test a more complex tool with async operations
        pain_scanner = {
            "jsonrpc": "2.0", 
            "method": "tools/call",
            "params": {
                "name": "subreddit_pain_point_scanner",
                "arguments": {
                    "subreddit_names": ["learnpython"],
                    "time_filter": "week",
                    "limit": 5,
                    "use_nlp": False,  # Start without NLP to test basic async
                    "cluster_similar": False
                }
            },
            "id": 3
        }
        
        print("\nTesting subreddit_pain_point_scanner tool (may take a moment)...")
        response = send_request(proc, pain_scanner)
        
        if "result" in response:
            print("✅ Complex tool responded successfully!")
            content = response["result"]["content"]
            if content:
                data = json.loads(content[0]["text"])
                print(f"Found {len(data.get('pain_points', []))} pain points")
        else:
            print("❌ Complex tool failed:", response.get("error", "Unknown error"))
            
    except Exception as e:
        print(f"Test error: {e}")
        stderr = proc.stderr.read()
        if stderr:
            print(f"Server errors: {stderr}")
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()