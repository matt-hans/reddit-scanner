#!/usr/bin/env python3
"""Test MCP protocol steps"""

import json
import subprocess
import sys
import time

def send_and_receive(process, message):
    """Send a message and get response"""
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()
    
    # Read until we get a complete JSON response
    buffer = ""
    while True:
        char = process.stdout.read(1)
        if not char:
            break
        buffer += char
        if char == '\n':
            line = buffer.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            buffer = ""
    return None

def test_protocol():
    # Start server
    process = subprocess.Popen(
        [sys.executable, "reddit_scanner.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(1)  # Give server time to start
    
    try:
        # Step 1: Initialize
        init_msg = {
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
        
        response = send_and_receive(process, init_msg)
        print("Init response:", response)
        
        # Step 2: Send initialized notification
        initialized_msg = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        }
        
        process.stdin.write(json.dumps(initialized_msg) + "\n")
        process.stdin.flush()
        
        # Step 3: List tools
        tools_msg = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        
        response = send_and_receive(process, tools_msg)
        print("Tools response:", response)
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Stderr: {process.stderr.read()}")
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    test_protocol()