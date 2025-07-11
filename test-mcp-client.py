#!/usr/bin/env python3
"""
Simple MCP client to test the Reddit scanner server
"""
import json
import sys

# Send an initialization request
init_request = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "protocolVersion": "0.1.0",
        "capabilities": {}
    },
    "id": 1
}

print(json.dumps(init_request))
sys.stdout.flush()

# Wait for response
response = input()
print(f"Received: {response}", file=sys.stderr)

# List available tools
list_tools = {
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 2
}

print(json.dumps(list_tools))
sys.stdout.flush()

# Wait for response
response = input()
print(f"Available tools: {response}", file=sys.stderr)