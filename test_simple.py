#!/usr/bin/env python3
"""Simple test with same structure as reddit_scanner"""

from fastmcp import FastMCP
from mcp.types import TextContent
from typing import List
import json

mcp = FastMCP("test-server")

@mcp.tool()
async def simple_tool(message: str = "hello") -> List[TextContent]:
    """A simple test tool."""
    result = {"message": message, "status": "ok"}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

if __name__ == "__main__":
    mcp.run("stdio")