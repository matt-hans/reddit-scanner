#!/usr/bin/env python3
"""Minimal FastMCP test"""

from fastmcp import FastMCP
from mcp.types import TextContent

app = FastMCP("test-server")

@app.tool()
def test_tool(message: str = "hello") -> list[TextContent]:
    """A simple test tool."""
    return [TextContent(type="text", text=f"Response: {message}")]

if __name__ == "__main__":
    app.run("stdio")