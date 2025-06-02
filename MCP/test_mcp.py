from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test")

@mcp.tool()
async def hello() -> str:
    """Simple test tool."""
    return "Hello from MCP!"

if __name__ == "__main__":
    mcp.run(transport='stdio')