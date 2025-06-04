#!/usr/bin/env python3
"""
GraphRAG MCP client for testing the GraphRAG server.

This client connects to the GraphRAG MCP server using streamable HTTP transport.
"""

import asyncio
import os
import sys
from datetime import timedelta
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class GraphRAGClient:
    """GraphRAG MCP client."""

    def __init__(self, server_url: str = "http://localhost:8111/mcp"):
        self.server_url = server_url
        self.session: ClientSession | None = None

    async def connect(self):
        """Connect to the MCP server."""
        print(f"🔗 Attempting to connect to {self.server_url}...")

        try:
            # Create transport without auth (GraphRAG server doesn't use OAuth)
            print("📡 Opening StreamableHTTP transport connection...")
            async with streamablehttp_client(
                url=self.server_url,
                auth=None,  # No auth required for GraphRAG server
                timeout=timedelta(seconds=60),
            ) as (read_stream, write_stream, get_session_id):
                await self._run_session(read_stream, write_stream, get_session_id)

        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            import traceback
            traceback.print_exc()

    async def _run_session(self, read_stream, write_stream, get_session_id):
        """Run the MCP session with the given streams."""
        print("🤝 Initializing MCP session...")
        async with ClientSession(read_stream, write_stream) as session:
            self.session = session
            print("⚡ Starting session initialization...")
            await session.initialize()
            print("✨ Session initialization complete!")

            print(f"\n✅ Connected to GraphRAG MCP server at {self.server_url}")
            if get_session_id:
                session_id = get_session_id()
                if session_id:
                    print(f"Session ID: {session_id}")

            # Run interactive loop
            await self.interactive_loop()

    async def list_tools(self):
        """List available tools from the server."""
        if not self.session:
            print("❌ Not connected to server")
            return

        try:
            result = await self.session.list_tools()
            if hasattr(result, "tools") and result.tools:
                print("\n📋 Available tools:")
                for i, tool in enumerate(result.tools, 1):
                    print(f"{i}. {tool.name}")
                    if tool.description:
                        print(f"   Description: {tool.description}")
                    if hasattr(tool, "inputSchema") and tool.inputSchema:
                        print(f"   Input schema: {tool.inputSchema}")
                    print()
            else:
                print("No tools available")
        except Exception as e:
            print(f"❌ Failed to list tools: {e}")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None):
        """Call a specific tool."""
        if not self.session:
            print("❌ Not connected to server")
            return

        try:
            print(f"\n🔧 Calling tool '{tool_name}' with arguments: {arguments}")
            result = await self.session.call_tool(tool_name, arguments or {})
            
            if hasattr(result, "content"):
                for content in result.content:
                    if hasattr(content, "type") and content.type == "text":
                        print(f"\n📝 Result:\n{content.text}")
                    else:
                        print(f"\n📝 Result:\n{content}")
            else:
                print(f"\n📝 Result:\n{result}")
        except Exception as e:
            print(f"❌ Failed to call tool '{tool_name}': {e}")
            import traceback
            traceback.print_exc()

    async def test_graphrag_tools(self):
        """Test GraphRAG-specific tools."""
        print("\n🧪 Running GraphRAG tool tests...")
        
        # List tools first
        await self.list_tools()
        
        # Test non-streaming search
        print("\n📌 Testing non-streaming search...")
        test_query = "What are the main themes and topics in the data?"
        await self.call_tool("search", {"query": test_query})
        
        # Test streaming search
        print("\n📌 Testing streaming search...")
        print("Note: Streaming results will be displayed as they arrive")
        await self.call_tool("stream_search", {"query": test_query})

    async def interactive_loop(self):
        """Run interactive command loop."""
        print("\n🎯 GraphRAG Interactive MCP Client")
        print("Commands:")
        print("  list - List available tools")
        print("  search <query> - Run a non-streaming search")
        print("  stream <query> - Run a streaming search")
        print("  test - Run automated tests")
        print("  quit - Exit the client")
        print()

        while True:
            try:
                command = input("graphrag> ").strip()

                if not command:
                    continue

                if command == "quit":
                    break

                elif command == "list":
                    await self.list_tools()

                elif command == "test":
                    await self.test_graphrag_tools()

                elif command.startswith("search "):
                    query = command[7:].strip()
                    if query:
                        await self.call_tool("search", {"query": query})
                    else:
                        print("❌ Please provide a search query")

                elif command.startswith("stream "):
                    query = command[7:].strip()
                    if query:
                        await self.call_tool("stream_search", {"query": query})
                    else:
                        print("❌ Please provide a search query")

                else:
                    print("❌ Unknown command. Try 'list', 'search <query>', 'stream <query>', 'test', or 'quit'")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()


async def main():
    """Main entry point."""
    # Check for custom server URL
    server_port = os.getenv("GRAPHRAG_MCP_PORT", "8111")
    server_url = f"http://localhost:{server_port}/mcp"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        server_url = sys.argv[1]

    print("🚀 GraphRAG MCP Client")
    print(f"Connecting to: {server_url}")

    # Start connection
    client = GraphRAGClient(server_url)
    await client.connect()


if __name__ == "__main__":
    asyncio.run(main())