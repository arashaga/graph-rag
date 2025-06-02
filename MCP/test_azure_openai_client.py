import argparse
import asyncio
import json
import os
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import aiohttp

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    """MCP Client for interacting with an MCP Streamable HTTP server using Azure OpenAI"""

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Initialize Azure OpenAI client
        self.azure_openai = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("GRAPHRAG_API_BASE"),
            api_key=os.getenv("GRAPHRAG_API_KEY"),
            api_version="2025-02-01-preview"  # Latest API version with function calling support
        )
          # Model deployment name - set this to your deployment name
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

    async def test_server_health(self, base_url: str) -> bool:
        """Test if the MCP server is responding to HTTP requests"""
        try:
            # Use /health endpoint for health check
            health_url = base_url.rstrip("/") + "/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url) as response:
                    print(f"Server health check - Status: {response.status} at {health_url}")
                    return response.status < 500
        except Exception as e:
            print(f"Server health check failed: {str(e)}")
            return False

    async def connect_to_streamable_http_server(
        self, server_url: str, headers: Optional[dict] = None
    ):
        """Connect to an MCP server running with HTTP Streamable transport"""
        print(f"Attempting to connect to: {server_url}")
        # Only try the canonical /mcp/ endpoint
        try:
            self._streams_context = streamablehttp_client(
                url=server_url,
                headers=headers or {},
            )
            read_stream, write_stream, _ = await self._streams_context.__aenter__()
            self._session_context = ClientSession(read_stream, write_stream)
            self.session: ClientSession = await self._session_context.__aenter__()
            await self.session.initialize()
            print(f"Successfully connected to: {server_url}")
            return
        except Exception as e:
            print(f"Failed to connect to {server_url}: {str(e)}")
            raise

    def _convert_mcp_tools_to_openai_format(self, mcp_tools) -> List[Dict[str, Any]]:
        """Convert MCP tools to Azure OpenAI function calling format"""
        openai_tools = []
        
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools

    async def process_query(self, query: str) -> str:
        """Process a query using Azure OpenAI and available MCP tools"""
        try:
            # Get available tools from MCP server
            response = await self.session.list_tools()
            available_tools = self._convert_mcp_tools_to_openai_format(response.tools)
            
            print(f"Available tools: {[tool['function']['name'] for tool in available_tools]}")

            messages = [{"role": "user", "content": query}]

            # Initial Azure OpenAI API call with tools
            response = await self.azure_openai.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                tools=available_tools if available_tools else None,
                tool_choice="auto" if available_tools else None,
                max_tokens=2000,
                temperature=0.1
            )

            message = response.choices[0].message
            
            # Handle tool calls if any
            if message.tool_calls:
                # Add the assistant's response to the conversation
                messages.append({
                    "role": "assistant", 
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    print(f"Calling tool: {tool_name} with args: {tool_args}")
                    
                    try:
                        # Call the MCP tool
                        result = await self.session.call_tool(tool_name, tool_args)
                        tool_result = result.content[0].text if result.content else "No result"
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                        
                    except Exception as e:
                        print(f"Error calling tool {tool_name}: {str(e)}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {str(e)}"
                        })

                # Get final response from Azure OpenAI with tool results
                final_response = await self.azure_openai.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.1
                )
                
                return final_response.choices[0].message.content

            else:
                # No tool calls, return the response directly
                return message.content or "No response generated"

        except Exception as e:
            return f"Error processing query: {str(e)}"

    async def test_streaming_search(self, query: str) -> str:
        """Test the streaming search functionality specifically"""
        try:
            print("Testing streaming search...")
            
            # Call the stream_search tool directly
            result = await self.session.call_tool("stream_search", {"query": query})
            
            if result.content:
                return result.content[0].text
            else:
                return "No streaming result received"
                
        except Exception as e:
            return f"Error in streaming search: {str(e)}"

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Azure OpenAI Client Started!")
        print("Available commands:")
        print("  - Type any query for normal GraphRAG search")
        print("  - Type 'stream <query>' for streaming search") 
        print("  - Type 'tools' to list available tools")
        print("  - Type 'quit' to exit")

        while True:
            try:
                user_input = input("\nQuery: ").strip()

                if user_input.lower() == "quit":
                    break
                
                elif user_input.lower() == "tools":
                    # List available tools
                    response = await self.session.list_tools()
                    print("\nAvailable tools:")
                    for tool in response.tools:
                        print(f"  - {tool.name}: {tool.description}")
                    continue
                
                elif user_input.lower().startswith("stream "):
                    # Test streaming search directly
                    query = user_input[7:]  # Remove "stream " prefix
                    response = await self.test_streaming_search(query)
                    print(f"\nStreaming Response:\n{response}")
                    
                else:
                    # Regular query processing
                    response = await self.process_query(user_input)
                    print(f"\nResponse:\n{response}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Properly clean up the session and streams"""
        try:
            if hasattr(self, '_session_context') and self._session_context:
                await self._session_context.__aexit__(None, None, None)
            if hasattr(self, '_streams_context') and self._streams_context:
                await self._streams_context.__aexit__(None, None, None)
        except Exception as e:
            print(f"Error during cleanup: {e}")


async def main():
    """Main function to run the MCP client"""
    parser = argparse.ArgumentParser(description="Run MCP Azure OpenAI Client with Streamable HTTP")
    parser.add_argument(
        "--mcp-localhost-port", 
        type=int, 
        default=8111, 
        help="Localhost port for MCP server (default: 8111)"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test connection and list tools only"    )
    args = parser.parse_args()
    
    # Validate environment variables
    required_env_vars = ["GRAPHRAG_API_BASE", "GRAPHRAG_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment")
        return

    client = MCPClient()

    try:
        # Always use /mcp (no trailing slash) for MCP server
        server_url = f"http://localhost:{args.mcp_localhost_port}/mcp"
        print(f"Testing server health at http://localhost:{args.mcp_localhost_port}/health")
        
        # Test server health first
        if not await client.test_server_health(f"http://localhost:{args.mcp_localhost_port}"):
            print("Server health check failed. Please ensure the MCP server is running and accessible at /mcp.")
            return
        
        print(f"Connecting to MCP server at {server_url}")
        await client.connect_to_streamable_http_server(server_url)
        
        if args.test_connection:
            # Just test connection and list tools
            response = await client.session.list_tools()
            print(f"Successfully connected! Available tools: {[tool.name for tool in response.tools]}")
        else:
            # Run interactive chat loop
            await client.chat_loop()
            
    except Exception as e:
        print(f"Failed to connect or run: {str(e)}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
