import asyncio
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

    async def call_tool(self, tool_name: str, **kwargs):
        # Find the tool schema for input hints (optional, can skip)
        response = await self.session.list_tools()
        tool = next((t for t in response.tools if t.name == tool_name), None)
        if tool is None:
            print(f"No such tool: {tool_name}")
            return

        print(f"Calling tool '{tool_name}' with arguments: {kwargs}")
        result = await self.session.call_tool(tool_name, kwargs)
        print("\nResult:")
        print(result.content)

    async def chat_loop(self):
        print("\nMCP Client Started! Type 'help' for tool list, or 'quit' to exit.")
        response = await self.session.list_tools()
        tools = response.tools

        while True:
            try:
                cmd = input("\nEnter tool name or 'help' or 'quit': ").strip()
                if cmd.lower() == "quit":
                    break
                if cmd.lower() == "help":
                    print("\nAvailable tools:")
                    for tool in tools:
                        print(f"- {tool.name}: {tool.description}")
                    continue

                tool = next((t for t in tools if t.name == cmd), None)
                if tool is None:
                    print("Unknown tool. Type 'help' to see available tools.")
                    continue

                # Build arguments interactively
                kwargs = {}
                input_schema = tool.inputSchema or {}
                for k, v in (input_schema.get("properties") or {}).items():
                    default = v.get("default", "")
                    prompt = f"Enter value for '{k}' ({v.get('type','str')})"
                    if default:
                        prompt += f" [default={default}]"
                    prompt += ": "
                    val = input(prompt).strip()
                    if not val and default != "":
                        val = default
                    # Try to cast types for int/bool/float
                    if v.get("type") == "integer":
                        val = int(val)
                    elif v.get("type") == "boolean":
                        val = val.lower() in ("1", "true", "yes")
                    elif v.get("type") == "number":
                        val = float(val)
                    kwargs[k] = val

                await self.call_tool(cmd, **kwargs)
            except Exception as e:
                print(f"Error: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
