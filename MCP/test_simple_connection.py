import asyncio
import aiohttp

async def test_mcp_connection():
    """Simple test to check if MCP server is responding correctly"""
    url = "http://localhost:8111/mcp"
    
    # Test basic HTTP connection
    try:
        async with aiohttp.ClientSession() as session:
            # Try different HTTP methods
            print("Testing GET request...")
            async with session.get(url) as response:
                print(f"GET Response: {response.status}")
                text = await response.text()
                print(f"GET Body: {text[:200]}...")
            
            print("\nTesting POST request...")
            headers = {'Content-Type': 'application/json'}
            data = '{"jsonrpc": "2.0", "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}}, "id": 1}'
            
            async with session.post(url, headers=headers, data=data) as response:
                print(f"POST Response: {response.status}")
                text = await response.text()
                print(f"POST Body: {text[:500]}...")
                
    except Exception as e:
        print(f"HTTP test failed: {e}")

    # Test if server accepts SSE/EventSource connections
    try:
        print("\nTesting SSE connection...")
        headers = {
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                print(f"SSE Response: {response.status}")
                print(f"Content-Type: {response.headers.get('Content-Type')}")
                
                # Read a few chunks
                async for data in response.content:
                    print(f"SSE Data: {data}")
                    break
                    
    except Exception as e:
        print(f"SSE test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
