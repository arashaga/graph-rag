# GraphRAG MCP StreamableHTTP Server

This is a Model Context Protocol (MCP) server that provides GraphRAG global search functionality using the StreamableHTTP transport with uvicorn for true streaming capabilities.

## Features

- **Auto-initialization**: GraphRAG initializes automatically when the server starts
- **StreamableHTTP Transport**: Uses uvicorn and FastAPI for HTTP-based MCP communication
- **True Streaming**: The `stream_search` tool provides real-time streaming of search results
- **Two Search Tools**:
  - `search_graphrag`: Standard search with detailed metadata
  - `stream_search`: Streaming search that yields chunks as they arrive

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Create a `.env` file with:
   ```
   GRAPHRAG_API_KEY=your_api_key
   GRAPHRAG_LLM_MODEL=your_model_name
   GRAPHRAG_API_BASE=your_api_base_url
   ```

3. **Prepare Data Directory**:
   Ensure you have a `./fabric` directory (or modify the path in the code) containing:
   - `communities.parquet`
   - `entities.parquet`
   - `community_reports.parquet`

## Running the Server

### StreamableHTTP Mode (Recommended)
```bash
python graphrag_global_search_server.py
```

This will start the server at `http://127.0.0.1:8000/mcp` with:
- Automatic GraphRAG initialization on startup
- StreamableHTTP transport for better performance
- True streaming capabilities

## Testing

### Prerequisites

#### Option 1: Azure OpenAI Client (Recommended)

For the Azure OpenAI test client:

1. Install additional dependencies:
```bash
pip install openai python-dotenv
```

2. Set up your Azure OpenAI configuration in a `.env` file:
```
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

#### Option 2: Anthropic Claude Client

For the Anthropic Claude test client:

1. Install additional dependencies:
```bash
pip install anthropic python-dotenv
```

2. Set up your Anthropic API key in a `.env` file:
```
ANTHROPIC_API_KEY=your_api_key_here
```

### Running Tests

1. **Start the server:**
```bash
python graphrag_global_search_server.py
```
The server will run on `http://127.0.0.1:8000` by default.

2. **Test with Azure OpenAI client (recommended):**
```bash
python test_azure_openai_client.py
```
This will start an interactive chat session where you can ask questions. The client will:
- Connect to your MCP server using StreamableHTTP
- Use Azure OpenAI with function calling to process your queries
- Automatically call GraphRAG tools when appropriate
- Provide conversational responses with proper tool integration

Available commands in the Azure OpenAI client:
- Type any query for normal GraphRAG search with AI assistance
- Type `stream <query>` for direct streaming search testing
- Type `tools` to list available MCP tools
- Type `quit` to exit

Example interaction:
```
Query: What are the main themes in the data?
Calling tool: search_graphrag with args: {'query': 'What are the main themes in the data?', 'include_context_data': False}
Based on the GraphRAG analysis, the main themes in the data include...
```

3. **Alternative: Test with Anthropic Claude client:**
```bash
python test_streamable_http_client.py
```
This provides the same functionality but uses Anthropic's Claude model instead of Azure OpenAI.

4. **Alternative: Simple MCP test (basic functionality):**
```bash
python test_mcp.py
```
This provides a basic test of the MCP protocol without AI model integration.

5. **Test connection only:**
```bash
python test_azure_openai_client.py --test-connection
```
This will test the connection to the MCP server and list available tools without starting the interactive chat.

6. **Use with Claude Desktop via mcp-remote:**
```bash
npx -y mcp-remote http://127.0.0.1:8000/mcp
```

### Client Options

#### Azure OpenAI Client
```bash
python test_azure_openai_client.py --mcp-localhost-port 8000 --test-connection
```

#### Anthropic Claude Client
```bash
python test_streamable_http_client.py --mcp-localhost-port 8000
```

Type `quit` to exit the interactive session.

## Available Tools

### `search_graphrag`
Performs a comprehensive GraphRAG global search with detailed metadata.

**Parameters**:
- `query` (string): The search query
- `include_context_data` (boolean): Whether to include context data in response

**Returns**: Detailed search results with token usage and metadata

### `stream_search`
Performs a streaming GraphRAG search that yields results in real-time chunks.

**Parameters**:
- `query` (string): The search query

**Returns**: Streaming response chunks as they become available

## Architecture

The server uses:
- **FastMCP**: For MCP protocol implementation
- **FastAPI**: For HTTP server functionality
- **Uvicorn**: For ASGI server with streaming support
- **GraphRAG**: For global search capabilities

The server automatically initializes GraphRAG during startup using the `lifespan` event, ensuring all search tools are ready when the server becomes available.

## Configuration

You can modify the initialization parameters in the `initialize_server()` function:
- `input_dir`: Path to your GraphRAG data directory (default: "./fabric")
- `community_level`: Community level for search (default: 2)

## Error Handling

The server includes comprehensive error handling:
- Initialization errors are captured and reported
- Both tools check initialization status before processing
- Graceful error messages for missing dependencies or configuration

## Streaming Details

The `stream_search` tool uses GraphRAG's native `stream_search` method when available, providing true streaming where results are yielded as individual chunks rather than being collected and returned all at once. This is particularly useful for long searches where you want to see results as they're generated.

## Dependencies

- `graphrag`: GraphRAG library for global search
- `fastmcp`: FastMCP for MCP server implementation
- `fastapi`: Web framework for HTTP endpoints
- `uvicorn`: ASGI server for production deployment
- `pandas`: Data manipulation
- `tiktoken`: Token encoding
- `python-dotenv`: Environment variable management
