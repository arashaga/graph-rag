import os
import logging
from typing import AsyncIterator
import asyncio

# -- Your GraphRAG imports here
import pandas as pd
import tiktoken
from dotenv import load_dotenv

from graphrag.config.enums import ModelType, AuthType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch

# MCP imports
from mcp.server.fastmcp import FastMCP
import mcp.types as types

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 1. Create MCP Server with proper configuration
WORKSPACE_MCP_PORT = int(os.getenv("GRAPHRAG_MCP_PORT", 8111))
WORKSPACE_MCP_BASE_URI = os.getenv("GRAPHRAG_MCP_BASE_URI", "http://localhost")

server = FastMCP(
    name="graphrag_tool",
    server_url=f"{WORKSPACE_MCP_BASE_URI}:{WORKSPACE_MCP_PORT}/mcp",
    port=WORKSPACE_MCP_PORT
)

# 2. GraphRAG Initialization (Global State)
class GraphRAGServer:
    def __init__(self):
        self.search_engine = None
        self.initialized = False
        self.init_error = None
        self._init_lock = asyncio.Lock()

    async def initialize(self, input_dir: str, community_level: int):
        async with self._init_lock:
            if self.initialized:
                return
                
            logger.info("Starting GraphRAG server initialization...")
            try:
                load_dotenv()
                api_key = os.getenv("GRAPHRAG_API_KEY")
                llm_model = os.getenv("GRAPHRAG_LLM_MODEL")
                api_base = os.getenv("GRAPHRAG_API_BASE")

                if not all([api_key, llm_model, api_base]):
                    raise ValueError("Missing required environment variables.")

                config = LanguageModelConfig(
                    api_key=api_key,
                    auth_type=AuthType.APIKey,
                    type=ModelType.AzureOpenAIChat,
                    model=llm_model,
                    deployment_name=llm_model,
                    max_retries=20,
                    api_base=api_base,
                    api_version="2024-02-15-preview"
                )

                model = ModelManager().get_or_create_chat_model(
                    name="global_search",
                    model_type=ModelType.AzureOpenAIChat,
                    config=config,
                )

                token_encoder = tiktoken.encoding_for_model(llm_model)

                community_df = pd.read_parquet(f"{input_dir}/communities.parquet")
                entity_df = pd.read_parquet(f"{input_dir}/entities.parquet")
                report_df = pd.read_parquet(f"{input_dir}/community_reports.parquet")

                communities = read_indexer_communities(community_df, report_df)
                reports = read_indexer_reports(report_df, community_df, community_level)
                entities = read_indexer_entities(entity_df, community_df, community_level)

                context_builder = GlobalCommunityContext(
                    community_reports=reports,
                    communities=communities,
                    entities=entities,
                    token_encoder=token_encoder,
                )

                self.search_engine = GlobalSearch(
                    model=model,
                    context_builder=context_builder,
                    token_encoder=token_encoder,
                    max_data_tokens=12_000,
                    map_llm_params={"max_tokens": 1000, "temperature": 0.0, "response_format": {"type": "json_object"}},
                    reduce_llm_params={"max_tokens": 5000, "temperature": 0.0},
                    allow_general_knowledge=False,
                    json_mode=True,
                    context_builder_params={
                        "use_community_summary": False,
                        "shuffle_data": True,
                        "include_community_rank": True,
                        "min_community_rank": 0,
                        "community_rank_name": "rank",
                        "include_community_weight": True,
                        "community_weight_name": "occurrence weight",
                        "normalize_community_weight": True,
                        "max_tokens": 20_000,
                        "context_name": "Reports",
                    },
                    concurrent_coroutines=32,
                    response_type="multiple paragraphs",
                )

                self.initialized = True
                logger.info("GraphRAG server initialized successfully!")

            except Exception as e:
                self.init_error = str(e)
                logger.error(f"Initialization Error: {self.init_error}")

    async def ensure_initialized(self):
        """Ensure the server is initialized before use"""
        if not self.initialized and not self.init_error:
            await self.initialize(input_dir="./fabric", community_level=2)

# Make a global instance
graph_rag_server = GraphRAGServer()

# 3. MCP tools
# @server.tool()
# async def stream_search(query: str) -> AsyncIterator[str]:
#     """
#     Streamed search using GraphRAG.
    
#     Args:
#         query: The search query string
        
#     Yields:
#         Search results as they are generated
#     """
#     # Ensure initialization
#     await graph_rag_server.ensure_initialized()
    
#     if not graph_rag_server.initialized:
#         msg = f"Server initialization failed: {graph_rag_server.init_error}" if graph_rag_server.init_error else "Server not initialized."
#         yield msg
#         return

#     # # Check if search engine has stream_search method
#     # if hasattr(graph_rag_server.search_engine, 'stream_search'):
#     #     # If it supports streaming, use it
#     #     async for chunk in graph_rag_server.search_engine.stream_search(query):
#     #         yield chunk
#     # else:
#     #     # Otherwise, fall back to regular search
#     #     try:
#     #         result = await graph_rag_server.search_engine.search(query)
#     #         yield result.response
#     #     except Exception as e:
#     #         logger.error(f"Search error: {e}")
#     #         yield f"Error performing search: {str(e)}"
#         chunks = []
    
#     # Check if search engine has stream_search method
#     if hasattr(graph_rag_server.search_engine, 'stream_search'):
#         # If it supports streaming, use it
#         async for chunk in graph_rag_server.search_engine.stream_search(query):
#             yield chunk
#     else:
#         # Otherwise, fall back to regular search
#         try:
#             result = await graph_rag_server.search_engine.search(query)
#             yield result.response
#         except Exception as e:
#             logger.error(f"Search error: {e}")
#             yield f"Error performing search: {str(e)}"

@server.tool()
async def search(query: str) -> types.CallToolResult:
    """
    Non-streamed search using GraphRAG.
    
    Args:
        query: The search query string
        
    Returns:
        Search results
    """
    # Ensure initialization
    await graph_rag_server.ensure_initialized()
    
    if not graph_rag_server.initialized:
        msg = f"Server initialization failed: {graph_rag_server.init_error}" if graph_rag_server.init_error else "Server not initialized."
        return types.CallToolResult(
            isError=True,
            content=[types.TextContent(type="text", text=msg)]
        )

    try:
        result = await graph_rag_server.search_engine.search(query)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.response)]
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        return types.CallToolResult(
            isError=True,
            content=[types.TextContent(type="text", text=f"Error performing search: {str(e)}")]
        )

# 4. Add a health check route
@server.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint"""
    from starlette.responses import JSONResponse
    
    # Try to initialize if not already done
    await graph_rag_server.ensure_initialized()
    
    return JSONResponse({
        "status": "ok",
        "initialized": graph_rag_server.initialized,
        "error": graph_rag_server.init_error
    })

# 5. Run the server
if __name__ == "__main__":
    # Use streamable-http transport for HTTP-based MCP
    server.run(transport="streamable-http")