import os
import asyncio
from typing import Any, Optional, AsyncGenerator
import pandas as pd
import tiktoken
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

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
from graphrag.callbacks.query_callbacks import QueryCallbacks

# Initialize FastMCP server with stateless HTTP support for streaming
mcp = FastMCP("graphrag", stateless_http=False)

class GraphRAGServer:
    """GraphRAG MCP Server for global search functionality."""
    
    def __init__(self):
        self.search_engine = None
        self.context_builder = None
        self.initialized = False
        self.initialization_error = None        
    async def initialize(self, input_dir: str = "./fabric", community_level: int = 2):
        """Initialize the GraphRAG search engine with data from the specified directory."""
        try:
            # Load environment variables
            load_dotenv()
            
            # Get API configuration from environment
            api_key = os.getenv("GRAPHRAG_API_KEY")
            llm_model = os.getenv("GRAPHRAG_LLM_MODEL")
            api_base = os.getenv("GRAPHRAG_API_BASE")
            
            if not all([api_key, llm_model, api_base]):
                raise ValueError("Missing required environment variables: GRAPHRAG_API_KEY, GRAPHRAG_LLM_MODEL, GRAPHRAG_API_BASE")
            
            # Configure language model
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
            
            # Load data files
            community_df = pd.read_parquet(f"{input_dir}/communities.parquet")
            entity_df = pd.read_parquet(f"{input_dir}/entities.parquet")
            report_df = pd.read_parquet(f"{input_dir}/community_reports.parquet")
            
            # Read and process data
            communities = read_indexer_communities(community_df, report_df)
            reports = read_indexer_reports(report_df, community_df, community_level)
            entities = read_indexer_entities(entity_df, community_df, community_level)
            
            # Initialize context builder
            self.context_builder = GlobalCommunityContext(
                community_reports=reports,
                communities=communities,
                entities=entities,
                token_encoder=token_encoder,
            )
            
            # Context builder parameters
            context_builder_params = {
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
            }
            
            # LLM parameters
            map_llm_params = {
                "max_tokens": 1000,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            
            reduce_llm_params = {
                "max_tokens": 5000,
                "temperature": 0.0,
            }
            
            # Initialize search engine
            self.search_engine = GlobalSearch(
                model=model,
                context_builder=self.context_builder,
                token_encoder=token_encoder,
                max_data_tokens=12_000,
                map_llm_params=map_llm_params,
                reduce_llm_params=reduce_llm_params,
                allow_general_knowledge=False,
                json_mode=True,
                context_builder_params=context_builder_params,
                concurrent_coroutines=32,
                response_type="multiple paragraphs",
            )
            
            self.initialized = True
            print(f"GraphRAG initialized successfully with {len(reports)} reports, {len(communities)} communities, and {len(entities)} entities.")
            
        except Exception as e:
            self.initialization_error = str(e)
            print(f"Failed to initialize GraphRAG: {str(e)}")

# Global instance
graphrag_server = GraphRAGServer()

@mcp.tool()
async def search_graphrag(query: str, include_context_data: bool = False) -> str:
    """Perform a global search using GraphRAG.
    
    Args:
        query: The search query to execute
        include_context_data: Whether to include detailed context data in the response
    """
    if not graphrag_server.initialized:
        if graphrag_server.initialization_error:
            return f"GraphRAG initialization failed: {graphrag_server.initialization_error}"
        return "GraphRAG is not initialized."
    
    try:
        # Perform the search
        result = await graphrag_server.search_engine.search(query)
        
        # Format response
        response_parts = [
            f"Query: {query}",
            f"Response: {result.response}",
            f"LLM calls: {result.llm_calls}",
            f"Prompt tokens: {result.prompt_tokens}",
            f"Output tokens: {result.output_tokens}"
        ]
        
        if include_context_data and hasattr(result, 'context_data'):
            response_parts.append(f"Context data available: {list(result.context_data.keys())}")
        
        return "\n\n".join(response_parts)
        
    except Exception as e:
        return f"Search failed: {str(e)}"

@mcp.tool()
async def stream_search(query: str) -> AsyncGenerator[str, None]:
    """Perform a streaming search using GraphRAG with real-time chunk yielding.
    
    Args:
        query: The search query to execute
    
    Yields:
        Chunks of the search response as they become available
    """
    if not graphrag_server.initialized:
        if graphrag_server.initialization_error:
            yield f"GraphRAG initialization failed: {graphrag_server.initialization_error}"
            return
        yield "GraphRAG is not initialized."
        return
    
    try:
        # Check if the search engine has a stream_search method
        if hasattr(graphrag_server.search_engine, 'stream_search'):
            # Use the native streaming method with real-time yielding
            async for chunk in graphrag_server.search_engine.stream_search(query):
                yield chunk
        else:
            # Fallback to regular search if streaming is not available
            result = await graphrag_server.search_engine.search(query)
            yield result.response
        
    except Exception as e:
        yield f"Search failed: {str(e)}"

async def initialize_server():
    """Initialize the GraphRAG server on startup."""
    # You can modify these parameters as needed
    input_dir = "./fabric"  # Change this to your desired input directory
    community_level = 2     # Change this to your desired community level
    
    await graphrag_server.initialize(input_dir, community_level)

# Create FastAPI app with FastMCP integration
def create_app():
    """Create FastAPI application with MCP StreamableHTTP support."""
    # Create a FastAPI app with lifespan that initializes GraphRAG
    async def lifespan(app: FastAPI):
        # Initialize GraphRAG on startup
        await initialize_server()
        yield
        # Cleanup code would go here if needed
    
    app = FastAPI(lifespan=lifespan)

    # Health check endpoint
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Mount the MCP StreamableHTTP app at /mcp
    app.mount("/mcp", mcp.streamable_http_app())
    
    return app

if __name__ == "__main__":
    # Create the FastAPI app
    app = create_app()
    
    # Run with uvicorn for StreamableHTTP support
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8111,
        log_level="info"
    )