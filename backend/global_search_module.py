import os
import asyncio
import logging
import uuid
import warnings
import time
import multiprocessing
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
from semantic_kernel.contents import ChatHistory

from graphrag.config.enums import ModelType, AuthType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_communities,
)
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch

# Load environment variables
load_dotenv()

# Configuration class
class GraphRagBaseConfig:
    def __init__(self):
        required_env = ["GRAPHRAG_API_KEY", "GRAPHRAG_LLM_MODEL", 
                        "GRAPHRAG_EMBEDDING_MODEL", "GRAPHRAG_API_BASE",
                        "GRAPHRAG_DATA_DIR"]
        missing = [key for key in required_env if os.getenv(key) is None]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

        self.api_key = os.getenv("GRAPHRAG_API_KEY")
        self.llm_model = os.getenv("GRAPHRAG_LLM_MODEL")
        self.embedding_model = os.getenv("GRAPHRAG_EMBEDDING_MODEL")
        self.api_base = os.getenv("GRAPHRAG_API_BASE")
        self.data_dir = os.getenv("GRAPHRAG_DATA_DIR")

        self.token_encoder = tiktoken.encoding_for_model(self.llm_model)

config = GraphRagBaseConfig()

# Global data storage
_global_data = {}

def load_graphrag_data():
    global _global_data
    if _global_data:
        return _global_data

    data_dir = config.data_dir

    entity_df = pd.read_parquet(f"{data_dir}/entities.parquet")
    community_df = pd.read_parquet(f"{data_dir}/communities.parquet")
    relationship_df = pd.read_parquet(f"{data_dir}/relationships.parquet")
    report_df = pd.read_parquet(f"{data_dir}/community_reports.parquet")
    text_unit_df = pd.read_parquet(f"{data_dir}/text_units.parquet")

    entities = read_indexer_entities(entity_df, community_df, 2)
    relationships = read_indexer_relationships(relationship_df)
    reports = read_indexer_reports(report_df, community_df, 2)
    text_units = read_indexer_text_units(text_unit_df)
    communities = read_indexer_communities(community_df, report_df)

    _global_data = {
        'entities': entities,
        'relationships': relationships,
        'reports': reports,
        'text_units': text_units,
        'communities': communities,
        'token_encoder': config.token_encoder
    }
    return _global_data

async def execute_global_search_task(query: str):
    warnings.filterwarnings("ignore", message=".*AsyncLimiter.*")

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(1) as pool:
        result = pool.apply_async(_run_global_search_process, (query,))
        return result.get(timeout=240)

def _run_global_search_process(query: str):
    data = load_graphrag_data()
    unique_id = str(uuid.uuid4())[:8]
    chat_config = LanguageModelConfig(
        api_key=config.api_key,
        auth_type=AuthType.APIKey,
        type=ModelType.AzureOpenAIChat,
        model=config.llm_model,
        deployment_name=config.llm_model,
        max_retries=20,
        api_base=config.api_base,
        api_version="2024-02-15-preview"
    )

    chat_model = ModelManager().get_or_create_chat_model(
        name=f"global_search_{unique_id}",
        model_type=ModelType.AzureOpenAIChat,
        config=chat_config
    )

    context_builder = GlobalCommunityContext(
        community_reports=data['reports'],
        communities=data['communities'],
        entities=data['entities'],
        token_encoder=data['token_encoder']
    )
    
    search_engine = GlobalSearch(
        model=chat_model,
        callbacks=[],
        context_builder=context_builder,
        token_encoder=data['token_encoder'],
        map_llm_params={"max_tokens": 1000, "temperature": 0.0, "response_format": {"type": "json_object"}},
        reduce_llm_params={"max_tokens": 5000, "temperature": 0.0},        
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params={"max_tokens": 12000},
        concurrent_coroutines=8,
        response_type="multiple paragraphs"
    )

    # Handle streaming search with real-time output
    async def stream_and_collect_results():
        full_response = ""
        async for chunk in search_engine.stream_search(query):
            if hasattr(chunk, 'response'):
                chunk_text = chunk.response
            else:
                chunk_text = str(chunk)
            
            # Print each chunk as it arrives for real-time streaming
            print(chunk_text, end='', flush=True)
            full_response += chunk_text
        
        print()  # Add newline at the end
        return full_response

    result = asyncio.run(stream_and_collect_results())
    return result

async def perform_global_search_stream(query: str):
    data = load_graphrag_data()
    unique_id = str(uuid.uuid4())[:8]
    chat_config = LanguageModelConfig(
        api_key=config.api_key,
        auth_type=AuthType.APIKey,
        type=ModelType.AzureOpenAIChat,
        model=config.llm_model,
        deployment_name=config.llm_model,
        max_retries=20,
        api_base=config.api_base,
        api_version="2024-02-15-preview"
    )

    chat_model = ModelManager().get_or_create_chat_model(
        name=f"global_search_{unique_id}",
        model_type=ModelType.AzureOpenAIChat,
        config=chat_config
    )

    context_builder = GlobalCommunityContext(
        community_reports=data['reports'],
        communities=data['communities'],
        entities=data['entities'],
        token_encoder=data['token_encoder']
    )
    
    search_engine = GlobalSearch(
        model=chat_model,
        callbacks=[],
        context_builder=context_builder,
        token_encoder=data['token_encoder'],
        map_llm_params={"max_tokens": 1000, "temperature": 0.0, "response_format": {"type": "json_object"}},
        reduce_llm_params={"max_tokens": 5000, "temperature": 0.0},        
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params={"max_tokens": 12000},
        concurrent_coroutines=8,
        response_type="multiple paragraphs"
    )

    async for chunk in search_engine.stream_search(query):
        chunk_text = chunk.response if hasattr(chunk, 'response') else str(chunk)
        yield chunk_text
