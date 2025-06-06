import os
from dotenv import load_dotenv
import tiktoken
import uuid
import multiprocessing
import pickle
import time
from graphrag.config.enums import ModelType, AuthType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_communities,
)
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import kernel_function
import pandas as pd
import asyncio
import warnings

load_dotenv()



from pydantic import BaseModel

class QueryInput(BaseModel):
    query: str

class GraphRagBaseConfig:
    def __init__(self):
        self.api_key = os.getenv("GRAPHRAG_API_KEY")
        self.llm_model = os.getenv("GRAPHRAG_LLM_MODEL")
        self.embedding_model = os.getenv("GRAPHRAG_EMBEDDING_MODEL")
        self.api_base = os.getenv("GRAPHRAG_API_BASE")
        self.data_dir = os.getenv("GRAPHRAG_DATA_DIR")

        missing = [
            key for key in ["GRAPHRAG_API_KEY", "GRAPHRAG_LLM_MODEL", 
                            "GRAPHRAG_EMBEDDING_MODEL", "GRAPHRAG_API_BASE",
                            "GRAPHRAG_DATA_DIR"]
            if os.getenv(key) is None
        ]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

        self.token_encoder = tiktoken.encoding_for_model(self.llm_model)

config = GraphRagBaseConfig()
print("GRAPHRAG MODEL", config.llm_model)

# Global data storage to avoid reloading data every time
_global_data = {}

def load_graphrag_data():
    """Load GraphRAG data once at startup"""
    global _global_data
    
    if _global_data:  # Already loaded
        return _global_data
    
    print("Loading GraphRAG data...")
    
    data_dir = config.data_dir
    LANCEDB_URI = f"{data_dir}/lancedb"
    COMMUNITY_REPORT_TABLE = "community_reports"
    ENTITY_TABLE = "entities"
    COMMUNITY_TABLE = "communities"
    RELATIONSHIP_TABLE = "relationships"
    TEXT_UNIT_TABLE = "text_units"
    COMMUNITY_LEVEL = 2

    # Load DataFrames
    entity_df = pd.read_parquet(f"{data_dir}/{ENTITY_TABLE}.parquet")
    community_df = pd.read_parquet(f"{data_dir}/{COMMUNITY_TABLE}.parquet")
    relationship_df = pd.read_parquet(f"{data_dir}/{RELATIONSHIP_TABLE}.parquet")
    report_df = pd.read_parquet(f"{data_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
    text_unit_df = pd.read_parquet(f"{data_dir}/{TEXT_UNIT_TABLE}.parquet")

    entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)
    relationships = read_indexer_relationships(relationship_df)
    reports = read_indexer_reports(report_df, community_df, COMMUNITY_LEVEL)
    text_units = read_indexer_text_units(text_unit_df)
    communities = read_indexer_communities(community_df, report_df)

    # Load LanceDB Vector Store
    description_embedding_store = LanceDBVectorStore(
        collection_name="default-entity-description",
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)

    _global_data = {
        'entities': entities,
        'relationships': relationships,
        'reports': reports,
        'text_units': text_units,
        'communities': communities,
        'description_embedding_store': description_embedding_store,
        'token_encoder': tiktoken.encoding_for_model(config.llm_model)
    }
    
    print("GraphRAG data loaded")
    return _global_data

def _run_local_search_process(query):
    """
    Function to be run in a subprocess for LocalSearchTool. This avoids event loop conflicts and isolates async operations.
    """
    import warnings
    import uuid
    from graphrag.config.enums import ModelType, AuthType
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.manager import ModelManager
    from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
    from graphrag.query.structured_search.local_search.search import LocalSearch
    # Suppress AsyncLimiter warnings
    warnings.filterwarnings("ignore", message="This AsyncLimiter instance is being re-used across loops")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="This AsyncLimiter instance is being re-used across loops")

    data = load_graphrag_data()
    entities = data['entities']
    relationships = data['relationships']
    reports = data['reports']
    text_units = data['text_units']
    communities = data['communities']
    description_embedding_store = data['description_embedding_store']
    token_encoder = data['token_encoder']

    unique_id = str(uuid.uuid4())[:8]
    # LLM Model Config
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
        name=f"local_search_{unique_id}",
        model_type=ModelType.AzureOpenAIChat,
        config=chat_config,
    )

    embedding_config = LanguageModelConfig(
            api_key=config.api_key,
            auth_type=AuthType.APIKey,
            type=ModelType.AzureOpenAIEmbedding,
            model=config.embedding_model,
            deployment_name=config.embedding_model,
            api_base=config.api_base,
            api_version="2024-02-15-preview"
        )

    text_embedder = ModelManager().get_or_create_embedding_model(
        name="local_search_embedding",
        model_type=ModelType.AzureOpenAIEmbedding,
        config=embedding_config,
    )
    # Context builder (fix: use correct argument names as in the working notebook)
    _context_builder = LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=text_embedder,
            token_encoder=token_encoder,
        )
    context_builder_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            "max_tokens": 12_000,
        }

    model_params = {
            "max_tokens": 2_000,
            "temperature": 0.0,
        }

    search_engine = LocalSearch(
            model=chat_model,
            context_builder=_context_builder,
            token_encoder=token_encoder,
            model_params=model_params,
            context_builder_params=context_builder_params,
            response_type="multiple paragraphs"
        )
    import asyncio
    result = asyncio.run(search_engine.search(query))
    return result.response


def _run_global_search_process(query):
    import uuid
    import warnings
    import asyncio
    from graphrag.config.enums import ModelType, AuthType
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.manager import ModelManager
    from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
    from graphrag.query.structured_search.global_search.search import GlobalSearch
    start = time.time()
    print(f"[DEBUG] PID {os.getpid()} - global search subprocess started")
    # Suppress AsyncLimiter warnings
    warnings.filterwarnings("ignore", message="This AsyncLimiter instance is being re-used across loops")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="This AsyncLimiter instance is being re-used across loops")

    # Load pre-computed data
    data_start = time.time()
    data = load_graphrag_data()
    print(f"[DEBUG] PID {os.getpid()} - data loaded in {time.time() - data_start:.2f} sec")
    communities = data['communities']
    reports = data['reports']
    entities = data['entities']
    token_encoder = data['token_encoder']

    print(f"GlobalSearchTool Debug: communities count: {len(communities)}")
    print(f"GlobalSearchTool Debug: reports count: {len(reports)}")
    print(f"GlobalSearchTool Debug: entities count: {len(entities)}")

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
        config=chat_config,
    )

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,
        token_encoder=token_encoder,
    )

    context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }
    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    reduce_llm_params = {
        "max_tokens": 5000,
        "temperature": 0.0,
    }
    print("[DEBUG] Built search engine")
    search_engine = GlobalSearch(
        model=chat_model,
        callbacks=[],
        context_builder=context_builder,
        token_encoder=token_encoder,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=context_builder_params,
        concurrent_coroutines=8,
        response_type="multiple paragraphs",
    )
    llm_start = time.time()
    result = asyncio.run(search_engine.search(query))
    print("[DEBUG] Got result from search_engine.search()")
    print(f"[DEBUG] PID {os.getpid()} - LLM call took {time.time() - llm_start:.2f} sec")
    print(f"[DEBUG] PID {os.getpid()} - Total time: {time.time() - start:.2f} sec")
    return result.response

# --- Local Search Plugin ---
class LocalSearchPlugin:
    """Semantic Kernel plugin for local search using your existing logic."""
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

    @kernel_function(description="Answer a question using local search.")
    async def local_search(self, question: str) -> str:
        import time
        from local_search_module import perform_local_search_stream
        print("[Agentic] Invoking Local Search...")
        start = time.time()
        chunks = []
        async for chunk in perform_local_search_stream(question):
            chunks.append(chunk)
        elapsed = time.time() - start
        print(f"[Agentic] Local Search completed in {elapsed:.2f} seconds.")
        return "".join(chunks)

# --- Global Search Plugin ---
class GlobalSearchPlugin:
    """Semantic Kernel plugin for global search using your existing logic."""
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

    @kernel_function(description="Answer a question using global search.")
    async def global_search(self, question: str) -> str:
        import time
        from global_search_module import perform_global_search_stream
        print("[Agentic] Invoking Global Search...")
        start = time.time()
        chunks = []
        async for chunk in perform_global_search_stream(question):
            chunks.append(chunk)
        elapsed = time.time() - start
        print(f"[Agentic] Global Search completed in {elapsed:.2f} seconds.")
        return "".join(chunks)

# --- Agentic Search Agent using Semantic Kernel ---

async def agentic_search_semantic_kernel(question: str, search_method: str = "agentic") -> str:
    import time
    # AzureChatCompletion config
    deployment_name = config.llm_model
    endpoint = config.api_base
    api_key = config.api_key
    api_version = "2024-02-15-preview"

    service = AzureChatCompletion(
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )

    # Plugins
    local_plugin = LocalSearchPlugin(config, load_graphrag_data)
    global_plugin = GlobalSearchPlugin(config, load_graphrag_data)


    plugins = [local_plugin, global_plugin]
    instructions = (
        "Answer the user's question using both local and global search plugins. "
        "YOu MUST use both Tools Local and Global at the same time to put the response together. "
        "Combine the results, remove duplicate or similar content, and provide a detailed answer. "
        
    )

    agent = ChatCompletionAgent(
        service=service,
        name="AgenticSearch",
        instructions=instructions,
        plugins=plugins,
    )

    thread = None
    print("[Agentic] Starting agentic search...")
    total_start = time.time()
    response = await agent.get_response(messages=question, thread=thread)
    total_elapsed = time.time() - total_start
    print(f"[Agentic] Agentic search (total) completed in {total_elapsed:.2f} seconds.")
    return str(response)



# Indentation fix for import
from typing import AsyncGenerator

#### adding this for streaming support
async def agentic_search_semantic_kernel_stream(
    question: str,
    search_method: str = "agentic"
) -> AsyncGenerator[str, None]:
    import time

    deployment_name = config.llm_model
    endpoint = config.api_base
    api_key = config.api_key
    api_version = "2024-02-15-preview"

    service = AzureChatCompletion(
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )

    # Plugins
    local_plugin = LocalSearchPlugin(config, load_graphrag_data)
    global_plugin = GlobalSearchPlugin(config, load_graphrag_data)
    plugins = [local_plugin, global_plugin]

    instructions = (
        "Answer the user's question using both local and global search plugins. "
        "YOu MUST use both Tools Local and Global at the same time to put the response together. "
        "Combine the results, remove duplicate or similar content, and provide a detailed answer. "
    )

    agent = ChatCompletionAgent(
        service=service,
        name="AgenticSearch",
        instructions=instructions,
        plugins=plugins,
    )

    thread = None
    print("[Agentic] Starting agentic search (streaming)...")
    total_start = time.time()
    try:
        async for response_item in agent.invoke_stream(messages=question, thread=thread):
            # Each response_item.message is a StreamingChatMessageContent
            if hasattr(response_item.message, "content") and response_item.message.content:
                # Yield plain chunk text to the FastAPI endpoint
                yield response_item.message.content
    finally:
        total_elapsed = time.time() - total_start
        print(f"[Agentic] Agentic search (streaming) completed in {total_elapsed:.2f} seconds.")

# Sync wrapper for backward compatibility, if needed
async def execute_task_stream(task_instructions, search_method="agentic"):
    """
    Async wrapper for agentic_search_semantic_kernel_stream for FastAPI/main.py compatibility.
    """
    async for chunk in agentic_search_semantic_kernel_stream(task_instructions, search_method=search_method):
        yield chunk

# Optionally, add a sync wrapper for FastAPI/main.py compatibility

async def execute_task(task_instructions, search_method="agentic"):
    """
    Async wrapper for agentic_search_semantic_kernel for FastAPI/main.py compatibility.
    """
    return await agentic_search_semantic_kernel(task_instructions, search_method=search_method)
