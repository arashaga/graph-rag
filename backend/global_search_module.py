import os
from dotenv import load_dotenv
import tiktoken
import uuid
import multiprocessing
import warnings
import time
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
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
import pandas as pd
import asyncio
from pydantic import BaseModel

load_dotenv()

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

# Global data storage to avoid reloading data every time
_global_data = {}

def load_graphrag_data():
    """Load GraphRAG data once at startup"""
    global _global_data
    
    if _global_data:  # Already loaded
        return _global_data
    
    print("Loading GraphRAG data for Global Search...")
    
    data_dir = config.data_dir
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

    _global_data = {
        'entities': entities,
        'relationships': relationships,
        'reports': reports,
        'text_units': text_units,
        'communities': communities,
        'token_encoder': tiktoken.encoding_for_model(config.llm_model)
    }
    
    print("GraphRAG data loaded for Global Search")
    return _global_data

class GlobalSearchTool(BaseTool):
    name: str = "global_search"
    description: str = "Tool to perform global search operations using GraphRAG's index."
    input_dir: str = None
    llm_model: str = None
    api_key: str = None
    api_base: str = None

    def __init__(self, callbacks=None, **kwargs):
        super().__init__(**kwargs)
    
    args_schema: type[BaseModel] = QueryInput

    def _run(self, query):
        import multiprocessing
        import warnings
        
        warnings.filterwarnings("ignore", message="This AsyncLimiter instance is being re-used across loops")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="This AsyncLimiter instance is being re-used across loops")
        try:
            print("[DEBUG] Global search process started")
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(1) as pool:
                result = pool.apply_async(_run_global_search_process, (query,))
                response = result.get(timeout=240)
            return response
        except multiprocessing.TimeoutError:
            print(f"GlobalSearchTool timeout for query: {query}")
            import traceback
            traceback.print_exc()
            return "I'm sorry, the search operation timed out. Please try again with a simpler question."
        except Exception as e:
            print(f"GlobalSearchTool error for query '{query}': {e}")
            import traceback
            traceback.print_exc()
            return f"I'm sorry, there was an error processing your request: {str(e)}"

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

# Create agent for global search only
print("GRAPHRAG MODEL", config.llm_model)
global_search_agent = Agent(
    role="You are an expert in Microsoft Fabric product specializing in global search",
    goal="Your goal is to answer the user's questions about Microsoft Fabric using global search capabilities.",
    backstory='As an expert in Microsoft Fabric, you have extensive knowledge about its features,' 
    ' functionalities, and best practices. You specialize in providing high-level insights and comprehensive overviews using global search.',
    tools=[GlobalSearchTool()],
    llm=LLM(model=f'azure/{config.llm_model}'),
    verbose=True,
)

global_task_instructions = """
You are an expert in Microsoft Fabric product. Your goal is to answer the user's question below about Microsoft Fabric using GLOBAL SEARCH only.

Users' question:
{question}

You must follow the below rules:
1. You must use ONLY the global search tool provided to you to answer the user's questions.
2. You must provide a comprehensive and accurate answer based on the global search results.
3. You must not mention that you are using global search - just provide the answer naturally.
4. If the global search doesn't provide sufficient information, mention that the information might not be available in the global context.
"""

async def execute_global_search_task(task_instructions):
    task = Task(
        description=task_instructions,
        agent=global_search_agent,
        expected_output="A concise and accurate answer to the user's question about Microsoft Fabric based on global search."
    )
    crew = Crew(agents=[global_search_agent], tasks=[task])
    result = crew.kickoff()
    return result
