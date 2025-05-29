import os
from dotenv import load_dotenv
import tiktoken
import uuid
import multiprocessing
import warnings
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
    
    print("Loading GraphRAG data for Local Search...")
    
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
    
    print("GraphRAG data loaded for Local Search")
    return _global_data

class LocalSearchTool(BaseTool):
    name: str = "local_search"
    description: str = "Tool to perform local search operations using GraphRAG's index."

    # Declare all extra fields as Pydantic fields here
    input_dir: str = None
    llm_model: str = None
    embedding_model: str = None
    api_key: str = None
    api_base: str = None   
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    args_schema: type[BaseModel] = QueryInput

    def _run(self, query):
        import multiprocessing
        import warnings
        import time
        
        # Suppress AsyncLimiter warnings
        warnings.filterwarnings("ignore", message="This AsyncLimiter instance is being re-used across loops")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="This AsyncLimiter instance is being re-used across loops")
        
        try:
            # Use multiprocessing to completely isolate the async operations
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(1) as pool:
                result = pool.apply_async(_run_local_search_process, (query,))
                response = result.get(timeout=120)
            return response
        except multiprocessing.TimeoutError:
            print(f"LocalSearchTool timeout for query: {query}")
            return "I'm sorry, the search operation timed out. Please try again with a simpler question."
        except Exception as e:
            print(f"LocalSearchTool error for query '{query}': {e}")
            return f"I'm sorry, there was an error processing your request: {str(e)}"

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

# Create agent for local search only
print("GRAPHRAG MODEL", config.llm_model)
local_search_agent = Agent(
    role="You are an expert in Microsoft Fabric product specializing in local search",
    goal="Your goal is to answer the user's questions about Microsoft Fabric using local search capabilities.",
    backstory='As an expert in Microsoft Fabric, you have extensive knowledge about its features,' 
    ' functionalities, and best practices. You specialize in providing detailed, context-specific answers using local search.',
    tools=[LocalSearchTool()],
    llm=LLM(model=f'azure/{config.llm_model}'),
    verbose=True,
)

local_task_instructions = """
You are an expert in Microsoft Fabric product. Your goal is to answer the user's question below about Microsoft Fabric using LOCAL SEARCH only.

Users' question:
{question}

You must follow the below rules:
1. You must use ONLY the local search tool provided to you to answer the user's questions.
2. You must provide a comprehensive and accurate answer based on the local search results.
3. You must not mention that you are using local search - just provide the answer naturally.
4. If the local search doesn't provide sufficient information, mention that the information might not be available in the local context.
"""

async def execute_local_search_task(task_instructions):
    task = Task(
        description=task_instructions,
        agent=local_search_agent,
        expected_output="A concise and accurate answer to the user's question about Microsoft Fabric based on local search."
    )
    crew = Crew(agents=[local_search_agent], tasks=[task])
    result = crew.kickoff()
    return result
