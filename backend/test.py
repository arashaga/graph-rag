import os
import tiktoken
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

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

def test_llm():
    from graphrag.config.enums import ModelType, AuthType
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.manager import ModelManager

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
        name="test_global_search",
        model_type=ModelType.AzureOpenAIChat,
        config=chat_config,
    )
    print("[DEBUG] Created chat model")
    prompt = "What is Microsoft Fabric?"
    response = chat_model([{"role": "user", "content": prompt}])
    print("LLM direct response:", response)

test_llm()
