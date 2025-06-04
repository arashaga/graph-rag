import asyncio
import os
import logging
from semantic_kernel import Kernel
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory
#from samples.concepts.setup.chat_completion_services import Services, get_chat_completion_service_and_request_settings
from dotenv import load_dotenv

load_dotenv()

# async def execute_global_search_task_mcp(question: str) -> str:
#     kernel = Kernel()

#     from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings

#     chat_service = AzureChatCompletion(service_id="gloabal_search_service")
#     request_settings = AzureChatPromptExecutionSettings(service_id="gloabal_search_service")

#     # 1. Add an AI chat completion service (choose OpenAI, Azure, etc.)
#     chat_service, settings = get_chat_completion_service_and_request_settings(Services.AAZURE_OPENAI)
#     kernel.add_service(chat_service)
#     settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

#     # 2. Add your MCP plugin
#     plugin = MCPStreamableHttpPlugin(
#         name="GraphRag",
#         url="http://127.0.0.1:8111/mcp",
#         load_tools=True,
#         kernel=kernel
#     )
#     kernel.add_plugin(plugin)
#     # 3. Set up chat history
#     history = ChatHistory()

#     history.add_system_message(
#         "You are an expert in Microsoft Fabric providing global search-based answers. "
#         "Use your tools to search the knowledge graph. "
#         "Never ever use your internal knowledge, only use the tools provided. "
#         "If you don't know the answer, say 'I don't know'. "
#         "If you can't use your tool to answer just say I cannot invoke my tools to answer this question, please try again with a different question."
#     )
#     history.add_user_message(question)

#     # 4. Get a response
#     response = await kernel.get_service().get_streaming_chat_message_content(
#         history, settings, kernel=kernel
#     )
#     return response.content

async def execute_global_search_task_mcp_stream(question: str):
    kernel = Kernel()
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings

    deployment_name = "gpt-4o"  # Replace with your actual deployment name if different
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = "2024-02-15-preview"  # Use a known working version

    logging.warning(f"[DEBUG] deployment_name: {deployment_name}")
    logging.warning(f"[DEBUG] endpoint: {endpoint}")
    logging.warning(f"[DEBUG] api_key set: {bool(api_key)}")
    logging.warning(f"[DEBUG] api_version: {api_version}")

    if not deployment_name:
        raise ValueError("deployment_name is not set! Check your Azure deployment name.")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set!")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY is not set!")

    chat_service = AzureChatCompletion(
        service_id="gloabal_search_service",
        deployment_name=deployment_name,
        api_version=api_version,
        endpoint=endpoint,
        api_key=api_key
    )
    request_settings = AzureChatPromptExecutionSettings(service_id="gloabal_search_service")

    kernel.add_service(chat_service)
    settings = request_settings
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    plugin = MCPStreamableHttpPlugin(
        name="GraphRag",
        url="http://127.0.0.1:8111/mcp",
        load_tools=True,
        kernel=kernel
    )

    await plugin.connect()
    
    kernel.add_plugin(plugin)
    history = ChatHistory()
    history.add_system_message(
        "You are an expert in Microsoft Fabric providing global search-based answers. "
        "Use your plugin, MCP  tools to search the knowledge graph. "
        "Never ever use your internal knowledge, only use the tools provided. "
        "If you don't know the answer, say 'I don't know'. "
        "If you can't use your tool to answer just say I cannot invoke my tools to answer this question, please try again with a different question."
    )
    history.add_user_message(question)

    response_stream = kernel.get_service().get_streaming_chat_message_content(
        history, settings, kernel=kernel
    )
    async for chunk in response_stream:
        if chunk and hasattr(chunk, "content"):
            yield chunk.content

# Make sure you have set OPENAI_API_KEY or other credentials in your environment!