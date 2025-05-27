import json
import os
import shutil
from typing import AsyncGenerator
import uuid
import mimetypes
import dataclasses
from datetime import datetime
import jwt
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Query, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Form
from enum import Enum
from core.authentication import AuthenticationHelper
from core.sessionhelper import create_session_id
from decorators import authenticated, authenticated_path
from error import error_dict, error_response
from tasks import start_indexing, JOB_STATUS
from fastapi.responses import RedirectResponse,StreamingResponse,FileResponse
from fastapi import APIRouter # Make sure APIRouter is imported if not already
from pathlib import Path
from graph_rag_module import execute_task, task_instructions # Import task_instructions
from config import (
    CONFIG_AGENT_CLIENT,
    CONFIG_AGENTIC_RETRIEVAL_ENABLED,
    CONFIG_ASK_APPROACH,
    CONFIG_ASK_VISION_APPROACH,
    CONFIG_AUTH_CLIENT,
    CONFIG_BLOB_CONTAINER_CLIENT,
    CONFIG_CHAT_APPROACH,
    CONFIG_CHAT_HISTORY_BROWSER_ENABLED,
    CONFIG_CHAT_HISTORY_COSMOS_ENABLED,
    CONFIG_CHAT_VISION_APPROACH,
    CONFIG_CREDENTIAL,
    CONFIG_DEFAULT_REASONING_EFFORT,
    CONFIG_GPT4V_DEPLOYED,
    CONFIG_INGESTER,
    CONFIG_LANGUAGE_PICKER_ENABLED,
    CONFIG_OPENAI_CLIENT,
    CONFIG_QUERY_REWRITING_ENABLED,
    CONFIG_REASONING_EFFORT_ENABLED,
    CONFIG_SEARCH_CLIENT,
    CONFIG_SEMANTIC_RANKER_DEPLOYED,
    CONFIG_SPEECH_INPUT_ENABLED,
    CONFIG_SPEECH_OUTPUT_AZURE_ENABLED,
    CONFIG_SPEECH_OUTPUT_BROWSER_ENABLED,
    CONFIG_SPEECH_SERVICE_ID,
    CONFIG_SPEECH_SERVICE_LOCATION,
    CONFIG_SPEECH_SERVICE_TOKEN,
    CONFIG_SPEECH_SERVICE_VOICE,
    CONFIG_STREAMING_ENABLED,
    CONFIG_USER_BLOB_CONTAINER_CLIENT,
    CONFIG_USER_UPLOAD_ENABLED,
    CONFIG_VECTOR_SEARCH_ENABLED,
)





app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Load GraphRAG data once at startup
    print("🚀 Loading GraphRAG data at server startup...")
    from graph_rag_module import load_graphrag_data
    load_graphrag_data()
    print("✅ GraphRAG data loaded successfully!")
    
    AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
    AZURE_USE_AUTHENTICATION = os.getenv("AZURE_USE_AUTHENTICATION", "").lower() == "true"
    AZURE_ENFORCE_ACCESS_CONTROL = os.getenv("AZURE_ENFORCE_ACCESS_CONTROL", "").lower() == "true"
    AZURE_ENABLE_GLOBAL_DOCUMENT_ACCESS = os.getenv("AZURE_ENABLE_GLOBAL_DOCUMENT_ACCESS", "").lower() == "true"
    AZURE_ENABLE_UNAUTHENTICATED_ACCESS = os.getenv("AZURE_ENABLE_UNAUTHENTICATED_ACCESS", "").lower() == "true"
    AZURE_SERVER_APP_ID = os.getenv("AZURE_SERVER_APP_ID")
    AZURE_SERVER_APP_SECRET = os.getenv("AZURE_SERVER_APP_SECRET")
    AZURE_CLIENT_APP_ID = os.getenv("AZURE_CLIENT_APP_ID")
    AZURE_AUTH_TENANT_ID = os.getenv("AZURE_AUTH_TENANT_ID", AZURE_TENANT_ID)
    setattr(app.state, CONFIG_AUTH_CLIENT, AuthenticationHelper(
        search_index="",
        use_authentication=AZURE_USE_AUTHENTICATION,
        server_app_id=AZURE_SERVER_APP_ID,
        server_app_secret=AZURE_SERVER_APP_SECRET,
        client_app_id=AZURE_CLIENT_APP_ID,
        tenant_id=AZURE_AUTH_TENANT_ID,
        require_access_control=AZURE_ENFORCE_ACCESS_CONTROL,
        enable_global_documents=AZURE_ENABLE_GLOBAL_DOCUMENT_ACCESS,
        enable_unauthenticated_access=AZURE_ENABLE_UNAUTHENTICATED_ACCESS,
    ))
    #setattr(app.state, CONFIG_SEARCH_CLIENT, SearchClient(...))

# Fix Windows registry issue with mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return super().default(o)


@app.get("/assets/{path:path}")
async def assets(path: str):
    assets_dir = Path(__file__).resolve().parent / "static" / "assets"
    file_path = assets_dir / path
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(file_path)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="http://localhost:5173/")




app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Add both
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INPUT_FOLDER = "./input"
os.makedirs(INPUT_FOLDER, exist_ok=True)

class IndexingMethodEnum(str, Enum):
    standard = "Standard"
    fast = "Fast"


@app.get("/auth_setup")
async def auth_setup():
    try:
        auth_helper = getattr(app.state, CONFIG_AUTH_CLIENT, None)
        if auth_helper:
            return auth_helper.get_auth_setup_for_client()
        else:
            # Use real app registration values from env variables
            tenant_id = os.getenv("AZURE_AUTH_TENANT_ID", "16b3c013-d300-468d-ac64-7eda0820b6d3")
            client_id = os.getenv("AZURE_CLIENT_APP_ID", "4333edca-4f09-4892-bd0f-e3cfc1e1b33d")
            
            # For local development, modify the redirectUri to point to the local frontend
            redirectUri = "http://localhost:5173/redirect"
            
            return {
                "useLogin": True,
                "requireAccessControl": False,  # For easier local testing
                "enableUnauthenticatedAccess": True,  # Enable this for development to allow both auth and unauth access
                "msalConfig": {
                    "auth": {
                        "clientId": client_id,
                        "authority": f"https://login.microsoftonline.com/{tenant_id}",
                        "redirectUri": redirectUri,
                        "postLogoutRedirectUri": "http://localhost:5173/",
                        "navigateToLoginRequestUrl": True,
                    },
                    "cache": {
                        "cacheLocation": "localStorage",
                        "storeAuthStateInCookie": False,
                    },
                },
                "loginRequest": {"scopes": [".default"]},
                "tokenRequest": {"scopes": ["api://" + client_id + "/access_as_user"]},
            }
    except Exception as e:
        print(f"Error in auth_setup: {e}")
        # Use app registration values even in the fallback
        tenant_id = os.getenv("AZURE_AUTH_TENANT_ID", "16b3c013-d300-468d-ac64-7eda0820b6d3")
        client_id = os.getenv("AZURE_CLIENT_APP_ID", "4333edca-4f09-4892-bd0f-e3cfc1e1b33d")
        
        # For local development, modify the redirectUri to point to the local frontend
        redirectUri = "http://localhost:5173/redirect"
        
        return {
            "useLogin": True,
            "requireAccessControl": False,  # For easier local testing 
            "enableUnauthenticatedAccess": True,  # Enable this for development
            "msalConfig": {
                "auth": {
                    "clientId": client_id,
                    "authority": f"https://login.microsoftonline.com/{tenant_id}",
                    "redirectUri": redirectUri,
                    "postLogoutRedirectUri": "http://localhost:5173/",
                    "navigateToLoginRequestUrl": True,
                },
                "cache": {
                    "cacheLocation": "localStorage",
                    "storeAuthStateInCookie": False,
                },
            },
            "loginRequest": {"scopes": [".default"]},
            "tokenRequest": {"scopes": ["api://" + client_id + "/access_as_user"]},
        }

@app.post("/chat/")
async def chat_endpoint(request: Request):
    data = await request.json()
    
    # Handle the ChatAppRequest format from the frontend
    messages = data.get("messages", [])
    if not messages:
        return {"error": "Missing 'messages'"}
    
    # Extract the latest user message
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    if not user_messages:
        return {"error": "No user message found"}
    
    question = user_messages[-1].get("content", "")
    if not question:
        return {"error": "Empty question"}
    
    try:
        current_task_instructions = task_instructions.format(question=question)
        crew_output = await execute_task(current_task_instructions)
        
        # Convert CrewOutput object to string
        response = str(crew_output)
        
        # Return in the format expected by the frontend
        return {
            "message": {
                "content": response,
                "role": "assistant"
            },
            "context": {
                "data_points": [],
                "followup_questions": [],
                "thoughts": []
            },
            "session_state": data.get("session_state")
        }
    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        return {"error": f"Error processing request: {str(e)}"}


@app.post("/chat/stream/")
async def chat_stream_endpoint(request: Request):
    data = await request.json()
    
    # Handle the ChatAppRequest format from the frontend
    messages = data.get("messages", [])
    if not messages:
        return {"error": "Missing 'messages'"}
    
    # Extract the latest user message
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    if not user_messages:
        return {"error": "No user message found"}
    
    question = user_messages[-1].get("content", "")
    if not question:
        return {"error": "Empty question"}

    async def response_generator() -> AsyncGenerator[str, None]:
        try:
            current_task_instructions = task_instructions.format(question=question)
            crew_output = await execute_task(current_task_instructions)
            
            # Convert CrewOutput object to string
            response = str(crew_output)
            
            # Send initial context data with delta (required by frontend)
            yield json.dumps({
                "context": {
                    "data_points": [],
                    "followup_questions": [],
                    "thoughts": []
                },
                "delta": {
                    "content": "",
                    "role": "assistant"
                }
            }) + "\n"
            
            # Stream the response content
            yield json.dumps({
                "delta": {
                    "content": response,
                    "role": "assistant"
                }
            }) + "\n"
            
        except Exception as e:
            print(f"Error in chat_stream_endpoint: {e}")
            yield json.dumps({"error": f"Error processing request: {str(e)}"}) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")

@app.post("/upload/")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: IndexingMethodEnum = Query(IndexingMethodEnum.standard),
    index_name: str = Form(...)
):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are accepted.")

    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_index_name = "".join(c for c in index_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    output_folder = os.path.join("output", f"{timestamp}_{safe_index_name}")
    os.makedirs(output_folder, exist_ok=True)

    # Save uploaded file to the output_folder
    save_path = os.path.join(output_folder, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    JOB_STATUS[job_id] = {"status": "pending"}

    # Key fix: Pass output_folder, NOT save_path!
    background_tasks.add_task(start_indexing, job_id, output_folder, method)

    return {"job_id": job_id, "status": "pending"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    status = JOB_STATUS.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@app.get("/.auth/me")
async def auth_me_endpoint(request: Request):
    """
    Provides authentication information in a format that matches
    what the frontend expects from Azure App Service authentication.
    Uses the real auth token if provided in the Authorization header.
    """
    auth_helper = getattr(app.state, CONFIG_AUTH_CLIENT, None)
    use_auth = auth_helper.use_authentication if auth_helper else False
    
    if not use_auth:
        # Return empty array as if user is not authenticated
        return []
    
    # Check for an actual token in the Authorization header
    authorization = request.headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        try:
            # If we have a real token, use it
            # Try to decode the token to extract claims
            try:
                # This is just to extract information, not for validation
                decoded_token = jwt.decode(token, options={"verify_signature": False})
                user_claims = []
                
                # Convert token claims to the format expected by the frontend
                for claim_key, claim_value in decoded_token.items():
                    if claim_key in ["exp", "nbf", "iat"]:  # Skip timestamp claims
                        continue
                    user_claims.append({"typ": claim_key, "val": claim_value})
                
                print(f"Successfully processed token with user: {decoded_token.get('name', 'Unknown')}")
                
                return [{
                    "id_token": token,
                    "access_token": token,
                    "expires_on": str(decoded_token.get("exp", datetime.now().timestamp() + 3600)),
                    "provider": "aad",
                    "user_claims": user_claims
                }]
            except Exception as decode_error:
                print(f"Error decoding token: {decode_error}")
                # If we can't decode it, just use it as is with minimal claims
                return [{
                    "id_token": token,
                    "access_token": token,
                    "expires_on": str(datetime.now().timestamp() + 3600),  # 1 hour from now
                    "provider": "aad",
                    "user_claims": [
                        {"typ": "sub", "val": "authenticated-user"},
                        {"typ": "name", "val": "Authenticated User"}
                    ]
                }]
        except Exception as e:
            print(f"Error handling token: {e}")
            # Fall through to the mock token if token handling fails
    else:
        print("No Authorization header found or it doesn't start with 'Bearer '")
    
    # For local development without a token, check if unauthenticated access is allowed
    enable_unauthenticated = os.getenv("AZURE_ENABLE_UNAUTHENTICATED_ACCESS", "").lower() == "true"
    
    if not enable_unauthenticated:
        # If unauthenticated access is not allowed, return 401
        print("Unauthenticated access is not allowed and no valid token provided")
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # For development, return a mock token with basic claims
    user_id = str(uuid.uuid4())
    expiry_time = datetime.now().timestamp() + 3600  # 1 hour from now
    
    print("Returning mock token for local development")
    return [
        {
            "id_token": "mock_id_token_for_development",
            "access_token": "mock_access_token_for_development",
            "expires_on": str(expiry_time),
            "provider": "aad",
            "user_claims": [
                {"typ": "sub", "val": user_id},
                {"typ": "name", "val": "Local Development User"},
                {"typ": "preferred_username", "val": "localuser@example.com"},
                {"typ": "oid", "val": user_id}
            ]
        }
    ]

@app.get("/.auth/refresh")
async def auth_refresh_endpoint():
    """
    Emulates the Azure App Service's .auth/refresh endpoint for local development.
    This endpoint refreshes the authentication token.
    """
    auth_helper = getattr(app.state, CONFIG_AUTH_CLIENT, None)
    use_auth = auth_helper.use_authentication if auth_helper else False
    
    if not use_auth:
        # Return 401 Unauthorized if authentication is not enabled
        raise HTTPException(status_code=401, detail="Authentication is not enabled")
    
    # If authentication is enabled, return a success response
    # The frontend will then call .auth/me again to get the new token
    return {"refreshed": True}

@app.get("/.auth/logout")
async def auth_logout_endpoint(post_logout_redirect_uri: str = "/"):
    """
    Emulates the Azure App Service's .auth/logout endpoint for local development.
    This endpoint logs the user out and redirects to the specified URI.
    """
    # Since we don't have real authentication in local development,
    # we'll just redirect to the specified URI
    return RedirectResponse(url=post_logout_redirect_uri)

@app.get("/config")
async def config_endpoint():
    # Helper to convert env var string to boolean
    def get_bool_env(var_name: str, default: bool = False) -> bool:
        return os.getenv(var_name, str(default)).lower() == "true"

    return {
        #"showGPT4VOptions": CONFIG_GPT4V_DEPLOYED,
        #"showSemanticRankerOption": CONFIG_SEMANTIC_RANKER_DEPLOYED,
        #"showQueryRewritingOption": CONFIG_QUERY_REWRITING_ENABLED,
       # "showReasoningEffortOption": CONFIG_REASONING_EFFORT_ENABLED,
        "streamingEnabled": CONFIG_STREAMING_ENABLED,
        #"defaultReasoningEffort": CONFIG_DEFAULT_REASONING_EFFORT,
        #"showVectorOption": CONFIG_VECTOR_SEARCH_ENABLED,
        #"showUserUpload": CONFIG_USER_UPLOAD_ENABLED,
        "showLanguagePicker": CONFIG_LANGUAGE_PICKER_ENABLED,
        "showSpeechInput": CONFIG_SPEECH_INPUT_ENABLED,
        "showSpeechOutputBrowser": CONFIG_SPEECH_OUTPUT_BROWSER_ENABLED,
        "showSpeechOutputAzure": CONFIG_SPEECH_OUTPUT_AZURE_ENABLED,
        "showChatHistoryBrowser": CONFIG_CHAT_HISTORY_BROWSER_ENABLED,
        "showChatHistoryCosmos": CONFIG_CHAT_HISTORY_COSMOS_ENABLED,
        #"showAgenticRetrievalOption": CONFIG_AGENTIC_RETRIEVAL_ENABLED,
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://localhost:50505")
    uvicorn.run(app, host="0.0.0.0", port=50505)
