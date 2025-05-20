import os
import shutil
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Form
from enum import Enum

from tasks import start_indexing, JOB_STATUS

app = FastAPI()




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
