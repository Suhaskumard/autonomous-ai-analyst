from fastapi import APIRouter, File, UploadFile
from typing import Dict
import os
from ..utils.helpers import prepare_backend_dirs, generate_job_id

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict:
    """
    Accept CSV file, save to /data/, and return metadata
    """
    data_dir, _ = prepare_backend_dirs()
    
    file_path = os.path.join(data_dir, file.filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    # Mocking file metadata extraction
    file_size_kb = len(content) / 1024
    dataset_ref = f"ds_{generate_job_id()}"
    
    return {
        "status": "success",
        "dataset_ref": dataset_ref,
        "filename": file.filename,
        "size_kb": round(file_size_kb, 2),
        "message": "File uploaded successfully"
    }
