import os

def prepare_backend_dirs():
    """Ensure data and models directories exist"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    return data_dir, models_dir

import uuid

def generate_job_id():
    """Generate a unique job ID for the task"""
    return str(uuid.uuid4())
