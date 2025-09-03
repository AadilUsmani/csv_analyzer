import os
from fastapi import UploadFile

UPLOAD_DIR = "data/uploads"

def save_csv(file: UploadFile) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(file.file.read())
    return filepath

def get_latest_csv_path() -> str:
    files = os.listdir(UPLOAD_DIR)
    if not files:
        raise FileNotFoundError("No CSV uploaded yet.")
    latest_file = max(
        [os.path.join(UPLOAD_DIR, f) for f in files],
        key=os.path.getctime
    )
    return latest_file
