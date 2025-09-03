# main.py
import os
import uuid
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from services.csv_service import process_csv, get_cached_csv, generate_summary, generate_visualizations
from services.llm_service import query_csv_with_llm

UPLOAD_DIR = "uploads"
PLOTS_DIR = "plots"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

app = FastAPI(title="CSV Intelligence API")

# ✅ Fixed CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Removed trailing slash
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root route
@app.get("/")
async def root():
    return {"message": "CSV Intelligence API is running"}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Upload CSV
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    if not session_id:
        session_id = str(uuid.uuid4())

    filename = f"{session_id}.csv"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process CSV and cache
        process_csv(session_id, file_path)

        df = get_cached_csv(session_id)
        summary = generate_summary(df)
        return {"session_id": session_id, "message": "CSV uploaded and processed", "summary_preview": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

# ✅ Alias route for frontend
@app.post("/upload")
async def upload_csv_alias(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    return await upload_csv(file, session_id)

# Get summary
@app.get("/summary/{session_id}")
async def get_summary(session_id: str):
    try:
        df = get_cached_csv(session_id)
        summary = generate_summary(df)
        return JSONResponse(content=summary)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Get visualizations
@app.get("/visualize/{session_id}")
async def get_visualizations(session_id: str):
    try:
        df = get_cached_csv(session_id)
        plot_paths = generate_visualizations(df, session_id)
        filenames = [os.path.basename(p) for p in plot_paths]
        return {"plots": filenames}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Serve plot images
@app.get("/plot/{filename}")
async def serve_plot(filename: str):
    safe_path = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(safe_path, media_type="image/png")

# LLM prompt template
prompt_template = """
You are a professional data analyst. You are given a CSV dataset context and a user query.

Steps:
1. Analyze the dataset and figure out what information is most relevant.
2. Think step by step about what the query is asking.
3. Provide a final clear answer that is useful and well-structured.

Only output the final answer, not your reasoning.

Dataset Context:
{context}

User Query:
{query}

Final Answer:
"""

# Query CSV with LLM
@app.post("/query/{session_id}")
async def query_csv(session_id: str, query: str = Form(...)):
    try:
        df = get_cached_csv(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    summary = generate_summary(df)

    try:
        answer = query_csv_with_llm(df, summary, query, prompt_template)
        return {"session_id": session_id, "query": query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM query failed: {e}")
