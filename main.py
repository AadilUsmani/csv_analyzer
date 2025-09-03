# main.py
import os
import uuid
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# services should be a package (services/__init__.py) containing csv_service.py and llm_service.py
from services.csv_service import process_csv, get_cached_csv, generate_summary, generate_visualizations
from services.llm_service import query_csv_with_llm

UPLOAD_DIR = "uploads"
PLOTS_DIR = "plots"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

app = FastAPI(title="CSV Intelligence API")

# dev-friendly CORS — narrow this to your frontend URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://v0-csv-rag-dashboard.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    """
    Upload a CSV. If session_id provided (form field), use it; otherwise auto-generate.
    Returns session_id to be used for /summary, /visualize, /query.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    if not session_id:
        session_id = str(uuid.uuid4())

    filename = f"{session_id}.csv"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        # Save upload to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Let the csv_service load and cache the dataframe for the session
        process_csv(session_id, file_path)

        # Return a short summary preview so frontend can immediately show something
        df = get_cached_csv(session_id)
        summary = generate_summary(df)
        return {"session_id": session_id, "message": "CSV uploaded and processed", "summary_preview": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.get("/summary/{session_id}")
async def get_summary(session_id: str):
    """Return a JSON summary for the uploaded CSV tied to session_id."""
    try:
        df = get_cached_csv(session_id)
        summary = generate_summary(df)
        return JSONResponse(content=summary)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/visualize/{session_id}")
async def get_visualizations(session_id: str):
    """
    Generate and return list of plot file names for the session.
    Frontend can then fetch images from /plot/{filename}
    """
    try:
        df = get_cached_csv(session_id)
        plot_paths = generate_visualizations(df, session_id)  # returns list like ["plots/<id>_hist.png", ...]
        # Return only filenames (frontend will call /plot/{filename})
        filenames = [os.path.basename(p) for p in plot_paths]
        return {"plots": filenames}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/plot/{filename}")
async def serve_plot(filename: str):
    """Serve generated plot images (PNG)."""
    safe_path = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(safe_path, media_type="image/png")

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
@app.post("/query/{session_id}")
async def query_csv(session_id: str, query: str = Form(...)):
    """Ask the LLM a question about the CSV context for this session."""
    try:
        df = get_cached_csv(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # build structured summary/context
    summary = generate_summary(df)

    try:
        answer = query_csv_with_llm(df, summary, query, prompt_template)
        return {"session_id": session_id, "query": query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM query failed: {e}")
