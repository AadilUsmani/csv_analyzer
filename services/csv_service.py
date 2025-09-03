import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid

CACHE = {}  # In-memory session cache


def process_csv(session_id: str, file_path: str):
    """Load CSV into cache for this session."""
    df = pd.read_csv(file_path)
    CACHE[session_id] = df


def get_cached_csv(session_id: str) -> pd.DataFrame:
    """Retrieve cached DataFrame by session_id."""
    if session_id not in CACHE:
        raise ValueError("Session not found or CSV not uploaded.")
    return CACHE[session_id]



def generate_summary(df: pd.DataFrame) -> str:
    """
    Generate a summary of the DataFrame (compatible across Pandas versions).
    """
    try:
        # Try with datetime_is_numeric if available
        summary = df.describe(datetime_is_numeric=True)
    except TypeError:
        # Fallback for Pandas versions without datetime_is_numeric
        summary = df.describe()

    return summary.to_string()

def generate_visualizations(df: pd.DataFrame, session_id: str) -> list:
    """
    Generate simple visualizations and save as PNGs.
    Returns a list of file paths.
    """
    os.makedirs("plots", exist_ok=True)
    plot_paths = []

    # Example 1: Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    heatmap_path = f"plots/{session_id}_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    plot_paths.append(heatmap_path)

    # Example 2: Histogram of first numeric column
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[num_cols[0]], kde=True)
        hist_path = f"plots/{session_id}_hist.png"
        plt.savefig(hist_path)
        plt.close()
        plot_paths.append(hist_path)

    return plot_paths
