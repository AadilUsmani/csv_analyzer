import os
import matplotlib.pyplot as plt
from .file_handler import get_latest_csv_path

OUTPUT_DIR = "data/outputs"

def generate_graph(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    graph_path = os.path.join(OUTPUT_DIR, "graph.png")

    plt.figure(figsize=(6,4))
    df[df.columns[0]].value_counts().plot(kind="bar")
    plt.title("Distribution of first column")
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()
    
    return graph_path
