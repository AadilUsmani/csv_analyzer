import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def generate_summary(df: pd.DataFrame) -> dict:
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "description": df.describe(include="all").to_dict(),
        "correlation": df.corr(numeric_only=True).to_dict()
    }
    return summary

def generate_plot(df: pd.DataFrame, kind: str = "hist", column: str = None) -> str:
    plt.figure(figsize=(6,4))

    if kind == "hist" and column:
        df[column].hist()
    elif kind == "bar" and column:
        df[column].value_counts().plot(kind="bar")
    elif kind == "scatter" and column and len(column.split(",")) == 2:
        x, y = column.split(",")
        df.plot.scatter(x=x.strip(), y=y.strip())
    else:
        raise ValueError("Invalid plot type or column selection")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
