import plotly.graph_objects as go
import pandas as pd

def trend_lines(df: pd.DataFrame, x: str, ys: list):
    fig = go.Figure()
    for col in ys:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=col))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    return fig