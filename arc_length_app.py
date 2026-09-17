"""Streamlit viewer for inference arc-length CSVs.

Run: streamlit run arc_length_app.py
"""
import math
import textwrap
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ARTIFACTS = Path(__file__).parent / "artifacts" / "content" / "artifacts"
# group column (defines the ordinal sequence) and label column per dataset
DATASETS = {
    "severity": ("template", "severity_word"),
    "severity_flipped": ("template", "severity_word"),
    "severity_pairwise": ("template", "severity_word"),
    "severity_wording": ("task", "prompt"),
}
METRICS = ("arc_length_parallel", "arc_length_orthogonal")
MAX_COLS_PER_ROW = 5
TICK_CHARS = 32  # x tick labels are truncated to this many characters
TICK_FONT = 14
PANEL_TITLE_FONT = 16
FIG_TITLE_FONT = 19

st.set_page_config(page_title="Arc Lengths", layout="wide")
st.title("Arc lengths by filled-in phrase")

EXCLUDED_MODELS = {"Qwen3-14B", "Qwen3-8B"}
MODELS = tuple(
    sorted(
        p.name
        for p in ARTIFACTS.iterdir()
        if (p / "inference").is_dir() and p.name not in EXCLUDED_MODELS
    )
)
PALETTE = ("#3B7DD8", "#C4632E", "#2E8B7A", "#8C5BB0", "#B03A5B")


@st.cache_data
def load(dataset: str, models: tuple[str, ...]) -> pd.DataFrame:
    """Every model's arc lengths for one dataset, stacked, in CSV row order."""
    frames = []
    for model in models:
        path = ARTIFACTS / model / "inference" / dataset / f"{dataset}_arc_lengths.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, encoding="utf-8")
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def shorten(text: str, limit: int = TICK_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


c1, c2 = st.columns([1, 3])
dataset = c1.selectbox("Dataset", list(DATASETS))
group_col, label_col = DATASETS[dataset]

df = load(dataset, MODELS)
tasks = list(dict.fromkeys(df[group_col]))
task = c2.selectbox(group_col.replace("_", " ").capitalize(), tasks)

sub = df[df[group_col] == task]
# Category order = 1-based order of rows within the group, as written in the CSV.
order = list(dict.fromkeys(sub[label_col]))
ticktext = [shorten(value) for value in order]
present = [model for model in MODELS if model in set(sub["model"])]
if not present:
    st.warning("No model has arc lengths for this selection.")
    st.stop()
# Never leave empty columns: with fewer models than the cap, the panels stretch to fill the row.
cols = min(MAX_COLS_PER_ROW, len(present))
rows = math.ceil(len(present) / cols)
title = "<br>".join(textwrap.wrap(str(task), 110))


def build(metric: str) -> go.Figure:
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=present,
        horizontal_spacing=0.025,
        vertical_spacing=0.18,
    )
    for i, model in enumerate(present):
        row, col = divmod(i, cols)
        part = sub[sub["model"] == model]
        fig.add_trace(
            go.Scatter(
                x=part[label_col],
                y=part[metric],
                mode="lines+markers",
                name=model,
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                marker=dict(size=7),
                showlegend=False,
                customdata=part[[label_col]],
                hovertemplate=f"{model}<br>%{{customdata[0]}}<br>{metric}=%{{y:.2f}}<extra></extra>",
            ),
            row=row + 1,
            col=col + 1,
        )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=order,
        tickmode="array",
        tickvals=order,
        ticktext=ticktext,
        tickangle=45,
        tickfont=dict(size=TICK_FONT),
    )
    # Each panel autoscales independently; the values stay available on hover.
    fig.update_yaxes(showticklabels=False, title_text=None)
    fig.update_annotations(font=dict(size=PANEL_TITLE_FONT))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=FIG_TITLE_FONT)),
        height=400 * rows + 180,
        margin=dict(t=95 + 22 * title.count("<br>"), b=180),
    )
    return fig


for metric in METRICS:
    st.subheader(metric.replace("_", " "))
    st.plotly_chart(build(metric), width="stretch", key=f"chart_{metric}")

with st.expander("Raw data"):
    st.dataframe(sub, width="stretch")
