"""Streamlit viewer for inference arc-length CSVs.

Run: streamlit run arc_length_app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ARTIFACTS = Path(__file__).parent / "artifacts" / "content" / "artifacts"
# group column (defines the ordinal sequence) and label column per dataset
DATASETS = {
    "severity": ("template", "severity_word"),
    "severity_flipped": ("template", "severity_word"),
    "severity_pairwise": ("template", "severity_word"),
    "severity_wording": ("task", "prompt"),
}

st.set_page_config(page_title="Arc Lengths", layout="wide")
st.title("Arc lengths vs ordinal position")

models = sorted(p.name for p in ARTIFACTS.iterdir() if (p / "inference").is_dir())
c1, c2 = st.columns(2)
model = c1.selectbox("Model", models)
dataset = c2.selectbox("Dataset", list(DATASETS))


@st.cache_data
def load(model: str, dataset: str) -> pd.DataFrame:
    path = ARTIFACTS / model / "inference" / dataset / f"{dataset}_arc_lengths.csv"
    df = pd.read_csv(path, encoding="utf-8")
    group_col, _ = DATASETS[dataset]
    # Ordinal position = 1-based order of rows within each group, as written in the CSV.
    df["ordinal_position"] = df.groupby(group_col, sort=False).cumcount() + 1
    return df


df = load(model, dataset)
group_col, label_col = DATASETS[dataset]

groups = list(df[group_col].unique())
selected = st.multiselect(f"Filter {group_col}s (empty = all)", groups)
if selected:
    df = df[df[group_col].isin(selected)]

for metric in ("arc_length_parallel", "arc_length_orthogonal"):
    fig = px.line(
        df,
        x="ordinal_position",
        y=metric,
        color=group_col,
        markers=True,
        hover_data=[label_col],
        title=f"{metric} vs ordinal position — {model} / {dataset}",
    )
    fig.update_layout(height=550, legend=dict(font=dict(size=10)))
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")

with st.expander("Raw data"):
    st.dataframe(df, width="stretch")
