from __future__ import annotations

import streamlit as st

from app_utils import configure_page, hero, load_datasets, format_money, select_model_row


configure_page("Film Box Office Dashboard", icon="film")

data = load_datasets()
features = data["features"]
model_results = data["model_results"]
best_model = select_model_row(model_results)

hero(
    "Film Box Office Prediction Dashboard",
    "A four-part Streamlit workspace for dataset quality checks, exploratory analysis, model evaluation, and single-movie forecasting. Use the sidebar to move between Overview, EDA Explorer, Model Performance, and Single Movie Predictor.",
    eyebrow="Streamlit Workspace",
)

st.subheader("What this app covers")
st.markdown(
    """
    - `Overview`: data health, sample definitions, and headline model signals
    - `EDA Explorer`: filterable market patterns across genre, seasonality, budget, and language
    - `Model Performance`: holdout metrics, strict-slice behavior, and failure cases
    - `Single Movie Predictor`: primary deployment estimate with comparable-title context
    """
)

st.subheader("Project snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Feature rows", f"{len(features):,}")
c2.metric("Deployment model", best_model["Model"])
c3.metric("Holdout RMSE", format_money(best_model["RMSE"]))

st.caption("Start with Overview if you want the shortest path through the story.")
