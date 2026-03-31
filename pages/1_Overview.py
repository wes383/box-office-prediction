from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import configure_page, format_money, hero, load_datasets, release_timeline, select_model_row


configure_page("Overview", icon="film")

data = load_datasets()
features = data["features"]
all_released = data["all_released"]
strict = data["strict"]
model_results = data["model_results"].copy()
best_model = select_model_row(model_results)
timeline_df = release_timeline(all_released)
model_table = (
    model_results.sort_values(["SelectedForDeployment", "RMSE"], ascending=[False, True])
    if "SelectedForDeployment" in model_results.columns
    else model_results.sort_values("RMSE").reset_index(drop=True)
)

hero(
    "Overview",
    "This page frames the project before we dive into filters or single-title forecasting: how big the dataset is, how reliable different slices are, and how much signal the best model recovered versus naive baselines.",
)

top = st.columns(3)
top[0].metric("All released films", f"{len(all_released):,}")
top[1].metric("Strict sample", f"{len(strict):,}")
top[2].metric("Best holdout R2", f"{best_model['R2']:.3f}")

mid = st.columns([1.1, 1], gap="large")

with mid[0]:
    st.subheader("Data quality signals")
    quality = pd.DataFrame(
        [
            {"Metric": "Zero budget share", "Value": (features["budget_raw"] <= 0).mean() * 100},
            {"Metric": "Zero revenue share", "Value": (features["revenue_raw"] <= 0).mean() * 100},
            {"Metric": "Proxy theatrical share", "Value": features["is_conservative_theatrical_proxy"].fillna(0).mean() * 100},
            {"Metric": "Budget imputed share", "Value": features["budget_imputed_flag"].fillna(0).mean() * 100},
        ]
    )
    fig_quality = px.bar(
        quality,
        x="Value",
        y="Metric",
        orientation="h",
        template="plotly_white",
        color="Value",
        color_continuous_scale=["#d9cab3", "#a63d40"],
    )
    fig_quality.update_layout(coloraxis_showscale=False, xaxis_title="Percent")
    st.plotly_chart(fig_quality, use_container_width=True)

    st.subheader("Model table")
    st.dataframe(
        model_table[
            [col for col in ["Model", "SelectedForDeployment", "RMSE", "MAE", "R2", "WAPE", "Mean_RMSE", "Mean_R2"] if col in model_table.columns]
        ]
        .style.format({"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2": "{:.3f}", "WAPE": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

with mid[1]:
    st.subheader("Release timeline")
    fig_timeline = px.area(
        timeline_df,
        x="release_year",
        y="movies",
        template="plotly_white",
        line_shape="spline",
    )
    fig_timeline.update_traces(line_color="#1c6e8c", fillcolor="rgba(28, 110, 140, 0.20)")
    fig_timeline.update_layout(xaxis_title="Release year", yaxis_title="Film count")
    st.plotly_chart(fig_timeline, use_container_width=True)

    timeline_df["avg_revenue_display"] = timeline_df["avg_revenue"].map(format_money)
    st.dataframe(
        timeline_df.tail(10).sort_values("release_year", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

bottom = st.columns(3)
bottom[0].metric("Median revenue", format_money(float(all_released["revenue"].median())))
bottom[1].metric("Mean revenue", format_money(float(all_released["revenue"].mean())))
bottom[2].metric("Holdout start", str(data["holdout"]["release_date"].min().date()))

st.caption(
    "The main structural issue is not missingness. It is the large fraction of zero-budget and zero-revenue rows, which is why the strict and theatrical-proxy slices matter so much."
)
