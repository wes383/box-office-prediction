from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import (
    compute_budget_band_errors,
    configure_page,
    format_money,
    hero,
    load_datasets,
    plot_actual_vs_pred,
    select_model_row,
    top_over_predictions,
    top_under_predictions,
)


configure_page("Model Performance", icon="film")

data = load_datasets()
holdout = data["holdout"].copy()
model_results = data["model_results"].copy()
slice_metrics = data["holdout_slice_metrics"]
cv_summary = data["cv_summary"].copy()
cv_results = data["cv_results"].copy()
leaderboard_df = (
    model_results.sort_values(["SelectedForDeployment", "RMSE"], ascending=[False, True])
    if "SelectedForDeployment" in model_results.columns
    else model_results.sort_values("RMSE").reset_index(drop=True)
)


def metric_row_from_df(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series({"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "SMAPE": np.nan, "WAPE": np.nan})
    return df.iloc[0]


def resolve_slice_metrics(
    slice_metrics_df: pd.DataFrame,
    broad_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if "Slice" in slice_metrics_df.columns:
        overall = slice_metrics_df[slice_metrics_df["Slice"] == "all_holdout"]
        strict = slice_metrics_df[slice_metrics_df["Slice"] == "strict_holdout_proxy"]
        compare_df = slice_metrics_df.copy()
        return (
            metric_row_from_df(overall),
            metric_row_from_df(strict),
            compare_df,
        )

    overall = metric_row_from_df(broad_df)
    strict = metric_row_from_df(pd.DataFrame())
    compare_df = pd.DataFrame(
        [
            {"Slice": "all_holdout", "R2": overall.get("R2", np.nan)},
        ]
    )
    return overall, strict, compare_df

hero(
    "Model Performance",
    "This page separates aggregate goodness from practical behavior. The goal is not only to show which model won, but also where the ensemble still misses badly, especially on blockbuster upside.",
)

best_model = select_model_row(model_results)
overall_metrics, strict_metrics, metric_compare = resolve_slice_metrics(
    slice_metrics,
    model_results,
)

best_model_name = best_model["Model"]
best_cv = cv_summary[cv_summary["Model"] == best_model_name]
if not best_cv.empty:
    best_cv_row = best_cv.iloc[0]
    recent_rmse = best_cv_row.get("Recent_RMSE", np.nan)
    recent_strict_wape = best_cv_row.get("Recent_Strict_WAPE", np.nan)
    recent_strict_r2 = best_cv_row.get("Recent_Strict_R2", np.nan)
else:
    recent_rmse = recent_strict_wape = recent_strict_r2 = np.nan

top = st.columns(3)
top[0].metric("Deployment model", best_model_name)
top[1].metric("Recent Strict WAPE", f"{recent_strict_wape:.2f}%" if not pd.isna(recent_strict_wape) else "N/A")
top[2].metric("Recent Strict R2", f"{recent_strict_r2:.3f}" if not pd.isna(recent_strict_r2) else "N/A")

tabs = st.tabs(["Leaderboard", "Prediction Fit", "Error Anatomy", "Time CV", "Failure Cases"])

with tabs[0]:
    st.subheader("All-holdout leaderboard")
    st.dataframe(
        leaderboard_df.style.format(
            {"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2": "{:.3f}", "WAPE": "{:.1f}", "Mean_RMSE": "{:,.0f}", "Mean_R2": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_compare = px.bar(
        metric_compare,
        x="Slice",
        y="R2",
        color="Slice",
        template="plotly_white",
        color_discrete_sequence=["#a63d40", "#1c6e8c", "#d9cab3"],
    )
    fig_compare.update_layout(showlegend=False)
    st.plotly_chart(fig_compare, use_container_width=True)

with tabs[1]:
    strict_holdout = holdout[holdout["strict_holdout_proxy"] == 1].copy()
    st.subheader("All Holdout Fit")
    st.plotly_chart(plot_actual_vs_pred(holdout, "pred_revenue"), use_container_width=True)

    holdout["abs_error"] = np.abs(holdout["revenue"] - holdout["pred_revenue"])
    holdout["ape_gt_1m"] = np.where(holdout["revenue"] > 1e6, holdout["abs_error"] / holdout["revenue"], np.nan)
    fig_error = px.histogram(
        holdout.dropna(subset=["ape_gt_1m"]),
        x="ape_gt_1m",
        nbins=40,
        template="plotly_white",
        color_discrete_sequence=["#a63d40"],
    )
    fig_error.update_layout(xaxis_title="Absolute percentage error for titles above $1M", yaxis_title="Film count")
    st.plotly_chart(fig_error, use_container_width=True)

with tabs[2]:
    left, right = st.columns(2, gap="large")
    left.subheader("Budget-band errors")
    left.dataframe(
        compute_budget_band_errors(holdout, "pred_revenue").style.format(
            {"mae": "{:,.0f}", "median_ape_gt_1m": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    right.subheader("Strict-slice budget-band errors")
    right.dataframe(
        compute_budget_band_errors(strict_holdout, "pred_revenue").style.format(
            {"mae": "{:,.0f}", "median_ape_gt_1m": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[3]:
    st.subheader("Rolling time-based validation")
    if cv_summary.empty:
        st.info("Run `python scripts/train_box_office_models.py` to generate rolling time-series CV outputs.")
    else:
        st.dataframe(
            cv_summary.style.format(
                {
                    "Mean_RMSE": "{:,.0f}",
                    "Std_RMSE": "{:,.0f}",
                    "Mean_MAE": "{:,.0f}",
                    "Mean_R2": "{:.3f}",
                    "Mean_WAPE": "{:.1f}",
                    "Mean_Strict_R2": "{:.3f}",
                    "Mean_Strict_WAPE": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        fold_view = cv_results[["Fold", "Model", "RMSE", "R2", "Strict_R2", "HoldoutStart", "HoldoutEnd"]].copy()
        st.dataframe(
            fold_view.style.format({"RMSE": "{:,.0f}", "R2": "{:.3f}", "Strict_R2": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

        fig_cv = px.line(
            cv_results,
            x="Fold",
            y="R2",
            color="Model",
            markers=True,
            template="plotly_white",
        )
        fig_cv.update_layout(yaxis_title="Fold R2", xaxis_title="Time fold")
        st.plotly_chart(fig_cv, use_container_width=True)

st.caption(
    "The holdout leader is strong enough to beat naive baselines comfortably, but the biggest misses are still concentrated in runaway hits. That means ranking and directionality are useful today, while extreme-tail calibration still has room to improve."
)

with tabs[4]:
    st.subheader("Largest under-predictions")
    st.dataframe(top_under_predictions(holdout, "pred_revenue"), use_container_width=True, hide_index=True)
    st.subheader("Largest over-predictions")
    st.dataframe(top_over_predictions(holdout, "pred_revenue"), use_container_width=True, hide_index=True)
