from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import (
    configure_page,
    hero,
    load_datasets,
    plot_budget_vs_revenue,
    plot_revenue_histogram,
)


configure_page("EDA Explorer", icon="film")

data = load_datasets()
eda = data["eda"].copy()
features = data["features"].copy()
eda["release_year"] = pd.to_datetime(eda["release_date"]).dt.year

BUDGET_LABEL_MAP = {
    "Low (<10M)": "Low (<10M)",
    "Mid (10-50M)": "Mid (10-50M)",
    "High (50-100M)": "High (50-100M)",
    "Blockbuster (>100M)": "Blockbuster (>100M)",
}
SEASON_LABEL_MAP = {
    "Holiday": "Holiday",
    "Summer": "Summer",
    "Spring": "Spring",
    "Winter": "Winter",
}

hero(
    "EDA Explorer",
    "Use filters to move from broad market structure to a narrower release cohort. This page is built to answer which slices are more lucrative, more crowded, and more skewed by blockbuster outliers.",
)

with st.sidebar:
    st.header("Filters")
    year_min = int(eda["release_year"].min())
    year_max = int(eda["release_year"].max())
    year_range = st.slider("Release year", min_value=year_min, max_value=year_max, value=(max(year_min, year_max - 15), year_max))
    selected_genres = st.multiselect(
        "Primary genre",
        options=sorted(eda["primary_genre"].dropna().unique().tolist()),
        default=[],
    )
    selected_languages = st.multiselect(
        "Language",
        options=sorted(eda["original_language"].dropna().unique().tolist())[:30],
        default=["en"],
    )
    only_proxy = st.toggle("Only theatrical proxy", value=False)
    exclude_zero_revenue = st.toggle("Exclude zero revenue rows", value=False)

filtered = eda[(eda["release_year"] >= year_range[0]) & (eda["release_year"] <= year_range[1])].copy()
if selected_genres:
    filtered = filtered[filtered["primary_genre"].isin(selected_genres)]
if selected_languages:
    filtered = filtered[filtered["original_language"].isin(selected_languages)]
if only_proxy:
    filtered = filtered[filtered["is_conservative_theatrical_proxy"] == 1]
if exclude_zero_revenue:
    filtered = filtered[filtered["revenue"] > 0]

if "budget_category" in filtered.columns:
    filtered["budget_category"] = filtered["budget_category"].replace(BUDGET_LABEL_MAP)
if "season" in filtered.columns:
    filtered["season"] = filtered["season"].replace(SEASON_LABEL_MAP)

if "is_sequel" not in filtered.columns:
    filtered["is_sequel"] = (
        filtered["belongs_to_collection"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

competition_cols = ["id", "past_competition_index", "num_past_competitors"]
missing_competition = [col for col in competition_cols if col not in filtered.columns]
if missing_competition:
    available_competition = [col for col in competition_cols if col in features.columns]
    filtered = filtered.merge(
        features[available_competition].drop_duplicates(subset=["id"]),
        on="id",
        how="left",
        suffixes=("", "_feature"),
    )

summary_cols = st.columns(4)
summary_cols[0].metric("Films in view", f"{len(filtered):,}")
summary_cols[1].metric("Median revenue", f"${filtered['revenue'].median() / 1e6:.2f}M" if len(filtered) else "N/A")
summary_cols[2].metric("Median budget", f"${filtered['budget'].median() / 1e6:.2f}M" if len(filtered) else "N/A")
summary_cols[3].metric("Zero revenue share", f"{(filtered['revenue'] <= 0).mean() * 100:.1f}%" if len(filtered) else "N/A")

tab1, tab2, tab3 = st.tabs(["Distribution", "Seasonality", "Economics"])

with tab1:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.plotly_chart(plot_revenue_histogram(filtered), use_container_width=True)
    with right:
        genre_stats = (
            filtered.groupby("primary_genre", observed=False)
            .agg(films=("title", "count"), mean_revenue=("revenue", "mean"), median_revenue=("revenue", "median"))
            .sort_values("mean_revenue", ascending=False)
            .head(12)
            .reset_index()
        )
        fig_genre = px.bar(
            genre_stats,
            x="mean_revenue",
            y="primary_genre",
            orientation="h",
            template="plotly_white",
            color="median_revenue",
            color_continuous_scale=["#d9cab3", "#1c6e8c"],
        )
        fig_genre.update_layout(coloraxis_colorbar_title="Median revenue", xaxis_title="Mean revenue")
        st.plotly_chart(fig_genre, use_container_width=True)
    zero_revenue_count = int((filtered["revenue"] <= 0).sum())

    lower_left, lower_right = st.columns(2, gap="large")
    with lower_left:
        roi_df = filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=["ROI"]).copy()
        roi_df = roi_df[(roi_df["ROI"] >= roi_df["ROI"].quantile(0.01)) & (roi_df["ROI"] <= roi_df["ROI"].quantile(0.99))]
        fig_roi = px.histogram(
            roi_df,
            x="ROI",
            nbins=40,
            template="plotly_white",
            color_discrete_sequence=["#a63d40"],
        )
        fig_roi.update_layout(xaxis_title="ROI (%)", yaxis_title="Film count", title="ROI distribution")
        st.plotly_chart(fig_roi, use_container_width=True)
    with lower_right:
        rating_df = filtered.dropna(subset=["averageRating"]).copy()
        rating_df["rating_bucket_fine"] = pd.cut(
            rating_df["averageRating"],
            bins=[0, 4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 10],
            labels=["<=4.0", "4.0-5.0", "5.0-5.5", "5.5-6.0", "6.0-6.5", "6.5-7.0", "7.0-7.5", "7.5-8.0", ">8.0"],
            include_lowest=True,
        )
        rating_stats = (
            rating_df.groupby("rating_bucket_fine", observed=False)
            .agg(films=("title", "count"), mean_revenue=("revenue", "mean"), median_revenue=("revenue", "median"))
            .reset_index()
        )
        fig_rating = px.bar(
            rating_stats,
            x="rating_bucket_fine",
            y="mean_revenue",
            color="films",
            template="plotly_white",
            color_continuous_scale=["#d9cab3", "#1c6e8c"],
            title="Revenue by IMDb Rating Bucket",
        )
        fig_rating.update_layout(xaxis_title="IMDb rating bucket", yaxis_title="Mean revenue")
        st.plotly_chart(fig_rating, use_container_width=True)

    bottom_left, bottom_right = st.columns(2, gap="large")
    with bottom_left:
        genre_box_df = filtered[(filtered["revenue"] > 0) & filtered["primary_genre"].notna()].copy()
        top_genres = genre_box_df["primary_genre"].value_counts().head(8).index.tolist()
        genre_box_df = genre_box_df[genre_box_df["primary_genre"].isin(top_genres)]
        fig_genre_box = px.box(
            genre_box_df,
            x="primary_genre",
            y="revenue",
            color="primary_genre",
            template="plotly_white",
            title="Revenue distribution by top genres",
        )
        fig_genre_box.update_yaxes(type="log", title="Revenue (log scale)")
        fig_genre_box.update_layout(showlegend=False, xaxis_title="Primary genre")
        st.plotly_chart(fig_genre_box, use_container_width=True)
    with bottom_right:
        sequel_stats = (
            filtered.assign(sequel_group=filtered["is_sequel"].fillna(0).astype(int).map({0: "Non-sequel", 1: "Sequel"}))
            .groupby("sequel_group", observed=False)
            .agg(films=("title", "count"), mean_revenue=("revenue", "mean"), median_revenue=("revenue", "median"))
            .reset_index()
        )
        fig_sequel = px.bar(
            sequel_stats,
            x="sequel_group",
            y="mean_revenue",
            color="films",
            template="plotly_white",
            color_continuous_scale=["#d9cab3", "#a63d40"],
            title="Sequel vs non-sequel revenue",
        )
        fig_sequel.update_layout(xaxis_title="Release type", yaxis_title="Mean revenue")
        st.plotly_chart(fig_sequel, use_container_width=True)

with tab2:
    month_stats = (
        filtered.groupby("release_month", observed=False)
        .agg(mean_revenue=("revenue", "mean"), median_revenue=("revenue", "median"), films=("title", "count"))
        .reset_index()
    )
    season_stats = (
        filtered.groupby("season", observed=False)
        .agg(mean_revenue=("revenue", "mean"), median_revenue=("revenue", "median"), films=("title", "count"))
        .reset_index()
    )
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(
        px.line(
            month_stats,
            x="release_month",
            y=["mean_revenue", "median_revenue"],
            markers=True,
            template="plotly_white",
        ),
        use_container_width=True,
    )
    c2.plotly_chart(
        px.bar(
            season_stats.sort_values("mean_revenue", ascending=False),
            x="season",
            y="mean_revenue",
            color="films",
            template="plotly_white",
            color_continuous_scale=["#d9cab3", "#a63d40"],
        ),
        use_container_width=True,
    )

    lower_left, lower_right = st.columns(2, gap="large")
    year_stats = (
        filtered.groupby("release_year", observed=False)
        .agg(mean_revenue=("revenue", "mean"), median_revenue=("revenue", "median"), films=("title", "count"))
        .reset_index()
        .sort_values("release_year")
    )
    weekday_stats = (
        filtered.groupby("release_dayofweek", observed=False)
        .agg(mean_revenue=("revenue", "mean"), median_revenue=("revenue", "median"), films=("title", "count"))
        .reset_index()
    )
    weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    weekday_stats["weekday"] = weekday_stats["release_dayofweek"].map(weekday_map)

    lower_left.plotly_chart(
        px.line(
            year_stats,
            x="release_year",
            y=["mean_revenue", "median_revenue"],
            template="plotly_white",
            markers=True,
            title="Revenue trend by release year",
        ),
        use_container_width=True,
    )
    lower_right.plotly_chart(
        px.bar(
            weekday_stats,
            x="weekday",
            y="mean_revenue",
            color="films",
            template="plotly_white",
            color_continuous_scale=["#d9cab3", "#a63d40"],
            title="Revenue by release weekday",
        ),
        use_container_width=True,
    )

    bottom_left, bottom_right = st.columns(2, gap="large")
    comp_df = filtered[filtered["past_competition_index"].notna()].copy()
    if len(comp_df):
        comp_df["competition_band"] = pd.qcut(
            comp_df["past_competition_index"].rank(method="first"),
            q=4,
            labels=["Low", "Mid-low", "Mid-high", "High"],
        )
        comp_stats = (
            comp_df.groupby("competition_band", observed=False)
            .agg(films=("title", "count"), median_revenue=("revenue", "median"), mean_comp=("past_competition_index", "mean"))
            .reset_index()
        )
        bottom_left.plotly_chart(
            px.bar(
                comp_stats,
                x="competition_band",
                y="median_revenue",
                color="films",
                template="plotly_white",
                color_continuous_scale=["#d9cab3", "#1c6e8c"],
                title="Median revenue by competition level",
            ),
            use_container_width=True,
        )
        bottom_right.plotly_chart(
            px.scatter(
                comp_df.sample(min(len(comp_df), 3000), random_state=42),
                x="past_competition_index",
                y="revenue",
                color="primary_genre",
                hover_name="title",
                template="plotly_white",
                opacity=0.55,
                title="Competition vs revenue",
            ).update_yaxes(type="log", title="Revenue (log scale)"),
            use_container_width=True,
        )

with tab3:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.plotly_chart(plot_budget_vs_revenue(filtered), use_container_width=True)
    with right:
        company_stats = (
            filtered.assign(primary_company=filtered["production_companies"].fillna("Unknown").str.split("|").str[0])
            .groupby("primary_company", observed=False)
            .agg(films=("title", "count"), mean_revenue=("revenue", "mean"))
            .query("films >= 10")
            .sort_values("mean_revenue", ascending=False)
            .head(12)
            .reset_index()
        )
        fig_company = px.bar(
            company_stats,
            x="mean_revenue",
            y="primary_company",
            orientation="h",
            template="plotly_white",
            color="films",
            color_continuous_scale=["#d9cab3", "#1c6e8c"],
        )
        fig_company.update_layout(yaxis_title="Primary company", xaxis_title="Mean revenue")
        st.plotly_chart(fig_company, use_container_width=True)

    lower_left, lower_right = st.columns(2, gap="large")
    runtime_df = filtered[filtered["runtime"] > 0].copy()
    lower_left.plotly_chart(
        px.scatter(
            runtime_df,
            x="runtime",
            y="revenue",
            color="primary_genre",
            hover_name="title",
            template="plotly_white",
            opacity=0.55,
            title="Runtime vs revenue",
        ).update_yaxes(type="log", title="Revenue (log scale)"),
        use_container_width=True,
    )

    language_stats = (
        filtered.groupby("original_language", observed=False)
        .agg(films=("title", "count"), mean_revenue=("revenue", "mean"))
        .sort_values("films", ascending=False)
        .head(12)
        .reset_index()
    )
    lower_right.plotly_chart(
        px.bar(
            language_stats,
            x="original_language",
            y="films",
            color="mean_revenue",
            template="plotly_white",
            color_continuous_scale=["#d9cab3", "#1c6e8c"],
            title="Top languages by film count and mean revenue",
        ),
        use_container_width=True,
    )

    bottom_left, bottom_right = st.columns(2, gap="large")
    roi_budget_df = filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=["ROI", "budget_category"]).copy()
    roi_budget_df = roi_budget_df[
        (roi_budget_df["ROI"] >= roi_budget_df["ROI"].quantile(0.01))
        & (roi_budget_df["ROI"] <= roi_budget_df["ROI"].quantile(0.99))
    ]
    bottom_left.plotly_chart(
        px.box(
            roi_budget_df,
            x="budget_category",
            y="ROI",
            color="budget_category",
            template="plotly_white",
            title="ROI distribution by budget band",
        ).update_layout(showlegend=False, xaxis_title="Budget band", yaxis_title="ROI (%)"),
        use_container_width=True,
    )

    english_stats = (
        filtered.assign(language_group=filtered["original_language"].eq("en").map({True: "English", False: "Non-English"}))
        .groupby("language_group", observed=False)
        .agg(films=("title", "count"), mean_revenue=("revenue", "mean"), median_revenue=("revenue", "median"))
        .reset_index()
    )
    bottom_right.plotly_chart(
        px.bar(
            english_stats,
            x="language_group",
            y="mean_revenue",
            color="films",
            template="plotly_white",
            color_continuous_scale=["#d9cab3", "#1c6e8c"],
            title="English vs non-English revenue",
        ).update_layout(xaxis_title="Language group", yaxis_title="Mean revenue"),
        use_container_width=True,
    )

st.subheader("Filtered dataset preview")
preview_cols = ["title", "release_date", "primary_genre", "budget", "revenue", "averageRating", "season"]
st.dataframe(filtered[preview_cols].sort_values("release_date", ascending=False).head(200), use_container_width=True, hide_index=True)
