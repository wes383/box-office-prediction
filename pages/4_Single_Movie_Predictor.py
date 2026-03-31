from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_utils import (
    build_prediction_result,
    build_reference_catalogs,
    configure_page,
    describe_prediction_context,
    extract_director,
    extract_writer,
    format_money,
    hero,
    load_datasets,
    load_models,
    parse_cast_list,
    parse_countries,
    parse_companies,
    season_from_month,
)


configure_page("Single Movie Predictor", icon="film")

data = load_datasets()
models = load_models()
features = data["features"].copy()
catalogs = build_reference_catalogs(features)


def build_preset_catalog(df: pd.DataFrame) -> pd.DataFrame:
    preset_df = (
        df.loc[
            :,
            [
                "id",
                "title",
                "release_date",
                "budget",
                "runtime",
                "genres",
                "primary_genre",
                "original_language",
                "production_countries",
                "belongs_to_collection",
                "production_companies",
                "crew",
                "cast",
                "keywords",
            ],
        ]
        .copy()
        .sort_values(["title", "release_date"])
    )
    preset_df["release_year"] = pd.to_datetime(preset_df["release_date"]).dt.year
    preset_df["preset_label"] = (
        preset_df["title"].fillna("Untitled")
        + " ("
        + preset_df["release_year"].fillna(0).astype(int).astype(str)
        + ")"
    )
    return preset_df.drop_duplicates(subset=["preset_label"], keep="last").reset_index(drop=True)


def parse_preset_genres(genres_value: object, fallback: object, allowed_genres: list[str]) -> list[str]:
    values: list[str] = []
    if pd.notna(genres_value):
        for item in str(genres_value).split("|"):
            cleaned = item.strip()
            if cleaned and cleaned in allowed_genres:
                values.append(cleaned)
    if not values and pd.notna(fallback):
        cleaned = str(fallback).strip()
        if cleaned and cleaned in allowed_genres:
            values.append(cleaned)
    return list(dict.fromkeys(values))


def parse_preset_language(language_value: object, allowed_languages: list[str]) -> str:
    if pd.notna(language_value):
        raw = str(language_value)
        for splitter in ["|", ",", "/"]:
            raw = raw.replace(splitter, "|")
        for item in raw.split("|"):
            cleaned = item.strip()
            if cleaned and cleaned in allowed_languages:
                return cleaned
    return ""


def parse_language_text(language_value: str) -> str:
    return str(language_value or "").strip()


def is_collection_value(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text not in ("", "None", "nan", "NaN")


hero(
    "Single Movie Predictor",
    "This page scores a proposed release with a single deployment model plus comparable-title context, so the output stays decision-ready without pretending to be more precise than the data supports.",
)

presets = build_preset_catalog(features)
selected_preset_label = st.selectbox(
    "Choose a preset movie from the database",
    options=["Custom entry"] + presets["preset_label"].tolist(),
    index=0,
)

selected_preset = None
if selected_preset_label != "Custom entry":
    selected_preset = presets.loc[presets["preset_label"] == selected_preset_label].iloc[0]

default_title = selected_preset["title"] if selected_preset is not None else "Untitled Release"
default_budget_m = float(selected_preset["budget"]) / 1e6 if selected_preset is not None else 40.0
default_runtime = int(selected_preset["runtime"]) if selected_preset is not None else 110
default_release_date = (
    pd.Timestamp(selected_preset["release_date"]).date() if selected_preset is not None else pd.Timestamp("2026-07-10").date()
)
default_primary_genres = (
    parse_preset_genres(selected_preset["genres"], selected_preset["primary_genre"], catalogs["primary_genres"])
    if selected_preset is not None
    else [catalogs["primary_genres"][0]]
)
default_language = (
    parse_preset_language(selected_preset["original_language"], catalogs["languages"])
    if selected_preset is not None
    else catalogs["languages"][0]
)
if not default_language:
    default_language = catalogs["languages"][0]
default_is_sequel = is_collection_value(selected_preset["belongs_to_collection"]) if selected_preset is not None else False
default_collection_name = (
    str(selected_preset["belongs_to_collection"]).strip()
    if selected_preset is not None and is_collection_value(selected_preset["belongs_to_collection"])
    else ""
)
default_countries = parse_countries(selected_preset["production_countries"]) if selected_preset is not None else []
default_companies = parse_companies(selected_preset["production_companies"]) if selected_preset is not None else []
default_director = extract_director(selected_preset["crew"]) if selected_preset is not None else "Unknown"
default_cast_members = parse_cast_list(selected_preset["cast"], top_n=25) if selected_preset is not None else []
default_writer = extract_writer(selected_preset["crew"]) if selected_preset is not None else "Unknown"

preset_signature = selected_preset_label
if st.session_state.get("predictor_last_preset") != preset_signature:
    st.session_state["predictor_last_preset"] = preset_signature
    st.session_state["predictor_title"] = default_title
    st.session_state["predictor_budget_m"] = float(default_budget_m)
    st.session_state["predictor_runtime"] = int(default_runtime)
    st.session_state["predictor_release_date"] = default_release_date
    st.session_state["predictor_primary_genres"] = default_primary_genres
    st.session_state["predictor_language_text"] = default_language
    st.session_state["predictor_is_sequel"] = default_is_sequel
    st.session_state["predictor_collection_name"] = default_collection_name
    st.session_state["predictor_countries"] = ", ".join(default_countries) if default_countries else ""
    st.session_state["predictor_companies"] = ", ".join(default_companies) if default_companies else ""
    st.session_state["predictor_director"] = default_director if default_director else "Unknown"
    st.session_state["predictor_writer"] = default_writer if default_writer else "Unknown"
    st.session_state["predictor_cast_members"] = ", ".join(default_cast_members) if default_cast_members else ""
    st.session_state["predictor_keywords"] = str(selected_preset["keywords"]) if selected_preset is not None and pd.notna(selected_preset.get("keywords")) else ""

with st.form("movie_predictor_form"):
    left, right = st.columns(2, gap="large")

    with left:
        title = st.text_input("Title", key="predictor_title")
        budget_m = st.number_input("Budget (USD, millions)", min_value=0.0, step=5.0, key="predictor_budget_m")
        runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=300, step=5, key="predictor_runtime")
        release_date = st.date_input("Release date", key="predictor_release_date")
        primary_genres = st.multiselect(
            "Primary genre(s)",
            options=catalogs["primary_genres"],
            key="predictor_primary_genres",
            help="You can choose one or more genres.",
        )
        language_text = st.text_input(
            "Original language",
            key="predictor_language_text",
            help="Type a single language code, e.g. 'en' or 'ja'.",
        )

    with right:
        is_sequel = st.toggle("Part of a collection / sequel", key="predictor_is_sequel")
        collection_name = st.text_input(
            "Collection / franchise (optional)",
            key="predictor_collection_name",
            help="Use the known franchise or collection name when applicable, e.g. 'Marvel Cinematic Universe' or 'Toy Story Collection'.",
        )
        countries_text = st.text_input(
            "Production countries (comma-separated names or ISO codes)",
            key="predictor_countries",
            help="Type country names or ISO codes separated by commas, e.g. 'United States, United Kingdom' or 'US, GB'.",
        )
        companies_text = st.text_input(
            "Production companies (comma-separated)",
            key="predictor_companies",
            help="Type company names separated by commas, e.g. 'Warner Bros., Legendary Pictures'",
        )
        director_text = st.text_input(
            "Director (comma-separated for multiple)",
            key="predictor_director",
            help="Type director names separated by commas, e.g. 'Joel Coen, Ethan Coen'",
        )
        writer_text = st.text_input(
            "Writer (comma-separated for multiple)",
            key="predictor_writer",
            help="Type writer names separated by commas, e.g. 'Jonathan Nolan, Christopher Nolan'",
        )
        cast_text = st.text_input(
            "Lead cast (comma-separated)",
            key="predictor_cast_members",
            help="Type actor names separated by commas, e.g. 'Leonardo DiCaprio, Tom Hardy'",
        )
        keywords_text = st.text_input(
            "Keywords (pipe-separated)",
            key="predictor_keywords",
            help="Type keywords separated by pipes, e.g. 'superhero|sequel|space travel'",
        )
        submitted = st.form_submit_button("Predict box office", use_container_width=True)

if submitted:
    language = parse_language_text(language_text)
    countries = [c.strip().upper() for c in countries_text.split(",") if c.strip()] if countries_text.strip() else []
    companies = [c.strip() for c in companies_text.split(",") if c.strip()] if companies_text.strip() else []
    cast_members = [c.strip() for c in cast_text.split(",") if c.strip()] if cast_text.strip() else []
    directors = [d.strip() for d in director_text.split(",") if d.strip()] if director_text.strip() else ["Unknown"]
    writers = [w.strip() for w in writer_text.split(",") if w.strip()] if writer_text.strip() else ["Unknown"]
    keywords_text = st.session_state.get("predictor_keywords", "")
    release_ts = pd.Timestamp(release_date)
    result = build_prediction_result(
        title=title,
        budget=budget_m * 1e6,
        runtime=float(runtime),
        release_date=release_ts,
        primary_genres=primary_genres or default_primary_genres,
        languages=language or default_language,
        is_sequel=is_sequel,
        collection_name=collection_name.strip() or None,
        countries=countries,
        companies=companies,
        directors=directors,
        writers=writers,
        cast_members=cast_members,
        features_df=features,
        models=models,
        keywords=keywords_text,
    )

    top = st.columns(3)
    top[0].metric("Primary estimate", format_money(result.point_estimate))
    top[1].metric("Comparable range", f"{format_money(result.comp_low)} to {format_money(result.comp_high)}")
    top[2].metric("Estimated season", season_from_month(release_ts.month))

    st.info(describe_prediction_context(result))
    st.caption(
        f"Primary estimate currently comes from the selected deployment model: `{result.model_outputs['deployment_model_name']}`."
    )

    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Comparable P25", "Comparable Median", "Point Estimate", "Comparable P75"],
                y=[
                    result.comp_low,
                    result.comp_mid,
                    result.point_estimate,
                    result.comp_high,
                ],
                marker_color=["#d9cab3", "#c4a16e", "#a63d40", "#85a8b8"],
            )
        )
        fig.update_layout(template="plotly_white", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Model feature snapshot")
        st.dataframe(
            result.feature_row.T.rename(columns={0: "value"}).reset_index().rename(columns={"index": "feature"}),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Comparable titles")
    comparable_display = result.comparables.copy()
    comparable_display["budget"] = comparable_display["budget_raw"].fillna(comparable_display["budget"])
    comparable_display["revenue"] = comparable_display["revenue_raw"].fillna(comparable_display["revenue"])
    comparable_display = comparable_display.drop(columns=["budget_raw", "revenue_raw"], errors="ignore")
    st.dataframe(
        comparable_display.assign(
            budget=lambda df: df["budget"].map(format_money),
            revenue=lambda df: df["revenue"].map(format_money),
            company_hist_revenue=lambda df: df["company_hist_revenue"].map(format_money),
            director_hist_revenue=lambda df: df["director_hist_revenue"].map(format_money),
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Model outputs")
    all_model_outputs = result.model_outputs
    model_outputs = pd.DataFrame(
        [
            {"Estimate": all_model_outputs["deployment_model_name"], "Source": "Current selected deployment model", "Revenue": all_model_outputs["deployment_pred"]},
            {"Estimate": "LightGBM(two-stage)", "Source": "Saved LightGBM two-stage artifact", "Revenue": all_model_outputs["broad_lgb"]},
            {"Estimate": "Ensemble(two-stage)", "Source": "50/50 average of LightGBM and XGBoost two-stage models", "Revenue": all_model_outputs["broad_ensemble"]},
            {"Estimate": "XGBoost(two-stage)", "Source": "Saved XGBoost two-stage artifact", "Revenue": all_model_outputs["broad_xgb"]},
            {"Estimate": "Comparable low", "Source": "Historical comparable titles P25", "Revenue": result.comp_low},
            {"Estimate": "Comparable median", "Source": "Historical comparable titles P50", "Revenue": result.comp_mid},
            {"Estimate": "Comparable high", "Source": "Historical comparable titles P75", "Revenue": result.comp_high},
        ]
    )
    st.dataframe(
        model_outputs.assign(Revenue=lambda df: df["Revenue"].map(format_money)),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("Choose a preset movie to auto-fill the form, or keep `Custom entry` and enter a new title from scratch.")
