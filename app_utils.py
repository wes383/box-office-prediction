from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

MODEL_FEATURES = [
    "log_budget",
    "log_runtime",
    "budget_per_minute",
    "log_budget_per_minute",
    "release_year",
    "release_month",
    "release_quarter",
    "release_month_sin",
    "release_month_cos",
    "post_covid_flag",
    "is_summer",
    "is_holiday",
    "is_friday",
    "is_month_end",
    "is_weekend_release",
    "budget_imputed_flag",
    "budget_was_zero_raw",
    "runtime_is_flagged",
    "cast_avg_revenue",
    "cast_max_revenue",
    "cast_weighted_revenue",
    "cast_median_revenue",
    "cast_log_mean_revenue",
    "cast_recent_weighted_revenue",
    "cast_recent_hit_rate",
    "cast_recent_blockbuster_rate",
    "company_hist_revenue",
    "company_success_rate",
    "country_count",
    "is_us_production",
    "is_china_production",
    "is_uk_production",
    "is_france_production",
    "is_india_production",
    "is_japan_production",
    "is_korea_production",
    "country_hist_revenue",
    "country_hit_rate",
    "country_recent_revenue",
    "company_recent_revenue",
    "company_recent_success_rate",
    "company_recent_blockbuster_rate",
    "director_hist_revenue",
    "director_recent_revenue",
    "director_recent_hit_rate",
    "director_recent_max_revenue",
    "writer_hist_revenue",
    "writer_recent_revenue",
    "writer_recent_hit_rate",
    "writer_recent_max_revenue",
    "num_past_competitors",
    "past_competition_index",
    "same_genre_past_comp_count",
    "same_genre_past_comp_index",
    "is_sequel",
    "has_sequel_keyword",
    "franchise_count_prior",
    "franchise_mean_revenue_prior",
    "franchise_max_revenue_prior",
    "franchise_latest_revenue_prior",
    "franchise_hit_rate_prior",
    "franchise_recent_mean_revenue",
    "franchise_recent_max_revenue",
    "franchise_recency_years",
    "franchise_active_gap_flag",
    "ip_title_match_count_prior",
    "ip_title_match_mean_revenue_prior",
    "ip_title_match_max_revenue_prior",
    "ip_title_match_recent_max_revenue",
    "known_ip_proxy",
    "is_english",
    "title_length",
    "title_word_count",
    "genre_action",
    "genre_adventure",
    "genre_animation",
    "genre_comedy",
    "genre_crime",
    "genre_drama",
    "genre_family",
    "genre_fantasy",
    "genre_horror",
    "genre_music",
    "genre_mystery",
    "genre_romance",
    "genre_science_fiction",
    "genre_thriller",
    "genre_war",
    "genre_western",
    "genre_history",
    "genre_documentary",
    "keyword_count",
    "kw_based_on_novel_or_book",
    "kw_woman_director",
    "kw_murder",
    "kw_based_on_true_story",
    "kw_sequel",
    "kw_new_york_city",
    "kw_revenge",
    "kw_biography",
    "kw_duringcreditsstinger",
    "kw_friendship",
    "kw_black_and_white",
    "kw_love",
    "kw_christmas",
    "kw_lgbt",
    "kw_anime",
    "kw_serial_killer",
    "kw_remake",
    "kw_martial_arts",
    "kw_short_film",
    "kw_musical",
    "kw_based_on_comic",
    "kw_aftercreditsstinger",
    "kw_parent_child_relationship",
    "kw_high_school",
    "kw_police",
    "kw_coming_of_age",
    "kw_los_angeles_california",
    "kw_family",
    "kw_world_war_ii",
    "kw_gay_theme",
    "kw_superhero",
    "kw_space_travel",
    "kw_alien",
    "kw_war",
    "kw_zombie",
    "kw_heist",
    "kw_time_travel",
    "kw_artificial_intelligence",
    "kw_prison",
    "kw_vampire",
    "kw_dystopia",
    "kw_western",
    "kw_ocean",
    "kw_espionage",
    "kw_pirates",
    "kw_terrorist",
    "kw_conspiracy",
    "kw_sports",
    "kw_medical",
    "kw_drug",
    "kw_court",
    "kw_cult",
    "kw_dinosaur",
    "kw_robot",
    "kw_astronaut",
    "kw_soldier",
    "kw_detective",
    "kw_ninja",
    "kw_samurai",
    "kw_wizard",
    "kw_dragon",
    "kw_magic",
    "kw_superpower",
]

GENRE_FLAG_MAP = {
    "Action": "genre_action",
    "Adventure": "genre_adventure",
    "Animation": "genre_animation",
    "Comedy": "genre_comedy",
    "Crime": "genre_crime",
    "Drama": "genre_drama",
    "Family": "genre_family",
    "Fantasy": "genre_fantasy",
    "Horror": "genre_horror",
    "Music": "genre_music",
    "Mystery": "genre_mystery",
    "Romance": "genre_romance",
    "Science Fiction": "genre_science_fiction",
    "Thriller": "genre_thriller",
    "War": "genre_war",
    "Western": "genre_western",
    "History": "genre_history",
    "Documentary": "genre_documentary",
}

TITLE_TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "movie",
    "film",
    "story",
    "part",
    "chapter",
    "rise",
    "fall",
    "return",
    "returns",
    "last",
    "first",
}

TOP_COUNTRY_FLAGS = {
    "US": "is_us_production",
    "CN": "is_china_production",
    "GB": "is_uk_production",
    "FR": "is_france_production",
    "IN": "is_india_production",
    "JP": "is_japan_production",
    "KR": "is_korea_production",
}

COUNTRY_NAME_TO_ISO2 = {
    "UNITED STATES OF AMERICA": "US",
    "UNITED STATES": "US",
    "USA": "US",
    "US": "US",
    "U.S.": "US",
    "U.S.A.": "US",
    "CHINA": "CN",
    "CN": "CN",
    "UNITED KINGDOM": "GB",
    "UK": "GB",
    "GREAT BRITAIN": "GB",
    "BRITAIN": "GB",
    "ENGLAND": "GB",
    "GB": "GB",
    "FRANCE": "FR",
    "FR": "FR",
    "INDIA": "IN",
    "IN": "IN",
    "JAPAN": "JP",
    "JP": "JP",
    "SOUTH KOREA": "KR",
    "KOREA, REPUBLIC OF": "KR",
    "REPUBLIC OF KOREA": "KR",
    "KOREA": "KR",
    "KR": "KR",
    "HONG KONG": "HK",
    "HK": "HK",
    "TAIWAN": "TW",
    "TW": "TW",
    "GERMANY": "DE",
    "DE": "DE",
    "ITALY": "IT",
    "IT": "IT",
    "SPAIN": "ES",
    "ES": "ES",
    "CANADA": "CA",
    "CA": "CA",
    "AUSTRALIA": "AU",
    "AU": "AU",
    "MEXICO": "MX",
    "MX": "MX",
    "ARGENTINA": "AR",
    "AR": "AR",
    "BRAZIL": "BR",
    "BRASIL": "BR",
    "BR": "BR",
    "CHILE": "CL",
    "CL": "CL",
    "COLOMBIA": "CO",
    "CO": "CO",
    "PERU": "PE",
    "PE": "PE",
    "VENEZUELA": "VE",
    "VE": "VE",
    "RUSSIAN FEDERATION": "RU",
    "RUSSIA": "RU",
    "RU": "RU",
    "UKRAINE": "UA",
    "UA": "UA",
    "POLAND": "PL",
    "PL": "PL",
    "CZECH REPUBLIC": "CZ",
    "CZECHIA": "CZ",
    "CZ": "CZ",
    "SLOVAKIA": "SK",
    "SK": "SK",
    "HUNGARY": "HU",
    "HU": "HU",
    "ROMANIA": "RO",
    "RO": "RO",
    "BULGARIA": "BG",
    "BG": "BG",
    "SERBIA": "RS",
    "RS": "RS",
    "CROATIA": "HR",
    "HR": "HR",
    "SLOVENIA": "SI",
    "SI": "SI",
    "BOSNIA AND HERZEGOVINA": "BA",
    "BA": "BA",
    "MONTENEGRO": "ME",
    "ME": "ME",
    "NORTH MACEDONIA": "MK",
    "MACEDONIA": "MK",
    "MK": "MK",
    "GREECE": "GR",
    "GR": "GR",
    "TURKEY": "TR",
    "TURKIYE": "TR",
    "TR": "TR",
    "NETHERLANDS": "NL",
    "THE NETHERLANDS": "NL",
    "HOLLAND": "NL",
    "NL": "NL",
    "BELGIUM": "BE",
    "BE": "BE",
    "SWITZERLAND": "CH",
    "CH": "CH",
    "AUSTRIA": "AT",
    "AT": "AT",
    "DENMARK": "DK",
    "DK": "DK",
    "SWEDEN": "SE",
    "SE": "SE",
    "NORWAY": "NO",
    "NO": "NO",
    "FINLAND": "FI",
    "FI": "FI",
    "ICELAND": "IS",
    "IS": "IS",
    "IRELAND": "IE",
    "IE": "IE",
    "PORTUGAL": "PT",
    "PT": "PT",
    "LUXEMBOURG": "LU",
    "LU": "LU",
    "ESTONIA": "EE",
    "EE": "EE",
    "LATVIA": "LV",
    "LV": "LV",
    "LITHUANIA": "LT",
    "LT": "LT",
    "ISRAEL": "IL",
    "IL": "IL",
    "PALESTINE": "PS",
    "PALESTINIAN TERRITORY": "PS",
    "PS": "PS",
    "LEBANON": "LB",
    "LB": "LB",
    "JORDAN": "JO",
    "JO": "JO",
    "UNITED ARAB EMIRATES": "AE",
    "UAE": "AE",
    "AE": "AE",
    "SAUDI ARABIA": "SA",
    "SA": "SA",
    "QATAR": "QA",
    "QA": "QA",
    "KUWAIT": "KW",
    "KW": "KW",
    "EGYPT": "EG",
    "EG": "EG",
    "MOROCCO": "MA",
    "MA": "MA",
    "TUNISIA": "TN",
    "TN": "TN",
    "ALGERIA": "DZ",
    "DZ": "DZ",
    "SOUTH AFRICA": "ZA",
    "ZA": "ZA",
    "NIGERIA": "NG",
    "NG": "NG",
    "KENYA": "KE",
    "KE": "KE",
    "ETHIOPIA": "ET",
    "ET": "ET",
    "NEW ZEALAND": "NZ",
    "NZ": "NZ",
    "SINGAPORE": "SG",
    "SG": "SG",
    "MALAYSIA": "MY",
    "MY": "MY",
    "THAILAND": "TH",
    "TH": "TH",
    "INDONESIA": "ID",
    "ID": "ID",
    "PHILIPPINES": "PH",
    "PH": "PH",
    "VIETNAM": "VN",
    "VIET NAM": "VN",
    "VN": "VN",
    "CAMBODIA": "KH",
    "KH": "KH",
    "LAOS": "LA",
    "LA": "LA",
    "MONGOLIA": "MN",
    "MN": "MN",
    "PAKISTAN": "PK",
    "PK": "PK",
    "BANGLADESH": "BD",
    "BD": "BD",
    "SRI LANKA": "LK",
    "LK": "LK",
    "NEPAL": "NP",
    "NP": "NP",
    "IRAN": "IR",
    "IRAN, ISLAMIC REPUBLIC OF": "IR",
    "IR": "IR",
    "IRAQ": "IQ",
    "IQ": "IQ",
    "AFGHANISTAN": "AF",
    "AF": "AF",
    "KAZAKHSTAN": "KZ",
    "KZ": "KZ",
    "UZBEKISTAN": "UZ",
    "UZ": "UZ",
    "GEORGIA": "GE",
    "GE": "GE",
    "ARMENIA": "AM",
    "AM": "AM",
    "AZERBAIJAN": "AZ",
    "AZ": "AZ",
    "BELARUS": "BY",
    "BY": "BY",
    "MOLDOVA": "MD",
    "MD": "MD",
}


@dataclass
class PredictionResult:
    point_estimate: float
    broad_estimate: float
    comp_low: float
    comp_mid: float
    comp_high: float
    feature_row: pd.DataFrame
    comparables: pd.DataFrame
    model_outputs: Dict[str, float]


def configure_page(title: str, icon: str = "film") -> None:
    try:
        st.set_page_config(
            page_title=title,
            page_icon=icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, .stApp, 
        p, h1, h2, h3, h4, h5, h6, li, label, a,
        input, textarea, select, .stDataFrame, .js-plotly-plot text,
        button:not([data-testid*="Icon"]):not([class*="icon"]),
        div[class*="st-"]:not([data-testid*="Icon"]):not([class*="icon"]):not([class*="material"]),
        span[class*="st-"]:not([data-testid*="Icon"]):not([data-testid="stIconMaterial"]):not([class*="material"]) {
            font-family: 'Inter', sans-serif !important;
        }

        [data-testid="stIconMaterial"], span.material-symbols-rounded, .material-icons, [class*="material-symbols"] {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        }

        .block-container {
            padding-top: 4.5rem !important;
            padding-bottom: 2.5rem !important;
        }

        header[data-testid="stHeader"] {
            border-bottom: 1px solid rgba(128, 128, 128, 0.2) !important;
        }

        div[data-testid="stMetric"] {
            border-radius: 18px !important;
            padding: 1rem 1.1rem !important;
            border: 1px solid rgba(128, 128, 128, 0.2) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
        }
        
        div[data-testid="stMetricLabel"] > div {
            font-size: 0.76rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            opacity: 0.7;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            font-family: 'Inter', sans-serif !important;
        }
        div[data-testid="stMetricValue"] {
            font-weight: 800 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: break-word !important;
        }
        div[data-testid="stMetricValue"] > div,
        div[data-testid="stMetricValue"] p,
        div[data-testid="stMetricValue"] span {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: break-word !important;
        }

        .hero-card {
            border-radius: 24px;
            padding: 1.4rem 1.6rem;
            margin-top: 0.3rem;
            margin-bottom: 1.1rem;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        .hero-eyebrow {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-weight: 700;
            margin-bottom: 0.3rem;
            opacity: 0.7;
        }
        .hero-title {
            font-size: 2.0rem;
            line-height: 1.12;
            font-weight: 800;
            margin: 0.1rem 0 0.4rem 0;
        }
        .hero-copy {
            font-size: 0.97rem;
            line-height: 1.65;
            max-width: 72rem;
            opacity: 0.85;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, body: str, eyebrow: str = "Film Box Office Dashboard") -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.0f}K"
    return f"${value:,.0f}"


def metric_delta(current: float, baseline: float) -> str:
    if baseline == 0 or pd.isna(current) or pd.isna(baseline):
        return "N/A"
    delta = (current - baseline) / baseline * 100
    return f"{delta:+.1f}% vs baseline"


def _normalize_multi_select(value: str | Sequence[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    values = [str(item).strip() for item in value if str(item).strip()]
    return list(dict.fromkeys(values))


@st.cache_data(show_spinner=False)
def load_datasets() -> Dict[str, pd.DataFrame]:
    date_cols = ["release_date"]
    features = pd.read_csv(DATA_DIR / "tmdb_box_office_features.csv", parse_dates=date_cols)
    eda = pd.read_csv(DATA_DIR / "tmdb_box_office_eda.csv", parse_dates=date_cols)
    all_released = pd.read_csv(DATA_DIR / "tmdb_box_office_all_released.csv", parse_dates=date_cols)
    strict = pd.read_csv(DATA_DIR / "tmdb_box_office_strict_subset.csv", parse_dates=date_cols)
    theatrical = pd.read_csv(DATA_DIR / "tmdb_box_office_theatrical_proxy.csv", parse_dates=date_cols)
    holdout = pd.read_csv(MODELS_DIR / "holdout_predictions.csv", parse_dates=date_cols)
    holdout_strict = pd.read_csv(MODELS_DIR / "holdout_predictions_strict.csv", parse_dates=date_cols)
    model_results = pd.read_csv(MODELS_DIR / "model_results.csv")
    holdout_slice_metrics = pd.read_csv(MODELS_DIR / "holdout_slice_metrics.csv")
    cv_results_path = MODELS_DIR / "time_series_cv_results.csv"
    cv_summary_path = MODELS_DIR / "time_series_cv_summary.csv"
    cv_results = pd.read_csv(cv_results_path, parse_dates=["HoldoutStart", "HoldoutEnd"]) if cv_results_path.exists() else pd.DataFrame()
    cv_summary = pd.read_csv(cv_summary_path) if cv_summary_path.exists() else pd.DataFrame()

    return {
        "features": features,
        "eda": eda,
        "all_released": all_released,
        "strict": strict,
        "theatrical": theatrical,
        "holdout": holdout,
        "holdout_strict": holdout_strict,
        "model_results": model_results,
        "holdout_slice_metrics": holdout_slice_metrics,
        "cv_results": cv_results,
        "cv_summary": cv_summary,
    }


def select_model_row(model_results: pd.DataFrame) -> pd.Series:
    if model_results.empty:
        return pd.Series(dtype=object)
    if "SelectedForDeployment" in model_results.columns:
        selected = model_results[model_results["SelectedForDeployment"].fillna(False).astype(bool)]
        if not selected.empty:
            return selected.iloc[0]
    if "DeploymentRank" in model_results.columns:
        return model_results.sort_values(["DeploymentRank", "RMSE"], ascending=[True, True]).iloc[0]
    return model_results.sort_values("RMSE").iloc[0]


@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, object]:
    model_results = pd.read_csv(MODELS_DIR / "model_results.csv")
    deployment_model = select_model_row(model_results)
    deployment_name = str(deployment_model.get("Model", "LightGBM(two-stage)"))
    return {
        "feature_cols": joblib.load(MODELS_DIR / "feature_cols.pkl"),
        "lgb": joblib.load(MODELS_DIR / "lgb_model.pkl"),
        "xgb": joblib.load(MODELS_DIR / "xgb_model.pkl"),
        "deployment_model_name": deployment_name,
    }


def season_from_month(month: int) -> str:
    if month in [5, 6, 7, 8]:
        return "Summer"
    if month in [11, 12]:
        return "Holiday"
    if month in [2, 3]:
        return "Spring"
    return "Winter"


def parse_companies(companies_str: str) -> List[str]:
    if pd.isna(companies_str) or companies_str in ("Unknown", ""):
        return []
    return [item.strip() for item in str(companies_str).split("|") if item.strip()]


def parse_countries(countries_str: str) -> List[str]:
    if pd.isna(countries_str) or countries_str in ("Unknown", ""):
        return []
    values: List[str] = []
    for item in str(countries_str).split("|"):
        cleaned = item.strip().upper()
        if not cleaned:
            continue
        normalized = COUNTRY_NAME_TO_ISO2.get(cleaned, cleaned if len(cleaned) == 2 else cleaned)
        values.append(normalized)
    return list(dict.fromkeys(values))


def parse_cast_list(cast_str: str, top_n: int = 10) -> List[str]:
    if pd.isna(cast_str) or cast_str in ("Unknown", ""):
        return []
    actors: List[str] = []
    for actor_info in str(cast_str).split("|")[:top_n]:
        actor_info = actor_info.strip()
        if not actor_info:
            continue
        actor_name = actor_info.split("(")[0].strip()
        if actor_name:
            actors.append(actor_name)
    return actors


def extract_director(crew_str: str) -> str:
    if pd.isna(crew_str) or crew_str in ("Unknown", ""):
        return "Unknown"
    for member in str(crew_str).split("|"):
        member = member.strip()
        if member.startswith("Director:"):
            return member.replace("Director:", "").strip()
    return "Unknown"


def extract_writer(crew_str: str) -> str:
    if pd.isna(crew_str) or crew_str in ("Unknown", ""):
        return "Unknown"
    for member in str(crew_str).split("|"):
        member = member.strip()
        if member.startswith("Writer:"):
            writers_raw = member.replace("Writer:", "").strip()
            if writers_raw:
                return ", ".join(w.split("(")[0].strip() for w in writers_raw.split(",") if w.strip())
    return "Unknown"


def significant_title_tokens(title: object) -> List[str]:
    if pd.isna(title) or title in ("", None):
        return []
    tokens = re.findall(r"[A-Za-z0-9]+", str(title).lower())
    cleaned = [
        token
        for token in tokens
        if len(token) >= 3 and not token.isdigit() and token not in TITLE_TOKEN_STOPWORDS
    ]
    return list(dict.fromkeys(cleaned))


@st.cache_data(show_spinner=False)
def build_reference_catalogs(features: pd.DataFrame) -> Dict[str, List[str]]:
    companies = (
        features["production_companies"]
        .fillna("")
        .str.split("|")
        .explode()
        .str.strip()
    )
    companies = companies[companies.ne("")].value_counts().head(200).index.tolist()

    directors = features["director"].fillna("Unknown").value_counts().head(150).index.tolist()
    directors = [d for d in directors if d != "Unknown"]

    actors = (
        features["cast"]
        .fillna("")
        .apply(parse_cast_list)
        .explode()
        .dropna()
        .value_counts()
        .head(200)
        .index.tolist()
    )

    primary_genres = features["primary_genre"].fillna("Unknown").value_counts().index.tolist()
    languages = features["original_language"].fillna("unknown").value_counts().head(25).index.tolist()
    return {
        "companies": companies,
        "directors": directors,
        "actors": actors,
        "primary_genres": primary_genres,
        "languages": languages,
    }


def compute_company_director_history(
    history_df: pd.DataFrame,
    companies: Sequence[str],
    directors: Sequence[str],
    min_hist: int = 3,
) -> Dict[str, float]:
    comp_revs: List[float] = []
    comp_rates: List[float] = []
    for comp in companies:
        comp_hist = history_df[
            history_df["production_companies"].fillna("").str.contains(comp, regex=False)
        ]
        if len(comp_hist) < min_hist:
            continue
        comp_revs.append(float(comp_hist["revenue"].mean()))
        comp_rates.append(
            float(
                ((comp_hist["budget"] > 0) & (comp_hist["revenue"] > comp_hist["budget"] * 2)).mean()
            )
        )

    director_revs: List[float] = []
    for director in directors:
        if not director or director == "Unknown":
            continue
        director_hist = history_df[history_df["director"] == director]
        if len(director_hist) >= min_hist:
            director_revs.append(float(director_hist["revenue"].mean()))
    director_hist_revenue = float(np.mean(director_revs)) if director_revs else 0.0
    return {
        "company_hist_revenue": float(np.mean(comp_revs)) if comp_revs else 0.0,
        "company_success_rate": float(np.mean(comp_rates)) if comp_rates else 0.0,
        "director_hist_revenue": director_hist_revenue,
    }


def compute_country_history(
    history_df: pd.DataFrame,
    countries: Sequence[str],
    release_date: pd.Timestamp,
    recent_years: int = 3,
    min_hist: int = 2,
) -> Dict[str, float]:
    country_hist_revenues: List[float] = []
    country_recent_revenues: List[float] = []
    country_hit_rates: List[float] = []

    for country in countries:
        country_hist = history_df[
            history_df["production_countries"].fillna("").str.contains(country, regex=False)
        ][["release_date", "revenue"]]
        if len(country_hist) < min_hist:
            continue
        country_hist_revenues.append(float(country_hist["revenue"].mean()))
        country_hit_rates.append(float((country_hist["revenue"] >= 1e8).mean()))
        recent_hist = country_hist[
            (release_date - country_hist["release_date"]).dt.days <= int(365.25 * recent_years)
        ]
        if not recent_hist.empty:
            country_recent_revenues.append(float(recent_hist["revenue"].mean()))

    feature_map = {
        "country_count": float(len(countries)),
        "country_hist_revenue": float(np.mean(country_hist_revenues)) if country_hist_revenues else 0.0,
        "country_hit_rate": float(np.mean(country_hit_rates)) if country_hit_rates else 0.0,
        "country_recent_revenue": float(np.mean(country_recent_revenues)) if country_recent_revenues else 0.0,
    }
    for country_code, feature_name in TOP_COUNTRY_FLAGS.items():
        feature_map[feature_name] = float(country_code in countries)
    return feature_map


def compute_recent_hotness_features(
    history_df: pd.DataFrame,
    release_date: pd.Timestamp,
    title: str,
    companies: Sequence[str],
    directors: Sequence[str],
    writers: Sequence[str],
    cast_members: Sequence[str],
    collection_name: str | None,
    cast_recent_years: int = 3,
    company_recent_years: int = 3,
    director_recent_years: int = 5,
    writer_recent_years: int = 5,
    franchise_recent_years: int = 8,
) -> Dict[str, float]:
    recent_cast_scores: List[float] = []
    recent_cast_hits: List[float] = []
    recent_cast_blockbusters: List[float] = []
    for actor in cast_members:
        actor_hist = history_df[history_df["cast"].fillna("").str.contains(actor, regex=False)][["release_date", "revenue"]]
        if actor_hist.empty:
            continue
        actor_hist = actor_hist[
            (release_date - actor_hist["release_date"]).dt.days <= int(365.25 * cast_recent_years)
        ]
        if actor_hist.empty:
            continue
        weighted_sum = 0.0
        weight_sum = 0.0
        for hist_date, hist_revenue in actor_hist.itertuples(index=False):
            gap_years = max((release_date - hist_date).days / 365.25, 0.0)
            weight = 0.85 ** gap_years
            weighted_sum += float(hist_revenue) * weight
            weight_sum += weight
        recent_cast_scores.append(weighted_sum / weight_sum if weight_sum else 0.0)
        recent_cast_hits.append(float((actor_hist["revenue"] >= 1e8).mean()))
        recent_cast_blockbusters.append(float((actor_hist["revenue"] >= 1.5e8).mean()))

    recent_company_revenue: List[float] = []
    recent_company_success: List[float] = []
    recent_company_blockbusters: List[float] = []
    for company in companies:
        comp_hist = history_df[
            history_df["production_companies"].fillna("").str.contains(company, regex=False)
        ][["release_date", "revenue", "budget"]]
        if comp_hist.empty:
            continue
        comp_hist = comp_hist[
            (release_date - comp_hist["release_date"]).dt.days <= int(365.25 * company_recent_years)
        ]
        if len(comp_hist) < 2:
            continue
        recent_company_revenue.append(float(comp_hist["revenue"].mean()))
        recent_company_success.append(float((comp_hist["revenue"] > comp_hist["budget"].clip(lower=1.0) * 2.0).mean()))
        recent_company_blockbusters.append(float((comp_hist["revenue"] >= 1.5e8).mean()))

    recent_director_revenue: List[float] = []
    recent_director_hits: List[float] = []
    recent_director_max: List[float] = []
    for director in directors:
        if not director or director == "Unknown":
            continue
        director_hist = history_df[history_df["director"] == director][["release_date", "revenue"]]
        if director_hist.empty:
            continue
        director_hist = director_hist[
            (release_date - director_hist["release_date"]).dt.days <= int(365.25 * director_recent_years)
        ]
        if director_hist.empty:
            continue
        recent_director_revenue.append(float(director_hist["revenue"].mean()))
        recent_director_hits.append(float((director_hist["revenue"] >= 1e8).mean()))
        recent_director_max.append(float(director_hist["revenue"].max()))

    recent_writer_revenue: List[float] = []
    recent_writer_hits: List[float] = []
    recent_writer_max: List[float] = []
    for writer in writers:
        if not writer or writer == "Unknown":
            continue
        writer_hist = history_df[history_df["writer"] == writer][["release_date", "revenue"]]
        if writer_hist.empty:
            continue
        writer_hist = writer_hist[
            (release_date - writer_hist["release_date"]).dt.days <= int(365.25 * writer_recent_years)
        ]
        if writer_hist.empty:
            continue
        recent_writer_revenue.append(float(writer_hist["revenue"].mean()))
        recent_writer_hits.append(float((writer_hist["revenue"] >= 1e8).mean()))
        recent_writer_max.append(float(writer_hist["revenue"].max()))

    franchise_count_prior = 0.0
    franchise_mean_revenue_prior = 0.0
    franchise_max_revenue_prior = 0.0
    franchise_latest_revenue_prior = 0.0
    franchise_hit_rate_prior = 0.0
    franchise_recent_mean_revenue = 0.0
    franchise_recent_max_revenue = 0.0
    franchise_recency_years = 0.0
    franchise_active_gap_flag = 0.0
    if collection_name and collection_name not in ("None", "Unknown", ""):
        franchise_hist = history_df[history_df["belongs_to_collection"].fillna("").eq(collection_name)][["release_date", "revenue"]]
        if not franchise_hist.empty:
            franchise_count_prior = float(len(franchise_hist))
            franchise_mean_revenue_prior = float(franchise_hist["revenue"].mean())
            franchise_max_revenue_prior = float(franchise_hist["revenue"].max())
            latest_row = franchise_hist.sort_values("release_date").iloc[-1]
            franchise_latest_revenue_prior = float(latest_row["revenue"])
            franchise_hit_rate_prior = float((franchise_hist["revenue"] >= 1e8).mean())
            franchise_recent = franchise_hist[
                (release_date - franchise_hist["release_date"]).dt.days <= int(365.25 * franchise_recent_years)
            ]
            franchise_recent_mean_revenue = float(franchise_recent["revenue"].mean()) if not franchise_recent.empty else 0.0
            franchise_recent_max_revenue = float(franchise_recent["revenue"].max()) if not franchise_recent.empty else 0.0
            franchise_recency_years = float((release_date - franchise_hist["release_date"].max()).days / 365.25)
            franchise_active_gap_flag = float(franchise_recency_years <= 5.0)

    title_tokens = significant_title_tokens(title)
    title_match_values: List[float] = []
    recent_title_match_values: List[float] = []
    if title_tokens:
        title_mask = pd.Series(False, index=history_df.index)
        for token in title_tokens:
            title_mask |= history_df["title"].fillna("").str.contains(token, case=False, regex=False)
        title_hist = history_df.loc[title_mask, ["release_date", "revenue", "title"]].drop_duplicates(subset=["title", "release_date"])
        if not title_hist.empty:
            title_match_values = title_hist["revenue"].astype(float).tolist()
            recent_title_hist = title_hist[
                (release_date - title_hist["release_date"]).dt.days <= int(365.25 * 10)
            ]
            recent_title_match_values = recent_title_hist["revenue"].astype(float).tolist()

    title_lower = (title or "").lower()
    has_sequel_keyword = any(token in title_lower for token in ["2", "3", "4", "part", "chapter", "returns", "ii", "iii", "iv"])
    known_ip_proxy = float(
        bool(collection_name and collection_name not in ("None", "Unknown", ""))
        or len(title_match_values) >= 2
        or (len(title_match_values) >= 1 and has_sequel_keyword)
    )

    return {
        "cast_recent_weighted_revenue": float(np.mean(recent_cast_scores)) if recent_cast_scores else 0.0,
        "cast_recent_hit_rate": float(np.mean(recent_cast_hits)) if recent_cast_hits else 0.0,
        "cast_recent_blockbuster_rate": float(np.mean(recent_cast_blockbusters)) if recent_cast_blockbusters else 0.0,
        "company_recent_revenue": float(np.mean(recent_company_revenue)) if recent_company_revenue else 0.0,
        "company_recent_success_rate": float(np.mean(recent_company_success)) if recent_company_success else 0.0,
        "company_recent_blockbuster_rate": float(np.mean(recent_company_blockbusters)) if recent_company_blockbusters else 0.0,
        "director_recent_revenue": float(np.mean(recent_director_revenue)) if recent_director_revenue else 0.0,
        "director_recent_hit_rate": float(np.mean(recent_director_hits)) if recent_director_hits else 0.0,
        "director_recent_max_revenue": float(np.mean(recent_director_max)) if recent_director_max else 0.0,
        "writer_hist_revenue": 0.0,
        "writer_recent_revenue": float(np.mean(recent_writer_revenue)) if recent_writer_revenue else 0.0,
        "writer_recent_hit_rate": float(np.mean(recent_writer_hits)) if recent_writer_hits else 0.0,
        "writer_recent_max_revenue": float(np.mean(recent_writer_max)) if recent_writer_max else 0.0,
        "franchise_count_prior": franchise_count_prior,
        "franchise_mean_revenue_prior": franchise_mean_revenue_prior,
        "franchise_max_revenue_prior": franchise_max_revenue_prior,
        "franchise_latest_revenue_prior": franchise_latest_revenue_prior,
        "franchise_hit_rate_prior": franchise_hit_rate_prior,
        "franchise_recent_mean_revenue": franchise_recent_mean_revenue,
        "franchise_recent_max_revenue": franchise_recent_max_revenue,
        "franchise_recency_years": franchise_recency_years,
        "franchise_active_gap_flag": franchise_active_gap_flag,
        "ip_title_match_count_prior": float(len(title_match_values)),
        "ip_title_match_mean_revenue_prior": float(np.mean(title_match_values)) if title_match_values else 0.0,
        "ip_title_match_max_revenue_prior": float(np.max(title_match_values)) if title_match_values else 0.0,
        "ip_title_match_recent_max_revenue": float(np.max(recent_title_match_values)) if recent_title_match_values else 0.0,
        "known_ip_proxy": known_ip_proxy,
    }


def compute_cast_history(
    history_df: pd.DataFrame,
    cast_members: Sequence[str],
    release_year: int,
    min_history: int = 3,
    decay_factor: float = 0.9,
) -> Dict[str, float]:
    actor_scores: List[float] = []
    actor_maxes: List[float] = []
    actor_medians: List[float] = []
    actor_log_means: List[float] = []

    for actor in cast_members:
        actor_hist = history_df[
            history_df["cast"].fillna("").str.contains(actor, regex=False)
        ][["release_year", "revenue"]]
        if len(actor_hist) < min_history:
            continue
        weighted_sum = 0.0
        weight_sum = 0.0
        revs = actor_hist["revenue"].astype(float).tolist()
        for hist_year, revenue in actor_hist.itertuples(index=False):
            year_gap = max(int(release_year) - int(hist_year), 0)
            weight = decay_factor ** year_gap
            weighted_sum += float(revenue) * weight
            weight_sum += weight
        actor_scores.append(weighted_sum / weight_sum if weight_sum else 0.0)
        actor_maxes.append(float(max(revs)))
        actor_medians.append(float(np.median(revs)))
        actor_log_means.append(float(np.mean(np.log1p(revs))))

    if not actor_scores:
        return {
            "cast_avg_revenue": 0.0,
            "cast_max_revenue": 0.0,
            "cast_weighted_revenue": 0.0,
            "cast_median_revenue": 0.0,
            "cast_log_mean_revenue": 0.0,
        }

    rank_weights = np.linspace(1.2, 0.8, num=len(actor_scores))
    return {
        "cast_avg_revenue": float(np.mean(actor_scores)),
        "cast_max_revenue": float(np.max(actor_maxes)),
        "cast_weighted_revenue": float(np.average(actor_scores, weights=rank_weights)),
        "cast_median_revenue": float(np.mean(actor_medians)),
        "cast_log_mean_revenue": float(np.mean(actor_log_means)),
    }


def compute_competition_history(
    history_df: pd.DataFrame,
    release_date: pd.Timestamp,
    primary_genres: Sequence[str] | str,
    window_days: int = 14,
) -> Dict[str, float]:
    window_start = release_date - pd.Timedelta(days=window_days)
    mask = (history_df["release_date"] >= window_start) & (history_df["release_date"] < release_date)
    competitors = history_df.loc[mask]
    genre_values = _normalize_multi_select(primary_genres)
    same_genre = competitors[competitors["primary_genre"].isin(genre_values)] if genre_values else competitors.iloc[0:0]

    num_past_competitors = float(len(competitors))
    same_genre_count = float(len(same_genre))

    return {
        "num_past_competitors": num_past_competitors,
        "past_competition_index": num_past_competitors * (float(competitors["budget"].mean()) / 1e6)
        if len(competitors)
        else 0.0,
        "same_genre_past_comp_count": same_genre_count,
        "same_genre_past_comp_index": same_genre_count * (float(same_genre["budget"].mean()) / 1e6)
        if len(same_genre)
        else 0.0,
    }


def build_feature_row(
    title: str,
    budget: float,
    runtime: float,
    release_date: pd.Timestamp,
    primary_genres: Sequence[str] | str,
    languages: Sequence[str] | str,
    is_sequel: bool,
    collection_name: str | None,
    countries: Sequence[str],
    companies: Sequence[str],
    directors: Sequence[str],
    writers: Sequence[str],
    cast_members: Sequence[str],
    features_df: pd.DataFrame,
    keywords: str = "",
) -> pd.DataFrame:
    history_df = features_df[features_df["release_date"] < release_date].copy()
    genre_values = _normalize_multi_select(primary_genres)
    language_values = _normalize_multi_select(languages)
    release_year = int(release_date.year)
    release_month = int(release_date.month)
    release_quarter = int(release_date.quarter)
    release_day = int(release_date.day)
    release_dayofweek = int(release_date.dayofweek)

    company_director = compute_company_director_history(history_df, companies, directors)
    country_features = compute_country_history(history_df, countries, release_date)
    cast_features = compute_cast_history(history_df, cast_members, release_year)
    competition = compute_competition_history(history_df, release_date, genre_values)
    recent_hotness = compute_recent_hotness_features(
        history_df=history_df,
        release_date=release_date,
        title=title,
        companies=companies,
        directors=directors,
        writers=writers,
        cast_members=cast_members,
        collection_name=collection_name,
    )

    title_clean = title.strip()
    title_lower = title_clean.lower()
    title_word_count = len([part for part in title_clean.split() if part])
    has_sequel_keyword = int(
        any(token in title_lower for token in ["2", "3", "4", "part", "chapter", "returns", "ii", "iii", "iv"])
    )
    budget_per_minute = float(budget) / float(runtime) if runtime else 0.0
    month_angle = 2.0 * np.pi * (release_month - 1.0) / 12.0

    feature_map = {name: 0.0 for name in MODEL_FEATURES}
    feature_map.update(
        {
            "log_budget": float(np.log1p(max(budget, 0))),
            "log_runtime": float(np.log1p(max(runtime, 1))),
            "budget_per_minute": budget_per_minute,
            "log_budget_per_minute": float(np.log1p(max(budget_per_minute, 0))),
            "release_year": release_year,
            "release_month": release_month,
            "release_quarter": release_quarter,
            "release_month_sin": float(np.sin(month_angle)),
            "release_month_cos": float(np.cos(month_angle)),
            "post_covid_flag": float(release_date >= pd.Timestamp("2020-03-01")),
            "is_summer": float(release_month in [5, 6, 7, 8]),
            "is_holiday": float(release_month in [11, 12]),
            "is_friday": float(release_dayofweek == 4),
            "is_month_end": float(release_day >= 25),
            "is_weekend_release": float(release_dayofweek in [4, 5, 6]),
            "budget_imputed_flag": 0.0,
            "budget_was_zero_raw": float(budget <= 0),
            "runtime_is_flagged": float(runtime < 30 or runtime > 300),
            "is_sequel": float(is_sequel),
            "has_sequel_keyword": float(has_sequel_keyword),
            "is_english": float("en" in language_values),
            "title_length": float(len(title_clean)),
            "title_word_count": float(title_word_count),
        }
    )
    feature_map.update(company_director)
    feature_map.update(country_features)
    feature_map.update(cast_features)
    feature_map.update(competition)
    feature_map.update(recent_hotness)

    for genre_value in genre_values:
        genre_flag = GENRE_FLAG_MAP.get(genre_value)
        if genre_flag:
            feature_map[genre_flag] = 1.0

    TOP_KEYWORDS = [
        "based on novel or book", "woman director", "murder", "based on true story", "sequel",
        "new york city", "revenge", "biography", "duringcreditsstinger", "friendship",
        "black and white", "love", "christmas", "lgbt", "anime",
        "serial killer", "remake", "martial arts", "short film", "musical",
        "based on comic", "aftercreditsstinger", "parent child relationship", "high school", "police",
        "coming of age", "los angeles, california", "family", "world war ii", "gay theme",
        "superhero", "space travel", "alien", "war", "zombie",
        "heist", "time travel", "artificial intelligence", "prison", "vampire",
        "dystopia", "western", "ocean", "espionage", "pirates",
        "terrorist", "conspiracy", "sports", "medical", "drug",
        "court", "cult", "dinosaur", "robot", "astronaut",
        "soldier", "detective", "ninja", "samurai", "wizard",
        "dragon", "magic", "superpower",
    ]
    if keywords and not pd.isna(keywords):
        kw_list = [k.strip().lower() for k in str(keywords).split("|") if k.strip()]
        feature_map["keyword_count"] = float(len(kw_list))
        for kw in TOP_KEYWORDS:
            kw_col = f"kw_{kw.replace(' ', '_').replace(',', '').replace('-', '_')}"
            feature_map[kw_col] = float(1.0 if kw in kw_list else 0.0)
    else:
        feature_map["keyword_count"] = 0.0

    feature_row = pd.DataFrame([{name: feature_map.get(name, 0.0) for name in MODEL_FEATURES}])
    return feature_row


def _predict_model_object(model_obj: object, feature_row: pd.DataFrame) -> float:
    if isinstance(model_obj, dict) and model_obj.get("kind") == "two_stage":
        base_pred = float(np.expm1(model_obj["base_model"].predict(feature_row)[0]))
        classifier = model_obj.get("classifier")
        tail_model = model_obj.get("tail_model")
        if classifier is None:
            final_pred = base_pred
        else:
            if hasattr(classifier, "predict_proba"):
                hit_prob = float(classifier.predict_proba(feature_row)[0, 1])
            else:
                hit_prob = float(classifier.predict(feature_row)[0])
            blend_scale = float(model_obj.get("blend_scale", 0.85))
            blend_cap = float(model_obj.get("blend_cap", 0.9))
            blend = float(np.clip(hit_prob * blend_scale, 0.0, blend_cap))
            if tail_model is None:
                uplift = 1.0 + float(np.clip(hit_prob - 0.5, 0.0, 0.5)) * float(model_obj.get("uplift_strength", 0.35))
                final_pred = base_pred * uplift
            else:
                tail_pred = float(np.expm1(tail_model.predict(feature_row)[0]))
                final_pred = base_pred * (1.0 - blend) + tail_pred * blend

        validity_classifier = model_obj.get("validity_classifier")
        if validity_classifier is not None:
            if hasattr(validity_classifier, "predict_proba"):
                valid_prob = float(validity_classifier.predict_proba(feature_row)[0, 1])
            else:
                valid_prob = float(validity_classifier.predict(feature_row)[0])
            gate_threshold = float(model_obj.get("validity_gate_threshold", 0.35))
            gate_floor = float(model_obj.get("validity_gate_floor", 0.05))
            gate_power = float(model_obj.get("validity_gate_power", 1.25))
            normalized_prob = float(np.clip((valid_prob - gate_threshold) / max(1.0 - gate_threshold, 1e-6), 0.0, 1.0))
            gate = gate_floor + (1.0 - gate_floor) * (normalized_prob ** gate_power)
            final_pred *= gate
        return final_pred

    pred = model_obj.predict(feature_row)
    if getattr(model_obj, "_predicts_log_target", False):
        return float(np.expm1(pred[0]))
    return float(pred[0])


def predict_with_models(feature_row: pd.DataFrame, models: Dict[str, object]) -> Dict[str, float]:
    broad_lgb = _predict_model_object(models["lgb"], feature_row)
    broad_xgb = _predict_model_object(models["xgb"], feature_row)
    broad_ensemble = 0.5 * broad_lgb + 0.5 * broad_xgb
    deployment_model_name = str(models.get("deployment_model_name", "LightGBM(two-stage)"))
    deployment_lookup = {
        "LightGBM(two-stage)": broad_lgb,
        "LightGBM(log-target)": broad_lgb,
        "XGBoost(two-stage)": broad_xgb,
        "XGBoost(log-target)": broad_xgb,
        "Ensemble(two-stage)": broad_ensemble,
    }
    deployment_pred = float(deployment_lookup.get(deployment_model_name, broad_lgb))

    return {
        "broad_lgb": broad_lgb,
        "broad_xgb": broad_xgb,
        "broad_ensemble": broad_ensemble,
        "deployment_model_name": deployment_model_name,
        "deployment_pred": deployment_pred,
    }


def find_comparables(
    features_df: pd.DataFrame,
    feature_row: pd.DataFrame,
    primary_genres: Sequence[str] | str,
    languages: Sequence[str] | str,
    is_sequel: bool,
    limit: int = 12,
) -> pd.DataFrame:
    df = features_df.copy()
    genre_values = _normalize_multi_select(primary_genres)
    language_values = _normalize_multi_select(languages)
    df["budget_distance"] = np.abs(np.log1p(df["budget"].clip(lower=0)) - float(feature_row["log_budget"].iloc[0]))
    df["runtime_distance"] = np.abs(np.log1p(df["runtime"].clip(lower=1)) - float(feature_row["log_runtime"].iloc[0]))
    df["genre_match"] = df["primary_genre"].isin(genre_values).astype(float)
    df["language_match"] = df["original_language"].isin(language_values).astype(float)
    df["sequel_match"] = (df["is_sequel"].fillna(0).astype(int) == int(is_sequel)).astype(float)
    df["similarity_score"] = (
        df["genre_match"] * 2.5
        + df["language_match"] * 1.0
        + df["sequel_match"] * 1.0
        - df["budget_distance"] * 1.4
        - df["runtime_distance"] * 0.6
    )
    cols = [
        "title",
        "release_date",
        "primary_genre",
        "budget",
        "revenue",
        "budget_raw",
        "revenue_raw",
        "averageRating",
        "company_hist_revenue",
        "director_hist_revenue",
        "similarity_score",
    ]
    return df.sort_values("similarity_score", ascending=False).head(limit)[cols].copy()


def build_prediction_result(
    title: str,
    budget: float,
    runtime: float,
    release_date: pd.Timestamp,
    primary_genres: Sequence[str] | str,
    languages: Sequence[str] | str,
    is_sequel: bool,
    collection_name: str | None,
    countries: Sequence[str],
    companies: Sequence[str],
    directors: Sequence[str],
    writers: Sequence[str],
    cast_members: Sequence[str],
    features_df: pd.DataFrame,
    models: Dict[str, object],
    keywords: str = "",
) -> PredictionResult:
    feature_row = build_feature_row(
        title=title,
        budget=budget,
        runtime=runtime,
        release_date=release_date,
        primary_genres=primary_genres,
        languages=languages,
        is_sequel=is_sequel,
        collection_name=collection_name,
        countries=countries,
        companies=companies,
        directors=directors,
        writers=writers,
        cast_members=cast_members,
        features_df=features_df,
        keywords=keywords,
    )
    preds = predict_with_models(feature_row, models)
    comparables = find_comparables(features_df, feature_row, primary_genres, languages, is_sequel)

    revenue_col = "revenue_raw" if "revenue_raw" in comparables.columns else "revenue"
    comp_low = float(comparables[revenue_col].quantile(0.25)) if len(comparables) else 0.0
    comp_mid = float(comparables[revenue_col].median()) if len(comparables) else preds["deployment_pred"]
    comp_high = float(comparables[revenue_col].quantile(0.75)) if len(comparables) else preds["deployment_pred"]

    return PredictionResult(
        point_estimate=float(preds["deployment_pred"]),
        broad_estimate=float(preds["deployment_pred"]),
        comp_low=comp_low,
        comp_mid=comp_mid,
        comp_high=comp_high,
        feature_row=feature_row,
        comparables=comparables,
        model_outputs=preds,
    )


def plot_revenue_histogram(df: pd.DataFrame, color: str | None = None) -> go.Figure:
    plot_df = df[df["revenue"] > 0].copy()
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Revenue bucket",
            yaxis_title="Film count",
            annotations=[
                dict(
                    text="No positive-revenue rows for the current filter.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16, color="#66788a"),
                )
            ],
        )
        return fig

    plot_df["log10_revenue"] = np.log10(plot_df["revenue"].astype(float))
    fig = px.histogram(
        plot_df,
        x="log10_revenue",
        color=color if color and color in df.columns else None,
        nbins=30,
        opacity=0.8,
        template="plotly_white",
    )
    tick_vals = [0, 3, 6, 9]
    tick_text = ["$1", "$1K", "$1M", "$1B"]
    fig.update_xaxes(title="Revenue buckets (log10 scale)", tickvals=tick_vals, ticktext=tick_text)
    fig.update_yaxes(title="Film count")
    fig.update_layout(
        legend_title_text=color.replace("_", " ").title() if color and color in df.columns else None
    )
    return fig


def plot_budget_vs_revenue(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="budget",
        y="revenue",
        color="primary_genre" if "primary_genre" in df.columns else None,
        hover_name="title" if "title" in df.columns else None,
        template="plotly_white",
        opacity=0.65,
    )
    fig.update_xaxes(type="log", title="Budget (log scale)")
    fig.update_yaxes(type="log", title="Revenue (log scale)")
    return fig


def plot_actual_vs_pred(holdout_df: pd.DataFrame, pred_col: str) -> go.Figure:
    df = holdout_df.copy()
    fig = px.scatter(
        df,
        x="revenue",
        y=pred_col,
        hover_name="title",
        hover_data=["release_date", "budget"],
        template="plotly_white",
        opacity=0.7,
    )
    
    valid_revenue = df[df["revenue"] > 0]["revenue"]
    valid_pred = df[df[pred_col] > 0][pred_col]
    
    if len(valid_revenue) > 0 and len(valid_pred) > 0:
        min_val = float(min(valid_revenue.min(), valid_pred.min()))
        max_val = float(max(valid_revenue.max(), valid_pred.max()))
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="#a63d40", dash="dash"))
        
        fig.update_xaxes(
            type="log", 
            title="Actual revenue (log scale)",
            range=[np.log10(min_val * 0.5), np.log10(max_val * 1.5)]
        )
        fig.update_yaxes(
            type="log", 
            title="Predicted revenue (log scale)",
            range=[0, np.log10(max_val * 1.5)]
        )
    else:
        fig.update_xaxes(type="log", title="Actual revenue (log scale)")
        fig.update_yaxes(type="log", title="Predicted revenue (log scale)")
    
    return fig


def compute_budget_band_errors(df: pd.DataFrame, pred_col: str = "pred_revenue") -> pd.DataFrame:
    work = df.copy()
    work["abs_error"] = np.abs(work["revenue"] - work[pred_col])
    work["ape_gt_1m"] = np.where(work["revenue"] > 1e6, work["abs_error"] / work["revenue"], np.nan)
    work["budget_band"] = pd.cut(
        work["budget"],
        bins=[-1, 1e7, 5e7, 1e8, np.inf],
        labels=["<10M", "10-50M", "50-100M", ">100M"],
    )
    summary = work.groupby("budget_band", observed=False).agg(
        movies=("title", "count"),
        mae=("abs_error", "mean"),
        median_ape_gt_1m=("ape_gt_1m", "median"),
    ).reset_index()
    return summary


def top_under_predictions(df: pd.DataFrame, pred_col: str = "pred_revenue", limit: int = 10) -> pd.DataFrame:
    work = df.copy()
    work["miss"] = work["revenue"] - work[pred_col]
    return work.sort_values("miss", ascending=False).head(limit)[
        ["title", "release_date", "budget", "revenue", pred_col, "miss"]
    ]


def top_over_predictions(df: pd.DataFrame, pred_col: str = "pred_revenue", limit: int = 10) -> pd.DataFrame:
    work = df.copy()
    work["miss"] = work[pred_col] - work["revenue"]
    return work.sort_values("miss", ascending=False).head(limit)[
        ["title", "release_date", "budget", "revenue", pred_col, "miss"]
    ]


def release_timeline(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["release_year"] = pd.to_datetime(work["release_date"]).dt.year
    return work.groupby("release_year", observed=False).agg(
        movies=("title", "count"),
        avg_revenue=("revenue", "mean"),
    ).reset_index()


def describe_prediction_context(result: PredictionResult) -> str:
    if result.point_estimate <= 0:
        return "The current input mix looks weak relative to the historical theatrical sample."
    if result.point_estimate >= result.comp_high:
        return "The model sees this setup as a strong upside release compared with similar historical titles."
    if result.point_estimate <= result.comp_low:
        return "The model is placing this title below the lower quartile of its comparable set."
    return "The model is landing inside the historical comparable range, so this looks more like a mid-case scenario."
