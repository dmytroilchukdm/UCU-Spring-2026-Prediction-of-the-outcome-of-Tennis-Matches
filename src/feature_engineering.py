"""
ATP Tennis – Feature Extraction Pipeline
=========================================
Input schema (one row per match):
    ATP, Location, Tournament, Date, Series, Court, Surface,
    Round, Best of, Winner, Loser, WRank, LRank, WPts, LPts

Output: one row per player per match, with features computed
from strictly historical data 
    - rank               : rank at time of match
    - avg_pts_1y         : mean points over trailing 365 days
    - avg_pts_1y_surface : mean points on this surface, trailing 365 days
    - surf_winrate_y1    : win rate on this surface, trailing 365 days
    - trend_slope        : linear slope of points over trailing 12 weeks (≈84 days)

Encoded context features (no leakage — these describe the match context, not history):
    - is_hard            : 1 if surface is Hard, else 0
    - is_clay            : 1 if surface is Clay, else 0
    - is_grass           : 1 if surface is Grass, else 0
    - is_outdoor         : 1 if court is Outdoor, else 0  (0 = Indoor)
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress


# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────

def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")          # adjust sep="," if comma-separated

    # Normalise column names
    df.columns = df.columns.str.strip()

    # Parse date — format is DD.MM.YYYY in your sample
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Drop rows where essential fields are missing
    df = df.dropna(subset=["Winner", "Loser", "Date", "WRank", "LRank", "WPts", "LPts"])

    # Coerce rank/points to numeric (sometimes "-" or missing appear)
    for col in ["WRank", "LRank", "WPts", "LPts"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["WRank", "LRank", "WPts", "LPts"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 2. RESHAPE: one row per player per match
# ─────────────────────────────────────────────

def reshape_to_player_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode each match into two rows — one for the winner, one for the loser —
    so every row represents a single player's experience of a single match.
    """
    df = df.reset_index(drop=True)
    df["match_id"] = df.index

    keep = ["match_id", "Date", "Tournament", "Surface", "Court", "Round", "Series"]

    winners = df[keep + ["Winner", "WRank", "WPts"]].copy()
    winners.columns = keep + ["player", "rank", "points"]
    winners["won"] = 1

    losers = df[keep + ["Loser", "LRank", "LPts"]].copy()
    losers.columns = keep + ["player", "rank", "points"]
    losers["won"] = 0

    long = pd.concat([winners, losers], ignore_index=True)
    long = long.sort_values(["player", "Date"]).reset_index(drop=True)
    return long


# ─────────────────────────────────────────────
# 3. FEATURE COMPUTATION (per player per match)
# ─────────────────────────────────────────────

WINDOW_DAYS  = 365   # 1-year lookback for winrate / avg_pts
TREND_DAYS   = 84    # ~12 weeks lookback for trend slope
MIN_MATCHES  = 3     # minimum history needed to emit a feature (else NaN)


def compute_features_for_player(group: pd.DataFrame) -> pd.DataFrame:
    """
    Receives all rows for a single player, sorted by Date.
    For each row i, uses only rows 0..i-1 as history (strict past).
    Adds 5 feature columns in-place and returns the group.
    """
    group = group.sort_values("Date").reset_index(drop=True)

    avg_pts_1y_list       = []
    avg_pts_1y_surf_list  = []
    surf_winrate_list     = []
    trend_slope_list      = []

    for i, row in group.iterrows():
        cutoff_1y    = row["Date"] - pd.Timedelta(days=WINDOW_DAYS)
        cutoff_trend = row["Date"] - pd.Timedelta(days=TREND_DAYS)

        # Strict past: all matches before this one
        past = group[group["Date"] < row["Date"]]

        # ── avg_pts_1y ──────────────────────────────────────────────
        past_1y = past[past["Date"] >= cutoff_1y]
        avg_pts_1y = past_1y["points"].mean() if len(past_1y) >= MIN_MATCHES else np.nan

        # ── avg_pts_1y_surface  &  surf_winrate_y1 ──────────────────
        past_surf = past_1y[past_1y["Surface"] == row["Surface"]]
        if len(past_surf) >= MIN_MATCHES:
            avg_pts_surf  = past_surf["points"].mean()
            surf_winrate  = past_surf["won"].mean()
        else:
            avg_pts_surf  = np.nan
            surf_winrate  = np.nan

        # ── trend_slope ─────────────────────────────────────────────
        # Linear slope of points over the last ~12 weeks.
        # Positive = player is gaining points (improving rank).
        past_trend = past[past["Date"] >= cutoff_trend].sort_values("Date")
        if len(past_trend) >= MIN_MATCHES:
            x = np.arange(len(past_trend), dtype=float)
            slope, *_ = linregress(x, past_trend["points"].values)
        else:
            slope = np.nan

        avg_pts_1y_list.append(avg_pts_1y)
        avg_pts_1y_surf_list.append(avg_pts_surf)
        surf_winrate_list.append(surf_winrate)
        trend_slope_list.append(slope)

    group["avg_pts_1y"]          = avg_pts_1y_list
    group["avg_pts_1y_surface"]  = avg_pts_1y_surf_list
    group["surf_winrate_y1"]     = surf_winrate_list
    group["trend_slope"]         = trend_slope_list

    return group


# ─────────────────────────────────────────────
# 4. ENCODE SURFACE & COURT (context features)
# ─────────────────────────────────────────────

# Canonical surface values found in ATP data
SURFACE_MAP = {
    "hard":  "Hard",
    "clay":  "Clay",
    "grass": "Grass",
    "carpet": "Carpet",   # legacy surface, treat as Hard below
}

def encode_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add one-hot surface columns and a binary court column.
    Normalises the raw Surface / Court strings first (handles mixed case,
    trailing spaces, or slight variations in the source data).
    """
    surf = df["Surface"].str.strip().str.title()   # e.g. "hard " → "Hard"
    court = df["Court"].str.strip().str.title()    # e.g. "outdoor" → "Outdoor"

    # ── Surface one-hot ─────────────────────────────────────────────
    # Carpet is very rare in modern ATP; map it to Hard for simplicity.
    df["is_hard"]  = surf.isin(["Hard", "Carpet"]).astype(int)
    df["is_clay"]  = (surf == "Clay").astype(int)
    df["is_grass"] = (surf == "Grass").astype(int)

    # Warn if any surface values fall outside known categories
    known = {"Hard", "Clay", "Grass", "Carpet"}
    unknown_surf = set(surf.unique()) - known
    if unknown_surf:
        print(f"  ⚠  Unknown Surface values (mapped to all-zero): {unknown_surf}")

    # ── Court one-hot (binary: Outdoor=1, Indoor=0) ──────────────────
    df["is_outdoor"] = (court == "Outdoor").astype(int)

    unknown_court = set(court.unique()) - {"Outdoor", "Indoor"}
    if unknown_court:
        print(f"  ⚠  Unknown Court values (mapped to Indoor=0): {unknown_court}")

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Surface × player-history interactions.
    These are zero outside the relevant surface, so differencing them
    across a pair gives a non-zero signal only when the match is on
    that surface — letting a linear model learn per-surface coefficients.
    Must be called after encode_context_features.
    """
    df["avg_pts_x_clay"]    = df["avg_pts_1y"]         * df["is_clay"]
    df["avg_pts_x_grass"]   = df["avg_pts_1y"]         * df["is_grass"]
    df["avg_pts_x_outdoor"] = df["avg_pts_1y"]         * df["is_outdoor"]
    df["surf_pts_x_clay"]   = df["avg_pts_1y_surface"] * df["is_clay"]
    df["surf_pts_x_grass"]  = df["avg_pts_1y_surface"] * df["is_grass"]
    df["winrate_x_clay"]    = df["surf_winrate_y1"]    * df["is_clay"]
    df["winrate_x_grass"]   = df["surf_winrate_y1"]    * df["is_grass"]
    df["winrate_x_outdoor"] = df["surf_winrate_y1"]    * df["is_outdoor"]
    df["trend_x_clay"]      = df["trend_slope"]        * df["is_clay"]
    df["trend_x_grass"]     = df["trend_slope"]        * df["is_grass"]
    return df


def extract_all_features(long: pd.DataFrame) -> pd.DataFrame:
    """
    Apply compute_features_for_player to every player group,
    then encode surface and court as context features.
    `rank` is already present from the reshape step (rank at match date).
    """
    print(f"Computing features for {long['player'].nunique()} players "
          f"across {long['Date'].dt.year.nunique()} years…")

    groups = [compute_features_for_player(g) for _, g in long.groupby("player")]
    featured = pd.concat(groups, ignore_index=True)

    print("Encoding surface and court features…")
    featured = encode_context_features(featured)

    print("Adding interaction features…")
    featured = add_interaction_features(featured)

    return featured


# ─────────────────────────────────────────────
# 5. SPLIT: train (2024) vs predict (2025-26)
# ─────────────────────────────────────────────

def split_train_predict(featured: pd.DataFrame):
    train   = featured[featured["Date"].dt.year == 2024].copy()
    predict = featured[featured["Date"].dt.year == 2025].copy()
    print(f"Train rows : {len(train):,}  ({train['Date'].dt.year.value_counts().to_dict()})")
    print(f"Predict rows: {len(predict):,}  ({predict['Date'].dt.year.value_counts().to_dict()})")
    return train, predict


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "atp_matches.csv"

    # Load
    matches = load_matches(path)
    print(f"Loaded {len(matches):,} matches  "
          f"({matches['Date'].min().date()} → {matches['Date'].max().date()})")

    # Reshape
    long = reshape_to_player_rows(matches)

    # Extract features
    featured = extract_all_features(long)

    # Split
    train, predict = split_train_predict(featured)

    # Save
    featured.to_csv("player_features_all.csv", index=False)
    train.to_csv("player_features_train.csv",   index=False)
    predict.to_csv("player_features_predict.csv", index=False)

    print("\nSample output (first 5 rows):")
    feature_cols = [
        "player", "Date", "Tournament", "Surface", "Court",
        "rank", "avg_pts_1y", "avg_pts_1y_surface",
        "surf_winrate_y1", "trend_slope",
        "is_hard", "is_clay", "is_grass", "is_outdoor",
    ]
    print(featured[feature_cols].dropna().head().to_string(index=False))