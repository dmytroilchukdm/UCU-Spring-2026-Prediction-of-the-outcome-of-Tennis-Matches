"""SVM pairwise ranking model — training and inference functions."""

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ── feature config ─────────────────────────────────────────────────────────────

DELTA_COLS = [
    "rank",
    "avg_pts_1y",
    "avg_pts_1y_surface",
    "surf_winrate_y1",
    "trend_slope",
    # surface interactions — differenced between players so the linear model
    # gets separate per-surface coefficients (zero outside the relevant surface)
    "avg_pts_x_clay",
    "avg_pts_x_grass",
    "avg_pts_x_outdoor",
    "surf_pts_x_clay",
    "surf_pts_x_grass",
    "winrate_x_clay",
    "winrate_x_grass",
    "winrate_x_outdoor",
    "trend_x_clay",
    "trend_x_grass",
]
CONTEXT_COLS = ["is_clay", "is_grass", "is_outdoor"]   # is_hard dropped (collinear)
PAIR_COLS    = [f"d_{c}" for c in DELTA_COLS] + CONTEXT_COLS


# ── data helpers ───────────────────────────────────────────────────────────────

def player_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """One feature row per player per tournament — earliest match, chronological."""
    keep = ["Tournament", "player", "Date"] + DELTA_COLS + CONTEXT_COLS
    return (
        df.sort_values("Date")
          .groupby(["Tournament", "player"], sort=False)
          .first()
          .reset_index()[keep]
          .dropna(subset=DELTA_COLS)
    )


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each match produces exactly two delta rows (original + mirror).
    Winner (won=1) features minus Loser (won=0) features → label +1.
    Context features are shared and identical on both rows — never negated.
    """
    df = df.dropna(subset=DELTA_COLS).copy()
    rows = []
    for _, grp in df.groupby("match_id"):
        w_rows = grp[grp["won"] == 1]
        l_rows = grp[grp["won"] == 0]
        if len(w_rows) != 1 or len(l_rows) != 1:
            continue
        fw  = w_rows.iloc[0]
        fl  = l_rows.iloc[0]
        ctx = {c: fw[c] for c in CONTEXT_COLS}

        rows.append({**{f"d_{c}": fw[c] - fl[c] for c in DELTA_COLS}, **ctx, "label":  1})
        rows.append({**{f"d_{c}": fl[c] - fw[c] for c in DELTA_COLS}, **ctx, "label": -1})

    return pd.DataFrame(rows)


# ── evaluation helpers ─────────────────────────────────────────────────────────

def svm_match_predictions(val_pairs, svm, scaler):
    scores = svm.decision_function(
        scaler.transform(val_pairs[PAIR_COLS].values.astype(np.float32))
    )
    val_pairs = val_pairs.copy()
    val_pairs["score"] = scores

    pos_rows = val_pairs[val_pairs["label"] ==  1].reset_index(drop=True)

    y_true, y_pred = [], []
    for i in range(len(pos_rows)):
        model_correct = pos_rows.iloc[i]["score"] > 0

        # Winner entry
        y_true.append(1)
        y_pred.append(1 if model_correct else 0)

        # Loser entry — opposite prediction
        y_true.append(0)
        y_pred.append(0 if model_correct else 1)

    return np.array(y_true), np.array(y_pred)


# ── model ──────────────────────────────────────────────────────────────────────

def _fit_svm(X_train: np.ndarray, y_train: np.ndarray):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    svm      = LinearSVC(C=1.0, max_iter=10_000)
    svm.fit(X_scaled, y_train)
    return svm, scaler


def evaluate_svm(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train SVM on train_df, evaluate on val_df, return result dict."""
    train_pairs = build_pairs(train_df)
    val_pairs   = build_pairs(val_df)

    X_train = train_pairs[PAIR_COLS].values.astype(np.float32)
    y_train = train_pairs["label"].values
    X_val   = val_pairs[PAIR_COLS].values.astype(np.float32)
    y_val   = val_pairs["label"].values

    svm, scaler = _fit_svm(X_train, y_train)

    # AUC — pair level (standard, invariant to match/pair choice)
    y_bin  = (y_val == 1).astype(int)
    scores = svm.decision_function(scaler.transform(X_val))
    auc    = float(roc_auc_score(y_bin, scores))

    # Confusion matrix — match level (one prediction per match, no forced symmetry)
    y_true, y_pred = svm_match_predictions(val_pairs, svm, scaler)

    return {
        "model":  "SVM",
        "auc":    auc,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


# ── inference ──────────────────────────────────────────────────────────────────

def rank_tournament(svm: LinearSVC, scaler: StandardScaler,
                    group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank all players in a tournament by average decision_function score
    against every other player in the group.
    group_df: output of player_snapshots filtered to one tournament.
    Returns DataFrame[player, score] sorted by score descending.
    """
    players = group_df.drop_duplicates("player").set_index("player")
    scores  = {}
    for a in players.index:
        opponents = [p for p in players.index if p != a]
        if not opponents:
            scores[a] = 0.0
            continue
        X = np.array([
            [players.loc[a, c] - players.loc[b, c] for c in DELTA_COLS]
            + [players.loc[a, c] for c in CONTEXT_COLS]
            for b in opponents
        ], dtype=np.float32)
        scores[a] = svm.decision_function(scaler.transform(X)).mean()

    return (
        pd.Series(scores, name="score")
          .sort_values(ascending=False)
          .reset_index()
          .rename(columns={"index": "player"})
    )


def predict_head_to_head(svm: LinearSVC, scaler: StandardScaler,
                         df: pd.DataFrame,
                         player_a: str, player_b: str,
                         surface: str, is_outdoor: int) -> dict:
    """
    Compare two players using their most recent feature snapshot.

    surface   : 'Hard', 'Clay', or 'Grass'
    is_outdoor: 1 = outdoor, 0 = indoor
    Returns   : dict with winner, loser, and margin (decision-function magnitude).
    """
    def latest(name):
        rows = df[df["player"] == name].sort_values("Date")
        if rows.empty:
            raise ValueError(f"Player not found: {name}")
        return rows.iloc[-1]

    fa, fb   = latest(player_a), latest(player_b)
    is_clay  = int(surface == "Clay")
    is_grass = int(surface == "Grass")

    X = np.array(
        [[fa[c] - fb[c] for c in DELTA_COLS] + [is_clay, is_grass, is_outdoor]],
        dtype=np.float32,
    )
    margin = svm.decision_function(scaler.transform(X))[0]
    winner = player_a if margin > 0 else player_b
    loser  = player_b if margin > 0 else player_a

    return {"winner": winner, "loser": loser, "margin": float(abs(margin))}
