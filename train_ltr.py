"""LightGBM LTR models — pointwise, pairwise, lambdarank."""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from metrics import compute_metrics, pointwise_match_predictions

FEATURE_COLS = [
    "rank",
    "avg_pts_1y",
    "avg_pts_1y_surface",
    "surf_winrate_y1",
    "trend_slope",
    "is_clay",
    "is_grass",
    "is_outdoor",
]

BASE_PARAMS = {
    "verbose":            -1,
    "feature_pre_filter": False,
}

CALLBACKS = [
    lgb.early_stopping(stopping_rounds=30, verbose=False),
    lgb.log_evaluation(period=50),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def clean_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only complete match pairs (one winner + one loser per match_id)."""
    df = df.dropna(subset=FEATURE_COLS).copy()
    counts = df.groupby("match_id")["won"].agg(["sum", "count"])
    valid  = counts[(counts["sum"] == 1) & (counts["count"] == 2)].index
    return df[df["match_id"].isin(valid)]


def build_ranking_data(df: pd.DataFrame):
    """
    Returns X, y, contiguous group array for LightGBM ranking.
    Grouped by Tournament so all player-match rows from the same tournament
    form one LTR query block.
    """
    df = df.sort_values(["Tournament", "match_id", "won"],
                        ascending=[True, True, False]).reset_index(drop=True)
    X      = df[FEATURE_COLS].values.astype(np.float32)
    y      = df["won"].values.astype(np.float32)
    groups = df.groupby("Tournament", sort=False).size().values
    return X, y, groups


# ── evaluation ─────────────────────────────────────────────────────────────────

def evaluate_pointwise(name: str, model: lgb.Booster, val_df: pd.DataFrame) -> dict:
    """
    Match-level evaluation for the pointwise binary model.
    Uses score_winner > score_loser per match instead of score > 0 per row,
    because binary probabilities are always in (0,1) and would threshold as
    'Winner' for every single row.
    """
    te = clean_pairs(val_df)

    y_true, y_pred = pointwise_match_predictions(te, model, FEATURE_COLS)

    te["_score"] = model.predict(te[FEATURE_COLS].values.astype(np.float32))
    pair_scores, pair_labels = [], []
    for _, match in te.groupby("match_id"):
        winner = match[match["won"] == 1]
        loser  = match[match["won"] == 0]
        if len(winner) != 1 or len(loser) != 1:
            continue
        w_score = winner.iloc[0]["_score"]
        l_score = loser.iloc[0]["_score"]
        pair_scores.extend([w_score - l_score, l_score - w_score])
        pair_labels.extend([1, 0])

    return {
        "model":  name,
        "auc":    float(roc_auc_score(pair_labels, pair_scores)),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def evaluate_lgb(name: str, model: lgb.Booster, val_df: pd.DataFrame) -> dict:
    te     = clean_pairs(val_df)
    X_val  = te[FEATURE_COLS].values.astype(np.float32)
    scores = model.predict(X_val)
    y_bin  = te["won"].values.astype(int)

    return compute_metrics(name, y_bin, scores)


# ── pointwise ──────────────────────────────────────────────────────────────────

def train_pointwise(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    tr = clean_pairs(train_df)
    te = clean_pairs(test_df)

    X_tr, y_tr, g_tr = build_ranking_data(tr)
    X_te, y_te, g_te = build_ranking_data(te)

    dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=FEATURE_COLS)
    dtest  = lgb.Dataset(X_te, label=y_te, group=g_te, feature_name=FEATURE_COLS,
                         reference=dtrain)

    params = {
        **BASE_PARAMS,
        "learning_rate":    0.0727173683915184,
        "num_leaves":       34,
        "min_data_in_leaf": 21,
        "feature_fraction": 0.6615959688535764,
        "bagging_fraction": 0.6463223060236656,
        "bagging_freq":     9,
        "lambda_l1":        0.8863041267050367,
        "lambda_l2":        1.7776764720980256,
        "objective":        "binary",
        "metric":           ["binary_logloss", "ndcg"],
        "ndcg_eval_at":     [1],
        "label_gain":       [0, 1],
    }

    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dtrain, dtest], valid_names=["train", "test"],
                      callbacks=CALLBACKS)
    model.save_model("model_pointwise.txt")
    return evaluate_pointwise("Pointwise", model, test_df)


# ── pairwise ───────────────────────────────────────────────────────────────────

def train_pairwise(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    tr = clean_pairs(train_df)
    te = clean_pairs(test_df)

    X_tr, y_tr, g_tr = build_ranking_data(tr)
    X_te, y_te, g_te = build_ranking_data(te)

    dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=FEATURE_COLS)
    dtest  = lgb.Dataset(X_te, label=y_te, group=g_te, feature_name=FEATURE_COLS,
                         reference=dtrain)

    params = {
        **BASE_PARAMS,
        "learning_rate":    0.04837082736323683,
        "num_leaves":       64,
        "min_data_in_leaf": 110,
        "feature_fraction": 0.5379779989855089,
        "bagging_fraction": 0.81543082263866,
        "bagging_freq":     6,
        "lambda_l1":        0.00012834460129174124,
        "lambda_l2":        0.0024015367057390255,
        "objective":        "rank_xendcg",
        "metric":           "ndcg",
        "ndcg_eval_at":     [1],
        "label_gain":       [0, 1],
    }

    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dtrain, dtest], valid_names=["train", "test"],
                      callbacks=CALLBACKS)
    model.save_model("model_pairwise.txt")
    return evaluate_lgb("Pairwise", model, test_df)


# ── lambdarank ─────────────────────────────────────────────────────────────────

def train_lambdarank(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    tr = clean_pairs(train_df)
    te = clean_pairs(test_df)

    X_tr, y_tr, g_tr = build_ranking_data(tr)
    X_te, y_te, g_te = build_ranking_data(te)

    dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=FEATURE_COLS)
    dtest  = lgb.Dataset(X_te, label=y_te, group=g_te, feature_name=FEATURE_COLS,
                         reference=dtrain)

    params = {
        **BASE_PARAMS,
        "learning_rate":    0.02418223838856672,
        "num_leaves":       15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.5891343931689867,
        "bagging_fraction": 0.8715591907578799,
        "bagging_freq":     3,
        "lambda_l1":        0.0008540982095558685,
        "lambda_l2":        0.0001398505350231253,
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [1],
        "label_gain":       [0, 1],
    }

    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dtrain, dtest], valid_names=["train", "test"],
                      callbacks=CALLBACKS)
    model.save_model("model_lambdarank.txt")
    return evaluate_lgb("LambdaRank", model, test_df)
