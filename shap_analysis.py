"""SHAP analysis for all four ATP tennis prediction models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import lightgbm as lgb

from svm_rank  import build_pairs, PAIR_COLS, DELTA_COLS, CONTEXT_COLS
from train_ltr import clean_pairs, build_ranking_data, FEATURE_COLS


# ── config ─────────────────────────────────────────────────────────────────────

PLOTS_DIR = "shap_plots"
N_SAMPLES  = 500   # subsample for speed — SHAP is O(n²) for some explainers


# ── helpers ────────────────────────────────────────────────────────────────────

def save_fig(name: str) -> None:
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = f"{PLOTS_DIR}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


def subsample(X: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    idx = np.random.default_rng(42).choice(len(X), min(n, len(X)), replace=False)
    return X[idx]


def summary_bar(shap_values, X, feature_names, title, name):
    """Mean absolute SHAP value bar chart."""
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title(title)
    save_fig(name)


def summary_dot(shap_values, X, feature_names, title, name):
    """Dot/beeswarm plot showing direction and magnitude."""
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False,
    )
    plt.title(title)
    save_fig(name)


def waterfall_single(explainer, shap_values, X, feature_names, title, name):
    """Explain one prediction — pick the match where model was most confident."""
    most_confident = np.argmax(np.abs(shap_values).sum(axis=1))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[most_confident],
            base_values=explainer.expected_value
                        if np.isscalar(explainer.expected_value)
                        else explainer.expected_value[0],
            data=X[most_confident],
            feature_names=feature_names,
        ),
        show=False,
    )
    plt.title(title)
    save_fig(name)


def dependence(shap_values, X, feature_names, feature, title, name):
    """How one feature's value affects its SHAP contribution."""
    shap.dependence_plot(
        feature, shap_values, X,
        feature_names=feature_names,
        show=False,
    )
    plt.title(title)
    save_fig(name)


# ── SVM ────────────────────────────────────────────────────────────────────────

def analyse_svm(val_df: pd.DataFrame) -> None:
    print("\n── SVM ──────────────────────────────────────────")
    svm    = joblib.load("svm_rank.pkl")
    scaler = joblib.load("scaler.pkl")

    val_pairs = build_pairs(val_df)
    X_val     = scaler.transform(
        val_pairs[PAIR_COLS].values.astype(np.float32)
    )
    X_sub = subsample(X_val)

    # LinearExplainer is exact for linear models
    explainer   = shap.LinearExplainer(svm, X_val)
    shap_values = explainer.shap_values(X_sub)

    summary_bar(shap_values, X_sub, PAIR_COLS,
                "SVM — Feature Importance (SHAP)", "svm_bar")
    summary_dot(shap_values, X_sub, PAIR_COLS,
                "SVM — Feature Direction (SHAP)", "svm_dot")
    waterfall_single(explainer, shap_values, X_sub, PAIR_COLS,
                     "SVM — Most Confident Prediction", "svm_waterfall")
    dependence(shap_values, X_sub, PAIR_COLS, "d_avg_pts_1y",
               "SVM — avg_pts_1y dependence", "svm_dependence_pts")


# ── LightGBM shared ────────────────────────────────────────────────────────────

def analyse_lgb(model_path: str,
                val_df: pd.DataFrame,
                name: str,
                prefix: str) -> None:
    print(f"\n── {name} ──────────────────────────────────────")
    model = lgb.Booster(model_file=model_path)
    te    = clean_pairs(val_df)
    X_val = te[FEATURE_COLS].values.astype(np.float32)
    X_sub = subsample(X_val)

    # TreeExplainer is exact and fast for LightGBM
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    # LightGBM may return a list of arrays for binary objectives
    # — take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    summary_bar(shap_values, X_sub, FEATURE_COLS,
                f"{name} — Feature Importance (SHAP)", f"{prefix}_bar")
    summary_dot(shap_values, X_sub, FEATURE_COLS,
                f"{name} — Feature Direction (SHAP)", f"{prefix}_dot")
    waterfall_single(explainer, shap_values, X_sub, FEATURE_COLS,
                     f"{name} — Most Confident Prediction", f"{prefix}_waterfall")
    dependence(shap_values, X_sub, FEATURE_COLS, "avg_pts_1y",
               f"{name} — avg_pts_1y dependence", f"{prefix}_dependence_pts")
    dependence(shap_values, X_sub, FEATURE_COLS, "trend_slope",
               f"{name} — trend_slope dependence", f"{prefix}_dependence_trend")


# ── cross-model comparison ─────────────────────────────────────────────────────

def plot_cross_model_importance(val_df: pd.DataFrame) -> None:
    """
    Single bar chart comparing mean |SHAP| per feature across all models.
    Shows at a glance which features each model relies on differently.
    """
    print("\n── Cross-model comparison ───────────────────────")

    svm    = joblib.load("svm_rank.pkl")
    scaler = joblib.load("scaler.pkl")

    # SVM — pairwise features
    val_pairs   = build_pairs(val_df)
    X_svm       = scaler.transform(val_pairs[PAIR_COLS].values.astype(np.float32))
    X_svm_sub   = subsample(X_svm)
    exp_svm     = shap.LinearExplainer(svm, X_svm)
    sv_svm      = exp_svm.shap_values(X_svm_sub)
    imp_svm     = pd.Series(np.abs(sv_svm).mean(axis=0), index=PAIR_COLS)

    # LightGBM models — shared features
    te    = clean_pairs(val_df)
    X_lgb = te[FEATURE_COLS].values.astype(np.float32)
    X_sub = subsample(X_lgb)

    importances = {"SVM (pairwise features)": imp_svm}

    for model_path, label in [
        ("model_pointwise.txt",  "Pointwise"),
        ("model_pairwise.txt",   "Pairwise"),
        ("model_lambdarank.txt", "LambdaRank"),
    ]:
        model = lgb.Booster(model_file=model_path)
        exp   = shap.TreeExplainer(model)
        sv    = exp.shap_values(X_sub)
        if isinstance(sv, list):
            sv = sv[1]
        importances[label] = pd.Series(np.abs(sv).mean(axis=0), index=FEATURE_COLS)

    # Plot LightGBM models side by side (same feature space)
    lgb_imp = pd.DataFrame({k: v for k, v in importances.items() if k != "SVM (pairwise features)"})
    lgb_imp.sort_values("LambdaRank", ascending=False).plot(
        kind="bar", figsize=(12, 5)
    )
    plt.title("LightGBM Models — Mean |SHAP| per Feature")
    plt.xlabel("Feature")
    plt.ylabel("Mean |SHAP|")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_fig("cross_model_lgb_comparison")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    val_df = pd.read_csv("player_features_predict.csv", parse_dates=["Date"])
    print(f"Loaded {len(val_df):,} val rows")

    analyse_svm(val_df)
    analyse_lgb("model_pointwise.txt",  val_df, "Pointwise",  "pw")
    analyse_lgb("model_pairwise.txt",   val_df, "Pairwise",   "pr")
    analyse_lgb("model_lambdarank.txt", val_df, "LambdaRank", "lr")
    plot_cross_model_importance(val_df)

    print(f"\nAll plots saved to ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
