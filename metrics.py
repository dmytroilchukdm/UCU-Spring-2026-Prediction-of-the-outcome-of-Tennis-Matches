"""
Shared evaluation utilities for all tennis LTR models.
Imported by svm_rank.py, train_ltr.py, and compare_models.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix


def compute_metrics(name: str, y_bin: np.ndarray, scores: np.ndarray) -> dict:
    y_pred = (scores > 0).astype(int)
    return {
        "model":  name,
        "auc":    float(roc_auc_score(y_bin, scores)),
        "y_true": y_bin.tolist(),
        "y_pred": y_pred.tolist(),
    }


def pointwise_match_predictions(val_df: pd.DataFrame,
                                model,
                                feature_cols: list) -> tuple:
    """
    Correct evaluation for pointwise models.
    For each match, compare winner score vs loser score.
    Predicted winner = player with higher score.

    Returns y_true, y_pred — two entries per match (one per player),
    so both True Loser and True Winner rows are populated in the confusion matrix.
    """
    val_df = val_df.copy()
    val_df["score"] = model.predict(val_df[feature_cols])

    y_true, y_pred = [], []
    for _, match in val_df.groupby("match_id"):
        if len(match) != 2:
            continue
        winner = match[match["won"] == 1]
        loser  = match[match["won"] == 0]
        if len(winner) != 1 or len(loser) != 1:
            continue

        w_score = winner.iloc[0]["score"]
        l_score = loser.iloc[0]["score"]
        model_correct = w_score > l_score

        y_true.extend([1, 0])
        y_pred.extend([1 if model_correct else 0,
                       0 if model_correct else 1])

    return np.array(y_true), np.array(y_pred)


def print_comparison_table(results: list) -> None:
    sep = "=" * 48
    print(f"\n{sep}")
    print(f"  MODEL COMPARISON -- 2025 hold-out")
    print(sep)
    print(f"  {'Model':<28} {'AUC-ROC':>10}")
    print("-" * 48)
    for r in results:
        print(f"  {r['model']:<28} {r['auc']:>10.4f}")
    print("-" * 48)
    best = max(results, key=lambda r: r["auc"])
    print(f"  Best -> {best['model']}  ({best['auc']:.4f})")
    print(sep)


def print_confusion_matrix_text(models_data: list) -> None:
    """Print confusion matrices as formatted text tables."""
    for m in models_data:
        y_true = np.asarray(m["y_true"])
        y_pred = np.asarray(m["y_pred"])
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total     = cm.sum()
        accuracy  = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)

        print(f"\n-- {m['name']} " + "-" * 44)
        print(f"                  Predicted Loser   Predicted Winner")
        print(f"  True Loser      {tn:>8} ({tn/total*100:4.1f}%)  {fp:>8} ({fp/total*100:4.1f}%)")
        print(f"  True Winner     {fn:>8} ({fn/total*100:4.1f}%)  {tp:>8} ({tp/total*100:4.1f}%)")
        print(f"  " + "-" * 49)
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}  (of predicted winners, how many truly won)")
        print(f"  Recall   : {recall:.4f}  (of true winners, how many did we catch)")
        print(f"  F1       : {f1:.4f}")



