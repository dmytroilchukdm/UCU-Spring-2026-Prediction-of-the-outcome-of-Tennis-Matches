"""Full ATP tennis match prediction pipeline — data loading through model comparison."""

import sqlite3
import numpy as np
import pandas as pd

from src.feature_engineering import (
    extract_all_features, reshape_to_player_rows, split_train_predict,
)
from svm_rank  import evaluate_svm
from train_ltr import train_pointwise, train_pairwise, train_lambdarank
from metrics   import print_comparison_table, print_confusion_matrix_text


DB_PATH  = "C:\\sqlite\\atp.db"
CSV_PATH = "atp_matches.csv"


def load_matches(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM atp_matches", conn)
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    for col in ["WRank", "LRank", "WPts", "LPts"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def build_features(csv_path: str) -> tuple:
    matches  = pd.read_csv(csv_path, parse_dates=["Date"])
    long     = reshape_to_player_rows(matches)
    featured = extract_all_features(long)
    train, predict = split_train_predict(featured)
    return featured, train, predict


def run_models(train_df: pd.DataFrame, predict_df: pd.DataFrame) -> list:
    results = []
    results.append(evaluate_svm(train_df, predict_df))
    results.append(train_pointwise(train_df, predict_df))
    results.append(train_pairwise(train_df, predict_df))
    results.append(train_lambdarank(train_df, predict_df))
    return results


def print_results(results: list) -> None:
    print_comparison_table(results)
    print_confusion_matrix_text([{
        "name":   r["model"],
        "y_true": np.array(r["y_true"]),
        "y_pred": np.array(r["y_pred"]),
    } for r in results])


if __name__ == "__main__":
    matches = load_matches(DB_PATH)
    print(f"Loaded {len(matches):,} matches  "
          f"({matches['Date'].min().date()} -> {matches['Date'].max().date()})")

    matches.to_csv(CSV_PATH, index=False)
    print(f"Saved   -> {CSV_PATH}")

    featured, train, predict = build_features(CSV_PATH)
    results                  = run_models(train, predict)
    print_results(results)
