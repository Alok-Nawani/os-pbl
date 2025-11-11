from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from backend.ml_model import FEATURE_NAMES, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multiple classifiers and print simple metrics")
    p.add_argument("--csv", required=True, help="Path to dataset CSV (must contain FEATURE_NAMES and 'policy')")
    p.add_argument("--out", required=True, help="Base path to save models (each model will append name before extension)")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    # Expect dataset to have precomputed features and 'policy' target
    missing = [c for c in FEATURE_NAMES + ["policy"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    X = df[FEATURE_NAMES]
    y = df["policy"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Prepare models
    models: Dict[str, object] = {
        "decision_tree": DecisionTreeClassifier(random_state=args.random_state, class_weight="balanced"),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=args.random_state, class_weight="balanced"),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=args.random_state)
    except Exception:
        # xgboost is optional; just skip if not installed
        print("xgboost not installed; skipping XGBoost model. Install xgboost to enable it.")

    labels = sorted(y.unique())

    base_out = args.out
    root, ext = os.path.splitext(base_out)
    if not ext:
        ext = ".joblib"

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Confusion matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])
        print(cm_df.to_string())
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        out_path = f"{root}_{name}{ext}"
        save_model(model, out_path)
        print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()


