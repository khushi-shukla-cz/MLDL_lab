import pickle
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Crop_recommendation.csv"

FEATURE_COLUMNS = [
    "Nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
TARGET_COLUMN = "label"
RANDOM_STATE = 42
METRICS_PATH = BASE_DIR / "metrics.json"


def get_model_candidates() -> dict:
    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    if XGBClassifier is not None:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=450,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    return candidates


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    # Remove empty/unnamed columns caused by trailing commas in CSV rows.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")

    return df


def evaluate_model(model_name: str, model, x_train_scaled, y_train, x_val_scaled, y_val, classes_count: int) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    }
    cv_results = cross_validate(
        clone(model),
        x_train_scaled,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    fitted = clone(model)
    fitted.fit(x_train_scaled, y_train)

    y_pred = fitted.predict(x_val_scaled)
    y_proba = fitted.predict_proba(x_val_scaled)

    top_k = 3 if classes_count >= 3 else classes_count
    metrics = {
        "model": model_name,
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(cv_results["test_f1_macro"])),
        "val_accuracy": float(accuracy_score(y_val, y_pred)),
        "val_f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "val_top3_accuracy": float(top_k_accuracy_score(y_val, y_proba, k=top_k, labels=np.arange(classes_count))),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
    }
    return metrics


def train_and_save(df: pd.DataFrame) -> None:
    x = df[FEATURE_COLUMNS].astype(float)
    y_raw = df[TARGET_COLUMN].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes_count = len(label_encoder.classes_)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler_train = StandardScaler()
    x_train_scaled = scaler_train.fit_transform(x_train)
    x_val_scaled = scaler_train.transform(x_val)

    candidates = get_model_candidates()
    benchmark_results = []

    for model_name, model in candidates.items():
        result = evaluate_model(
            model_name,
            model,
            x_train_scaled,
            y_train,
            x_val_scaled,
            y_val,
            classes_count,
        )
        benchmark_results.append(result)

    benchmark_results.sort(
        key=lambda row: (row["val_f1_macro"], row["val_accuracy"], row["cv_f1_macro_mean"]),
        reverse=True,
    )
    best_model_name = benchmark_results[0]["model"]

    # Refit scaler and best model on all data for final artifact training.
    final_scaler = StandardScaler()
    x_scaled_all = final_scaler.fit_transform(x)
    final_model = clone(candidates[best_model_name])
    final_model.fit(x_scaled_all, y)

    with open(BASE_DIR / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    with open(BASE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(final_scaler, f)

    with open(BASE_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    metrics_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATA_PATH),
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "class_count": classes_count,
        "classes": label_encoder.classes_.tolist(),
        "random_state": RANDOM_STATE,
        "validation_split": 0.2,
        "cv_folds": 5,
        "best_model": best_model_name,
        "benchmark_results": benchmark_results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)


def main() -> None:
    df = load_dataset(DATA_PATH)
    train_and_save(df)
    print("Training complete. Saved model.pkl, scaler.pkl, label_encoder.pkl, and metrics.json")


if __name__ == "__main__":
    main()
