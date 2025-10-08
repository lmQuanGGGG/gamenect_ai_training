from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from pipelines.dataset_builder import DatasetBuilder
from dotenv import load_dotenv
import os
from rich.console import Console

load_dotenv()
console = Console()


def main() -> None:
    model_path = Path(os.getenv("TRAINING_OUTPUT_DIR", "data/artifacts")) / "gamenect_matching_model.h5"
    if not model_path.exists():
        raise FileNotFoundError("Chua co model, hay chay scripts/train_model.py truoc")

    test_data_path = Path(os.getenv("RAW_DATA_DIR", "data/raw")) / "test_samples.json"
    if not test_data_path.exists():
        raise FileNotFoundError("Chua co test_samples.json, hay tao tap test tu collector")

    builder = DatasetBuilder(Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")))
    X_test, y_test, _ = builder.build_dataset(test_data_path)

    model = tf.keras.models.load_model(model_path.as_posix())
    predictions = model.predict(X_test, batch_size=256)
    y_prob = predictions.squeeze()
    y_pred = (y_prob >= 0.5).astype(int)

    # Tính AUC nếu có cả 2 class
    try:
        auc_score = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        console.print("[yellow]Warning: Only one class in test set, AUC not available[/yellow]")
        auc_score = 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": auc_score,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    console.rule("Ket qua danh gia")
    for k, v in metrics.items():
        console.print(f"{k}: {v}")

    report_path = Path(os.getenv("TRAINING_HISTORY_DIR", "logs/training_history"))
    report_path.mkdir(parents=True, exist_ok=True)
    output_file = report_path / "evaluation_report.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    console.print(f"Da luu bao cao tai {output_file}")


if __name__ == "__main__":
    main()
