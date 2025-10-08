from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import json
import numpy as np
from .feature_engineering import FeatureEngineer


class DatasetBuilder:
    """Xây dựng tập dữ liệu huấn luyện từ file JSON."""

    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = processed_dir
        self.feature_engineer = FeatureEngineer()
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def build_dataset(self, training_json: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        with training_json.open("r", encoding="utf-8") as f:
            samples = json.load(f)

        feature_list = []
        labels = []
        feature_names = None

        for sample in samples:
            user1 = sample.get("user1", {})
            user2 = sample.get("user2", {})
            liked = sample.get("liked", 0)

            vector, names = self.feature_engineer.build_pair_features(user1, user2)
            feature_list.append(vector)
            labels.append(liked)
            if feature_names is None:
                feature_names = names

        X = np.stack(feature_list, axis=0)
        y = np.array(labels, dtype=np.float32)
        return X, y, feature_names
