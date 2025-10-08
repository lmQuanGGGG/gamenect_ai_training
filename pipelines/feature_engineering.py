from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np


class FeatureEngineer:
    """Xử lý và biến đổi dữ liệu người dùng thành vector đặc trưng."""

    def __init__(self) -> None:
        self.rank_map = {
            "Gà Mờ": 0,
            "Tập Sự Truyền Thuyết": 1,
            "Chiến Binh Phèn": 2,
            "Thánh Né": 3,
            "Quái vật cân team": 4,
            "Trùm Cuối": 5,
            "Thượng Đế AFK": 6,
        }
        self.gender_map = {"Nam": 0, "Nữ": 1, "Khác": 2}
        self.game_style_map = {
            "Casual": 0,
            "Competitive": 1,
            "Streamer": 2,
            "Pro Player": 3,
            "Vừa chơi vừa học": 4,
        }
        self.looking_for_map = {
            "Bạn chơi game": 0,
            "Hẹn hò": 1,
            "Cả hai": 2,
            "Người chỉ dạy": 3,
            "Đồng đội lâu dài": 4,
        }

    def encode_user(self, user: Dict[str, Any]) -> Dict[str, float]:
        favorite_games = user.get("favoriteGames", [])
        interests = user.get("interests", [])
        features = {
            "rank": self.rank_map.get(user.get("rank"), 0),
            "gender": self.gender_map.get(user.get("gender"), 0),
            "age": float(user.get("age", 24)),
            "height": float(user.get("height", 165)),
            "win_rate": float(user.get("winRate", 50)) / 100.0,
            "play_time": float(user.get("playTime", 120)) / 480.0,
            "game_style": self.game_style_map.get(user.get("gameStyle"), 0),
            "looking_for": self.looking_for_map.get(user.get("lookingFor"), 0),
            "max_distance": float(user.get("maxDistance", 50)) / 200.0,
            "show_distance": 1.0 if user.get("showDistance", True) else 0.0,
            "is_verified": 1.0 if user.get("isVerified", False) else 0.0,
            "is_online": 1.0 if user.get("isOnline", False) else 0.0,
            "read_receipts": 1.0 if user.get("readReceiptsEnabled", False) else 0.0,
            "boost_count": float(user.get("boostCount", 0)) / 5.0,
            "super_likes_remaining": float(user.get("superLikesRemaining", 5)) / 5.0,
            "favorite_games_count": float(len(favorite_games)) / 10.0,
            "interests_count": float(len(interests)) / 10.0,
        }
        return features

    def build_pair_features(
        self, user1: Dict[str, Any], user2: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        u1 = self.encode_user(user1)
        u2 = self.encode_user(user2)
        shared_games = len(set(user1.get("favoriteGames", [])) & set(user2.get("favoriteGames", [])))
        shared_interests = len(set(user1.get("interests", [])) & set(user2.get("interests", [])))
        distance = self._compute_distance(user1, user2)

        features = []
        names = []

        for prefix, encoded in [("user1", u1), ("user2", u2)]:
            for k, v in encoded.items():
                features.append(v)
                names.append(f"{prefix}_{k}")

        features.extend(
            [
                float(shared_games) / 10.0,
                float(shared_interests) / 10.0,
                distance,
                abs(u1["age"] - u2["age"]) / 50.0,
                abs(u1["rank"] - u2["rank"]) / len(self.rank_map),
            ]
        )
        names.extend(
            [
                "shared_games",
                "shared_interests",
                "distance",
                "age_gap",
                "rank_gap",
            ]
        )

        return np.array(features, dtype=np.float32), names

    def _compute_distance(self, user1: Dict[str, Any], user2: Dict[str, Any]) -> float:
        lat1, lon1 = user1.get("latitude"), user1.get("longitude")
        lat2, lon2 = user2.get("latitude"), user2.get("longitude")
        if None in (lat1, lon1, lat2, lon2):
            return 1.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return min(km / 500.0, 1.0)
