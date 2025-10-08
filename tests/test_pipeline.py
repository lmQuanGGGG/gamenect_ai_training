import numpy as np
from pipelines.feature_engineering import FeatureEngineer


def test_feature_engineer_build_pair_features():
    ky_su = FeatureEngineer()
    user1 = {
        "favoriteGames": ["League of Legends", "Valorant"],
        "interests": ["Anime/Manga", "Công nghệ"],
        "rank": "Trùm Cuối",
        "gender": "Nam",
        "age": 24,
        "height": 175,
        "winRate": 70,
        "playTime": 210,
        "gameStyle": "Competitive",
        "lookingFor": "Đồng đội lâu dài",
        "latitude": 21.0285,
        "longitude": 105.8542,
        "maxDistance": 50,
        "showDistance": True,
        "isVerified": True,
    }
    user2 = {
        "favoriteGames": ["League of Legends", "CS:GO"],
        "interests": ["Anime/Manga", "Thể thao"],
        "rank": "Quái vật cân team",
        "gender": "Nam",
        "age": 26,
        "height": 180,
        "winRate": 65,
        "playTime": 180,
        "gameStyle": "Competitive",
        "lookingFor": "Bạn chơi game",
        "latitude": 21.03,
        "longitude": 105.85,
        "maxDistance": 30,
        "showDistance": True,
        "isVerified": False,
    }

    features, names = ky_su.build_pair_features(user1, user2)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == len(names)
    assert "shared_games" in names
