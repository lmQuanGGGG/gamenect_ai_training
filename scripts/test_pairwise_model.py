import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

def load_model(model_dir):
    """Load trained pairwise model"""
    model_dir = Path(model_dir)
    
    model = joblib.load(model_dir / 'pairwise_compatibility_model.pkl')
    scaler = joblib.load(model_dir / 'pairwise_scaler.pkl')
    feature_names = joblib.load(model_dir / 'pairwise_feature_names.pkl')
    
    return model, scaler, feature_names

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates"""
    from geopy.distance import geodesic
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 1000
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except:
        return 1000

def create_pair_features(user1, user2):
    """Create features for a user pair"""
    features = {}
    
    # Age
    age_diff = abs(user1['age'] - user2['age'])
    features['age_diff'] = age_diff
    features['age_compatible'] = 1 if age_diff <= 5 else 0
    features['age_within_10'] = 1 if age_diff <= 10 else 0
    
    # Distance
    distance = calculate_distance(
        user1.get('latitude'), user1.get('longitude'),
        user2.get('latitude'), user2.get('longitude')
    )
    features['distance_km'] = distance
    features['distance_within_10km'] = 1 if distance <= 10 else 0
    features['distance_within_50km'] = 1 if distance <= 50 else 0
    features['distance_within_100km'] = 1 if distance <= 100 else 0
    
    # Gender matching
    user1_looking_for = user1.get('interestedInGender', 'Tất cả')
    user2_looking_for = user2.get('interestedInGender', 'Tất cả')
    
    gender_match_1_to_2 = (user1_looking_for == 'Tất cả' or user1_looking_for == user2['gender'])
    gender_match_2_to_1 = (user2_looking_for == 'Tất cả' or user2_looking_for == user1['gender'])
    
    features['gender_match'] = 1 if (gender_match_1_to_2 and gender_match_2_to_1) else 0
    features['one_way_gender_match'] = 1 if (gender_match_1_to_2 or gender_match_2_to_1) else 0
    
    # Game style
    features['same_game_style'] = 1 if user1['game_style'] == user2['game_style'] else 0
    
    compatible_styles = {
        'Casual': ['Casual', 'Vừa chơi vừa học'],
        'Competitive': ['Competitive', 'Pro Player'],
        'Pro Player': ['Pro Player', 'Competitive'],
        'Streamer': ['Streamer', 'Casual', 'Vừa chơi vừa học'],
        'Vừa chơi vừa học': ['Casual', 'Vừa chơi vừa học', 'Streamer']
    }
    
    is_compatible = (user1['game_style'] in compatible_styles and 
                    user2['game_style'] in compatible_styles.get(user1['game_style'], []))
    features['compatible_game_style'] = 1 if is_compatible else 0
    
    # Skill level
    win_rate_diff = abs(user1['win_rate'] - user2['win_rate'])
    features['win_rate_diff'] = win_rate_diff
    features['similar_skill'] = 1 if win_rate_diff <= 20 else 0
    features['same_rank'] = 1 if user1['rank'] == user2['rank'] else 0
    
    # Activity
    play_time_ratio = min(user1['play_time'], user2['play_time']) / max(user1['play_time'], user2['play_time'], 1)
    features['play_time_ratio'] = play_time_ratio
    features['both_active'] = 1 if (user1['play_time'] > 1000 and user2['play_time'] > 1000) else 0
    
    # Engagement
    features['both_premium'] = 1 if (user1['is_premium'] and user2['is_premium']) else 0
    features['both_verified'] = 1 if (user1['is_verified'] and user2['is_verified']) else 0
    
    avg_profile = ((user1['num_interests'] + user2['num_interests']) / 10 +
                   (user1['num_photos'] + user2['num_photos']) / 6 +
                   (user1['bio_length'] + user2['bio_length']) / 200) / 3
    features['avg_profile_completeness'] = min(avg_profile, 1.0)
    
    # Similarities
    features['interest_count_similarity'] = 1 - abs(user1['num_interests'] - user2['num_interests']) / 10
    features['game_count_similarity'] = 1 - abs(user1['num_games'] - user2['num_games']) / 5
    features['same_looking_for'] = 1 if user1['looking_for'] == user2['looking_for'] else 0
    
    # Popularity
    features['user1_popularity'] = (user1['profile_views'] + user1['like_count']) / 100
    features['user2_popularity'] = (user2['profile_views'] + user2['like_count']) / 100
    features['popularity_balance'] = 1 - abs(features['user1_popularity'] - features['user2_popularity'])
    
    # Height
    height_diff = abs(user1['height'] - user2['height'])
    features['height_diff'] = height_diff
    features['height_compatible'] = 1 if height_diff <= 15 else 0
    
    # Age preferences
    age_pref_match_1_to_2 = (user1['minAge'] <= user2['age'] <= user1['maxAge'])
    age_pref_match_2_to_1 = (user2['minAge'] <= user1['age'] <= user2['maxAge'])
    
    features['age_preference_match'] = 1 if (age_pref_match_1_to_2 and age_pref_match_2_to_1) else 0
    features['one_way_age_match'] = 1 if (age_pref_match_1_to_2 or age_pref_match_2_to_1) else 0
    
    # Distance preferences
    distance_pref_match_1 = distance <= user1['maxDistance']
    distance_pref_match_2 = distance <= user2['maxDistance']
    
    features['distance_preference_match'] = 1 if (distance_pref_match_1 and distance_pref_match_2) else 0
    features['one_way_distance_match'] = 1 if (distance_pref_match_1 or distance_pref_match_2) else 0
    
    return features

def predict_compatibility(user1, user2, model, scaler, feature_names):
    """Predict compatibility between two users"""
    features = create_pair_features(user1, user2)
    
    # Create DataFrame with proper feature order
    X = pd.DataFrame([{col: features.get(col, 0) for col in feature_names}])
    
    # Xử lý NaN và inf values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    compatibility_score = probability[1] * 100
    
    return prediction, compatibility_score

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / 'models'
    data_file = base_dir / 'data' / 'raw' / 'users.json'
    
    # Load model
    print("Loading pairwise compatibility model...")
    model, scaler, feature_names = load_model(model_dir)
    
    # Load users
    with open(data_file, 'r', encoding='utf-8') as f:
        users_data = json.load(f)
    
    df = pd.DataFrame(users_data)
    
    # Prepare user data with safe defaults
    users = []
    for _, user in df.iterrows():
        # Xử lý location safely
        location = user.get('location', {})
        if not isinstance(location, dict):
            location = {}
        
        users.append({
            'user_id': user.get('__id', ''),
            'age': user.get('age', 25),
            'gender': user.get('gender', 'Unknown'),
            'height': user.get('height', 170),
            'play_time': user.get('playTime', 0),
            'win_rate': user.get('winRate', 50),
            'total_matches': user.get('totalMatches', 0),
            'friend_count': user.get('friendCount', 0),
            'profile_views': user.get('profileViews', 0),
            'like_count': user.get('likeCount', 0),
            'match_count': user.get('matchCount', 0),
            'game_style': user.get('gameStyle', 'Casual'),
            'rank': user.get('rank', 'Unknown'),
            'looking_for': user.get('lookingFor', 'Bạn chơi game'),
            'is_premium': user.get('isPremium', False),
            'is_verified': user.get('isVerified', False),
            'num_interests': len(user.get('interests', [])) if isinstance(user.get('interests'), list) else 0,
            'num_games': len(user.get('favoriteGames', [])) if isinstance(user.get('favoriteGames'), list) else 0,
            'num_photos': len(user.get('additionalPhotos', [])) if isinstance(user.get('additionalPhotos'), list) else 0,
            'bio_length': len(str(user.get('bio', ''))) if pd.notna(user.get('bio')) else 0,
            'latitude': location.get('latitude'),
            'longitude': location.get('longitude'),
            # Xử lý interestedInGender với default value
            'interestedInGender': user.get('interestedInGender', 'Tất cả') if pd.notna(user.get('interestedInGender')) else 'Tất cả',
            'minAge': user.get('minAge', 18),
            'maxAge': user.get('maxAge', 99),
            'maxDistance': user.get('maxDistance', 50.0)
        })
    
    # Test with random pairs
    print("\n" + "="*80)
    print("Testing Pairwise Compatibility Model")
    print("="*80)
    
    for i in range(5):
        user1, user2 = np.random.choice(users, 2, replace=False)
        
        prediction, score = predict_compatibility(user1, user2, model, scaler, feature_names)
        
        print(f"\n{'='*80}")
        print(f"Pair {i+1}:")
        print(f"\nUser 1: {user1['user_id']}")
        print(f"  Age: {user1['age']}, Gender: {user1['gender']}, Looking for: {user1.get('interestedInGender', 'N/A')}")
        print(f"  Game Style: {user1['game_style']}, Rank: {user1['rank']}")
        print(f"  Win Rate: {user1['win_rate']}%, Play Time: {user1['play_time']}h")
        
        print(f"\nUser 2: {user2['user_id']}")
        print(f"  Age: {user2['age']}, Gender: {user2['gender']}, Looking for: {user2.get('interestedInGender', 'N/A')}")
        print(f"  Game Style: {user2['game_style']}, Rank: {user2['rank']}")
        print(f"  Win Rate: {user2['win_rate']}%, Play Time: {user2['play_time']}h")
        
        print(f"\n{'🔥' if score >= 70 else '👍' if score >= 50 else '👎'} Compatibility Score: {score:.1f}%")
        print(f"Prediction: {'✅ COMPATIBLE' if prediction == 1 else '❌ INCOMPATIBLE'}")

if __name__ == "__main__":
    main()
