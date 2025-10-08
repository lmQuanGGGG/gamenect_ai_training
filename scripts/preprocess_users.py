import json
import pandas as pd
from pathlib import Path

def load_users_data(file_path):
    """Load users data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_features(users_data):
    """Extract relevant features from users data"""
    features = []
    
    for user in users_data:
        feature = {
            'user_id': user.get('__id', user.get('uid', user.get('id', ''))),
            'age': user.get('age', 0),
            'gender': user.get('gender', ''),
            'height': user.get('height', 0),
            'play_time': user.get('playTime', 0),
            'win_rate': user.get('winRate', 0),
            'total_matches': user.get('totalMatches', 0),
            'friend_count': user.get('friendCount', 0),
            'profile_views': user.get('profileViews', 0),
            'like_count': user.get('likeCount', 0),
            'match_count': user.get('matchCount', 0),
            'game_style': user.get('gameStyle', ''),
            'rank': user.get('rank', ''),
            'looking_for': user.get('lookingFor', ''),
            'is_premium': user.get('isPremium', False),
            'is_verified': user.get('isVerified', False),
            'num_interests': len(user.get('interests', [])),
            'num_games': len(user.get('favoriteGames', [])),
            'num_photos': len(user.get('additionalPhotos', [])),
            'bio_length': len(user.get('bio', '')),
        }
        features.append(feature)
    
    return pd.DataFrame(features)

def save_processed_data(df, output_path):
    """Save processed data to CSV"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Processed data saved to {output_path}")
    print(f"Total records: {len(df)}")
    print(f"\nData info:")
    print(df.info())

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'raw' / 'users.json'
    output_file = base_dir / 'data' / 'processed' / 'users_features.csv'
    
    # Create output directory if not exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    print("Loading users data...")
    users_data = load_users_data(input_file)
    
    print("Extracting features...")
    df = extract_features(users_data)
    
    # Save processed data
    save_processed_data(df, output_file)
    
    print("\nSample data:")
    print(df.head())
    
    print("\nBasic statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()
