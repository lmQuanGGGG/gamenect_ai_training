import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

def load_model(model_dir):
    """Load trained model and preprocessors"""
    model_dir = Path(model_dir)
    
    model = joblib.load(model_dir / 'user_compatibility_model.pkl')
    scaler = joblib.load(model_dir / 'scaler.pkl')
    label_encoders = joblib.load(model_dir / 'label_encoders.pkl')
    
    return model, scaler, label_encoders

def predict_compatibility(user_data, model, scaler, label_encoders):
    """Predict compatibility for a user"""
    # Encode categorical features
    categorical_cols = ['gender', 'game_style', 'rank', 'looking_for']
    
    for col in categorical_cols:
        if col in user_data and col in label_encoders:
            try:
                user_data[f'{col}_encoded'] = label_encoders[col].transform([user_data[col]])[0]
            except:
                user_data[f'{col}_encoded'] = 0
    
    # Prepare features
    numerical_features = [
        'age', 'height', 'play_time', 'win_rate', 'total_matches',
        'friend_count', 'profile_views', 'like_count', 'match_count',
        'num_interests', 'num_games', 'num_photos', 'bio_length'
    ]
    
    encoded_features = [f'{col}_encoded' for col in categorical_cols]
    feature_cols = numerical_features + encoded_features
    
    # Create feature vector
    X = np.array([[user_data.get(col, 0) for col in feature_cols]])
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return prediction, probability

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / 'models'
    data_file = base_dir / 'data' / 'processed' / 'users_features.csv'
    
    # Load model
    print("Loading model...")
    model, scaler, label_encoders = load_model(model_dir)
    
    # Load test data
    df = pd.read_csv(data_file)
    
    # Test with a few random users
    print("\nTesting with random users:")
    for _ in range(5):
        user = df.sample(1).iloc[0].to_dict()
        prediction, probability = predict_compatibility(user, model, scaler, label_encoders)
        
        print(f"\nUser ID: {user['user_id']}")
        print(f"Age: {user['age']}, Gender: {user['gender']}")
        print(f"Game Style: {user['game_style']}, Rank: {user['rank']}")
        print(f"Win Rate: {user['win_rate']}%, Play Time: {user['play_time']}h")
        print(f"Prediction: {'High Compatibility' if prediction == 1 else 'Low Compatibility'}")
        print(f"Probability: {probability}")

if __name__ == "__main__":
    main()
