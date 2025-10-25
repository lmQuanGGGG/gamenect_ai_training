"""
FastAPI server để serve trained models
"""
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import io
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="Gamenect ML API",
    description="Compatibility prediction API for Gamenect",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
pairwise_model = None
pairwise_scaler = None
feature_names = None

@app.on_event("startup")
async def load_models():
    """Load trained models on startup"""
    global pairwise_model, pairwise_scaler, feature_names
    
    model_dir = Path(__file__).parent.parent / 'models'
    
    try:
        print("Loading pairwise model...")
        pairwise_model = joblib.load(model_dir / 'pairwise_compatibility_model.pkl')
        pairwise_scaler = joblib.load(model_dir / 'pairwise_scaler.pkl')
        feature_names = joblib.load(model_dir / 'pairwise_feature_names.pkl')
        print(f"Models loaded! Features: {len(feature_names)}")
    except Exception as e:
        print(f"Error loading models: {e}")

class UserProfile(BaseModel):
    user_id: str
    age: int
    gender: str
    height: float = 170.0
    play_time: int = 0
    win_rate: float = 50.0
    game_style: str = "Casual"
    rank: str = "Unknown"
    looking_for: str = "Bạn chơi game"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    interestedInGender: str = "Tất cả"
    minAge: int = 18
    maxAge: int = 99
    maxDistance: float = 50.0
    isPremium: bool = False
    isVerified: bool = False
    num_interests: int = 0
    num_games: int = 0
    num_photos: int = 0
    bio_length: int = 0
    profile_views: int = 0
    like_count: int = 0
    match_count: int = 0
    total_matches: int = 0
    friend_count: int = 0

class RecommendationRequest(BaseModel):
    current_user: UserProfile
    candidate_users: List[UserProfile]
    top_k: int = 10000000
    preference_mode: str = "balanced"  # balanced, mentor_mentee, same_style, nearby

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Gamenect ML API",
        "version": "1.0.0",
        "endpoints": ["/health", "/recommend", "/docs"]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": pairwise_model is not None,
        "features_count": len(feature_names) if feature_names else 0
    }

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance using geopy"""
    from geopy.distance import geodesic
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 1000
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except:
        return 1000

def create_features(user1: dict, user2: dict, preference_mode: str = 'balanced') -> dict:
    """Create features for prediction - ĐỒNG BỘ với train_user_model.py"""
    f = {}
    
    # Age features
    age_diff = abs(user1['age'] - user2['age'])
    f['age_diff'] = age_diff
    f['age_compatible'] = 1 if age_diff <= 5 else 0
    f['age_within_10'] = 1 if age_diff <= 10 else 0
    
    # Distance features
    dist = calculate_distance(
        user1.get('latitude'), user1.get('longitude'),
        user2.get('latitude'), user2.get('longitude')
    )
    f['distance_km'] = min(dist, 1000)
    f['distance_within_10km'] = 1 if dist <= 10 else 0
    f['distance_within_50km'] = 1 if dist <= 50 else 0
    f['distance_within_100km'] = 1 if dist <= 100 else 0
    
    # Gender matching
    g1 = user1.get('interestedInGender', 'Tất cả')
    g2 = user2.get('interestedInGender', 'Tất cả')
    m12 = (g1 == 'Tất cả' or g1 == user2['gender'])
    m21 = (g2 == 'Tất cả' or g2 == user1['gender'])
    f['gender_match'] = 1 if (m12 and m21) else 0
    f['one_way_gender_match'] = 1 if (m12 or m21) else 0
    
    # Game style
    f['same_game_style'] = 1 if user1['game_style'] == user2['game_style'] else 0
    compat_styles = {
        'Casual': ['Casual', 'Vừa chơi vừa học'],
        'Competitive': ['Competitive', 'Pro Player'],
        'Pro Player': ['Pro Player', 'Competitive'],
        'Streamer': ['Streamer', 'Casual', 'Vừa chơi vừa học'],
        'Vừa chơi vừa học': ['Casual', 'Vừa chơi vừa học', 'Streamer']
    }
    f['compatible_game_style'] = 1 if (
        user1['game_style'] in compat_styles and
        user2['game_style'] in compat_styles.get(user1['game_style'], [])
    ) else 0
    
    # Skill level
    win_rate_diff = abs(user1['win_rate'] - user2['win_rate'])
    f['win_rate_diff'] = win_rate_diff
    f['similar_skill'] = 1 if win_rate_diff <= 20 else 0
    f['same_rank'] = 1 if user1['rank'] == user2['rank'] else 0
    
    # Activity
    play_time_ratio = min(user1['play_time'], user2['play_time']) / max(user1['play_time'], user2['play_time'], 1)
    f['play_time_ratio'] = play_time_ratio
    f['both_active'] = 1 if (user1['play_time'] > 1000 and user2['play_time'] > 1000) else 0
    
    # Premium/Verified
    f['both_premium'] = 1 if (user1.get('isPremium', False) and user2.get('isPremium', False)) else 0
    f['both_verified'] = 1 if (user1.get('isVerified', False) and user2.get('isVerified', False)) else 0
    
    # Profile completeness
    avg_profile = ((user1.get('num_interests', 0) + user2.get('num_interests', 0)) / 10 +
                   (user1.get('num_photos', 0) + user2.get('num_photos', 0)) / 6 +
                   (user1.get('bio_length', 0) + user2.get('bio_length', 0)) / 200) / 3
    f['avg_profile_completeness'] = min(avg_profile, 1.0)
    
    # Similarities
    f['interest_count_similarity'] = 1 - abs(user1.get('num_interests', 0) - user2.get('num_interests', 0)) / 10
    f['game_count_similarity'] = 1 - abs(user1.get('num_games', 0) - user2.get('num_games', 0)) / 5
    f['same_looking_for'] = 1 if user1['looking_for'] == user2['looking_for'] else 0
    
    # Popularity
    u1_pop = (user1.get('profile_views', 0) + user1.get('like_count', 0)) / 100
    u2_pop = (user2.get('profile_views', 0) + user2.get('like_count', 0)) / 100
    f['user1_popularity'] = min(u1_pop, 10)
    f['user2_popularity'] = min(u2_pop, 10)
    f['popularity_balance'] = 1 - min(abs(u1_pop - u2_pop), 1)
    
    # Height
    height_diff = abs(user1['height'] - user2['height'])
    f['height_diff'] = height_diff
    f['height_compatible'] = 1 if height_diff <= 15 else 0
    
    # Age preferences
    ap12 = (user1.get('minAge', 18) <= user2['age'] <= user1.get('maxAge', 99))
    ap21 = (user2.get('minAge', 18) <= user1['age'] <= user2.get('maxAge', 99))
    f['age_preference_match'] = 1 if (ap12 and ap21) else 0
    f['one_way_age_match'] = 1 if (ap12 or ap21) else 0
    
    # Distance preferences
    dp1 = dist <= user1.get('maxDistance', 50)
    dp2 = dist <= user2.get('maxDistance', 50)
    f['distance_preference_match'] = 1 if (dp1 and dp2) else 0
    f['one_way_distance_match'] = 1 if (dp1 or dp2) else 0
    
    # Deal breakers
    deal_breakers = 0
    if f['gender_match'] == 0: deal_breakers += 1
    if not (dp1 or dp2): deal_breakers += 1
    if f['age_preference_match'] == 0: deal_breakers += 1
    f['deal_breaker_count'] = deal_breakers
    f['has_deal_breakers'] = 1 if deal_breakers > 0 else 0
    
    # Mentor/mentee relationship
    skill_gap = user1.get('win_rate', 50) - user2.get('win_rate', 50)
    xp_gap = user1.get('play_time', 0) - user2.get('play_time', 0)
    f['can_mentor_12'] = 1 if (skill_gap > 15 and xp_gap > 500) else 0
    f['can_mentor_21'] = 1 if (skill_gap < -15 and xp_gap < -500) else 0
    f['has_mentor_relationship'] = 1 if (f['can_mentor_12'] or f['can_mentor_21']) else 0
    if f['has_mentor_relationship']:
        f['mentor_quality'] = min((abs(skill_gap)/50) + (abs(xp_gap)/2000), 1.0)
    else:
        f['mentor_quality'] = 0.0
    
    # Complementary roles
    style1 = user1.get('game_style')
    style2 = user2.get('game_style')
    rc = 0.0
    def s(a,b): return (style1==a and style2==b) or (style1==b and style2==a)
    if s('Competitive','Casual'): rc = 0.7
    if s('Pro Player','Competitive'): rc = max(rc, 0.9)
    if s('Streamer','Casual'): rc = max(rc, 0.8)
    if s('Vừa chơi vừa học','Competitive'): rc = max(rc, 0.85)
    if style1 == style2: rc = max(rc, 0.6)
    f['role_compatibility'] = rc
    f['complementary_roles'] = 1 if rc >= 0.7 else 0
    
    # Activity gap & engagement
    f['activity_gap'] = abs(np.log1p(user1.get('play_time', 0)) - np.log1p(user2.get('play_time', 0)))
    eng1 = (user1.get('profile_views', 0) + user1.get('like_count', 0) + user1.get('match_count', 0)) / 3
    eng2 = (user2.get('profile_views', 0) + user2.get('like_count', 0) + user2.get('match_count', 0)) / 3
    f['avg_engagement'] = (eng1 + eng2) / 2
    f['engagement_gap'] = abs(eng1 - eng2)
    
    # Compatibility factor score
    factors = [
        f['gender_match'],
        f['age_preference_match'],
        f['distance_preference_match'],
        f['compatible_game_style'],
        f['similar_skill'],
        1 if dist <= 50 else 0,
        1 if age_diff <= 7 else 0
    ]
    f['compatibility_factor_score'] = sum(factors) / len(factors)
    
    # Preference-based scoring
    if preference_mode == 'mentor_mentee':
        f['preference_score'] = (
            f['has_mentor_relationship'] * 0.4 +
            f['mentor_quality'] * 0.3 +
            (1 - min(win_rate_diff / 50, 1)) * 0.3
        )
    elif preference_mode == 'same_style':
        f['preference_score'] = (
            f['same_game_style'] * 0.3 +
            f['compatible_game_style'] * 0.2 +
            rc * 0.3 +
            f['similar_skill'] * 0.2
        )
    elif preference_mode == 'nearby':
        dist_score = max(0, 1 - dist / 100)
        f['preference_score'] = (
            dist_score * 0.5 +
            f['distance_preference_match'] * 0.3 +
            f['age_preference_match'] * 0.2
        )
    else:  # balanced
        f['preference_score'] = f['compatibility_factor_score']
    
    return f

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    """Get recommendations with preference mode"""
    if pairwise_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    valid_modes = ['balanced', 'mentor_mentee', 'same_style', 'nearby']
    if request.preference_mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid preference_mode. Must be one of: {valid_modes}"
        )
    
    try:
        current_user = request.current_user.dict()
        recommendations = []
        
        for candidate in request.candidate_users:
            candidate_dict = candidate.dict()
            
            # Create features với preference mode
            features = create_features(current_user, candidate_dict, request.preference_mode)
            
            # Predict
            X = pd.DataFrame([{col: features.get(col, 0) for col in feature_names}])
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            X_scaled = pairwise_scaler.transform(X)
            
            probability = pairwise_model.predict_proba(X_scaled)[0]
            base_score = float(probability[1] * 100)
            
            # Apply preference weight
            preference_weight = features.get('preference_score', 0.5)
            final_score = base_score * 0.7 + preference_weight * 100 * 0.3
            
            recommendations.append({
                'user_id': candidate_dict['user_id'],
                'compatibility_score': final_score,
                'base_score': base_score,
                'preference_score': preference_weight * 100,
                'distance_km': float(features.get('distance_km', 0)),
                'age_diff': int(features.get('age_diff', 0)),
                'win_rate_diff': float(features.get('win_rate_diff', 0)),
                'role_compatibility': float(features.get('role_compatibility', 0)),
                'has_mentor_relationship': bool(features.get('has_mentor_relationship', 0)),
                'complementary_roles': bool(features.get('complementary_roles', 0))
            })
        
        # Sort by final score
        recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        return {
            "recommendations": recommendations[:request.top_k],
            "total_candidates": len(request.candidate_users),
            "preference_mode": request.preference_mode
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_model")
async def reload_model(authorization: str = Header(None), model: UploadFile = File(None)):
    # Xác thực token
    token = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2:
            token = parts[1]
    if token != os.getenv("MODEL_DOWNLOAD_TOKEN"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Nhận file model trực tiếp
    if model is not None:
        try:
            bio = io.BytesIO(await model.read())
            bio.seek(0)
            new = joblib.load(bio)
            global pairwise_model, scaler, feature_names
            if isinstance(new, dict):
                pairwise_model = new.get("model", pairwise_model)
                scaler = new.get("scaler", scaler)
                feature_names = new.get("feature_names", feature_names)
            else:
                pairwise_model = new
            print("Model reload request received and loaded!")  # Log xác nhận
            return {"status": "ok", "message": "model reloaded (file upload)"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"reload failed: {e}")

    raise HTTPException(status_code=400, detail="No model file provided")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)