"""
FastAPI server — GameNect ML API v3.0
Synced với train_real_data.py v3.0 (62 features, all Firebase signals)
"""
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib, pandas as pd, numpy as np, io, os
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
MODEL_DIR   = BASE_DIR / "models"

pairwise_model  = None
pairwise_scaler = None
feature_names   = []

def load_models():
    global pairwise_model, pairwise_scaler, feature_names
    try:
        pairwise_model  = joblib.load(MODEL_DIR / "pairwise_compatibility_model.pkl")
        pairwise_scaler = joblib.load(MODEL_DIR / "pairwise_scaler.pkl")
        feature_names   = joblib.load(MODEL_DIR / "pairwise_feature_names.pkl")
        print(f"✅ Model loaded: {len(feature_names)} features")
    except Exception as e:
        print(f"⚠️  Model load failed: {e}")

load_models()

# ─────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Gamenect ML API",
    description="Compatibility prediction — real Firebase data trained (62 features)",
    version="3.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ─────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────
class UserProfile(BaseModel):
    user_id: str
    age: int
    gender: str
    height: float = 170.0
    play_time: int = 0              # playTime (giờ)
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

    # Premium / verified
    isPremium: bool = False
    isVerified: bool = False
    showOnlineStatus: bool = False

    # Activity
    isOnline: bool = False
    lastSeen: Optional[str] = None  # ISO datetime

    # Lists — Jaccard similarity
    favoriteGames: List[str] = []
    interests: List[str] = []

    # Social proof từ Firebase profile
    like_count: int = 0             # likeCount
    match_count: int = 0            # matchCount
    friend_count: int = 0           # friendCount
    profile_views: int = 0          # profileViews
    super_like_count: int = 0       # superLikeCount

    # Backward-compatible
    num_photos: int = 0
    bio_length: int = 0
    num_interests: int = 0
    num_games: int = 0
    total_matches: int = 0


class RecommendationRequest(BaseModel):
    current_user: UserProfile
    candidate_users: List[UserProfile]
    top_k: int = 10000000
    preference_mode: str = "balanced"  # balanced | mentor_mentee | same_style | nearby


# ─────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────
def calculate_distance(lat1, lon1, lat2, lon2) -> float:
    from geopy.distance import geodesic
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return 500.0
    try:
        if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
            return 500.0
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except Exception:
        return 500.0


def days_since(date_str) -> int:
    if not date_str:
        return 999
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        return max((datetime.now(timezone.utc) - dt).days, 0)
    except Exception:
        return 999


# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — 62 features
# ĐỒNG BỘ HOÀN TOÀN với scripts/train_real_data.py v3.0
# ─────────────────────────────────────────────────────────────────────
def create_features(u1: dict, u2: dict, mode: str = "balanced") -> dict:
    f = {}

    # ── GROUP 1: TUỔI (Homophily - McPherson 2001) ─────────────────
    age1 = float(u1.get("age", 25))
    age2 = float(u2.get("age", 25))
    ad   = abs(age1 - age2)
    f["age_diff"]          = ad
    f["age_compatible_5"]  = 1 if ad <= 5  else 0
    f["age_compatible_8"]  = 1 if ad <= 8  else 0
    f["age_compatible_12"] = 1 if ad <= 12 else 0
    mn1 = float(u1.get("minAge", 18)); mx1 = float(u1.get("maxAge", 60))
    mn2 = float(u2.get("minAge", 18)); mx2 = float(u2.get("maxAge", 60))
    f["age_pref_match"]    = 1 if (mn1 <= age2 <= mx1 and mn2 <= age1 <= mx2) else 0
    f["age_pref_one_way"]  = 1 if (mn1 <= age2 <= mx1 or  mn2 <= age1 <= mx2) else 0

    # ── GROUP 2: KHOẢNG CÁCH (Proximity - Festinger 1950) ──────────
    dist = calculate_distance(
        u1.get("latitude"), u1.get("longitude"),
        u2.get("latitude"), u2.get("longitude")
    )
    f["distance_km"]       = min(dist, 1000)
    f["dist_within_5km"]   = 1 if dist <= 5   else 0
    f["dist_within_20km"]  = 1 if dist <= 20  else 0
    f["dist_within_50km"]  = 1 if dist <= 50  else 0
    f["dist_within_100km"] = 1 if dist <= 100 else 0
    md1 = float(u1.get("maxDistance", 50))
    md2 = float(u2.get("maxDistance", 50))
    f["dist_pref_match"]   = 1 if (dist <= md1 and dist <= md2) else 0
    f["dist_pref_one_way"] = 1 if (dist <= md1 or  dist <= md2) else 0

    # ── GROUP 3: GIỚI TÍNH ─────────────────────────────────────────
    g1 = u1.get("gender", "Unknown")
    g2 = u2.get("gender", "Unknown")
    p1 = u1.get("interestedInGender", "Tất cả") or "Tất cả"
    p2 = u2.get("interestedInGender", "Tất cả") or "Tất cả"
    m12 = (p1 == "Tất cả" or p1 == g2)
    m21 = (p2 == "Tất cả" or p2 == g1)
    f["gender_match"]      = 1 if (m12 and m21) else 0
    f["gender_one_way"]    = 1 if (m12 or m21)  else 0
    f["same_gender"]       = 1 if g1 == g2       else 0

    # ── GROUP 4: KỸ NĂNG GAME (SBMM - Elo 1978) ───────────────────
    wr1 = float(u1.get("win_rate", 50))
    wr2 = float(u2.get("win_rate", 50))
    wrd = abs(wr1 - wr2)
    f["win_rate_diff"]     = wrd
    f["skill_similar_15"]  = 1 if wrd <= 15 else 0
    f["skill_similar_25"]  = 1 if wrd <= 25 else 0
    f["same_rank"]         = 1 if u1.get("rank", "") == u2.get("rank", "") else 0
    sg  = wr1 - wr2
    pt1 = float(u1.get("play_time", 0))
    pt2 = float(u2.get("play_time", 0))
    xg  = pt1 - pt2
    f["can_mentor_12"]       = 1 if (sg > 15  and xg > 500)  else 0
    f["can_mentor_21"]       = 1 if (sg < -15 and xg < -500) else 0
    f["has_mentor_relation"] = 1 if (f["can_mentor_12"] or f["can_mentor_21"]) else 0

    # ── GROUP 5: PHONG CÁCH CHƠI (Self-Categorization - Turner 1987)
    s1 = u1.get("game_style", "Unknown")
    s2 = u2.get("game_style", "Unknown")
    f["same_game_style"] = 1 if s1 == s2 else 0
    cm = {
        "Casual":            {"Casual", "Vừa chơi vừa học"},
        "Competitive":       {"Competitive", "Pro Player"},
        "Pro Player":        {"Pro Player", "Competitive"},
        "Streamer":          {"Streamer", "Casual", "Vừa chơi vừa học"},
        "Vừa chơi vừa học": {"Casual", "Vừa chơi vừa học", "Streamer"},
    }
    f["compat_style"] = 1 if (s1 in cm and s2 in cm.get(s1, set())) else 0
    rc = 0.0
    def sty(a, b): return (s1==a and s2==b) or (s1==b and s2==a)
    if sty("Pro Player", "Competitive"):        rc = 0.9
    if sty("Vừa chơi vừa học", "Competitive"): rc = max(rc, 0.85)
    if sty("Streamer", "Casual"):               rc = max(rc, 0.8)
    if sty("Competitive", "Casual"):            rc = max(rc, 0.7)
    if s1 == s2:                                rc = max(rc, 0.6)
    f["role_compatibility"]  = rc
    f["complementary_roles"] = 1 if rc >= 0.75 else 0

    # ── GROUP 6: GAME & SỞ THÍCH CHUNG (Jaccard 1912) ─────────────
    gs1 = set(u1.get("favoriteGames", []) or [])
    gs2 = set(u2.get("favoriteGames", []) or [])
    ig  = gs1 & gs2; ug = gs1 | gs2
    f["shared_game_count"]   = len(ig)
    f["shared_game_jaccard"] = len(ig) / max(len(ug), 1)
    f["has_common_game"]     = 1 if ig else 0
    i1 = set(u1.get("interests", []) or [])
    i2 = set(u2.get("interests", []) or [])
    ii = i1 & i2; ui = i1 | i2
    f["shared_interest_count"]   = len(ii)
    f["shared_interest_jaccard"] = len(ii) / max(len(ui), 1)
    f["has_common_interest"]     = 1 if ii else 0

    # ── GROUP 7: MỤC ĐÍCH TÌM KIẾM ────────────────────────────────
    f["same_looking_for"] = 1 if u1.get("looking_for", "") == u2.get("looking_for", "") else 0

    # ── GROUP 8: HOẠT ĐỘNG ─────────────────────────────────────────
    f["play_time_ratio"]   = min(pt1, pt2) / max(pt1, pt2, 1)
    f["both_active_1000h"] = 1 if (pt1 > 1000 and pt2 > 1000) else 0
    f["activity_gap_log"]  = abs(np.log1p(pt1) - np.log1p(pt2))

    # ── GROUP 9: PROFILE COMPLETENESS (Signaling - Spence 1973) ────
    def comp(u):
        sc = 0.0
        if u.get("bio_length", 0) > 0:                        sc += 0.20
        if u.get("num_photos", 0) > 0:                        sc += 0.20
        if u.get("favoriteGames") or u.get("num_games", 0) > 0: sc += 0.15
        if u.get("interests") or u.get("num_interests", 0) > 0: sc += 0.15
        if u.get("isVerified"):                                sc += 0.15
        if u.get("latitude") is not None:                      sc += 0.15
        return round(sc, 2)
    c1 = comp(u1); c2 = comp(u2)
    f["avg_completeness"]  = (c1 + c2) / 2
    f["completeness_diff"] = abs(c1 - c2)

    # ── GROUP 10: POPULARITY & SOCIAL PROOF ────────────────────────
    lc1 = np.log1p(float(u1.get("like_count",    0)))
    lc2 = np.log1p(float(u2.get("like_count",    0)))
    mc1 = np.log1p(float(u1.get("match_count",   0)))
    mc2 = np.log1p(float(u2.get("match_count",   0)))
    pv1 = np.log1p(float(u1.get("profile_views", 0)))
    pv2 = np.log1p(float(u2.get("profile_views", 0)))
    fc1 = np.log1p(float(u1.get("friend_count",  0)))
    fc2 = np.log1p(float(u2.get("friend_count",  0)))
    f["avg_like_count"]    = (lc1 + lc2) / 2
    f["avg_match_count"]   = (mc1 + mc2) / 2
    f["avg_profile_views"] = (pv1 + pv2) / 2
    f["avg_friend_count"]  = (fc1 + fc2) / 2
    f["popularity_gap"]    = abs(lc1 - lc2)
    f["match_rate_u1"]     = float(u1.get("match_count", 0)) / max(float(u1.get("like_count", 1)), 1)
    f["match_rate_u2"]     = float(u2.get("match_count", 0)) / max(float(u2.get("like_count", 1)), 1)
    f["avg_match_rate"]    = (f["match_rate_u1"] + f["match_rate_u2"]) / 2
    f["avg_super_likes"]   = (float(u1.get("super_like_count", 0)) + float(u2.get("super_like_count", 0))) / 2

    # ── GROUP 11: PREMIUM / VERIFIED ───────────────────────────────
    f["both_verified"]    = 1 if (u1.get("isVerified")        and u2.get("isVerified"))        else 0
    f["either_verified"]  = 1 if (u1.get("isVerified")        or  u2.get("isVerified"))        else 0
    f["both_premium"]     = 1 if (u1.get("isPremium")         and u2.get("isPremium"))         else 0
    f["either_premium"]   = 1 if (u1.get("isPremium")         or  u2.get("isPremium"))         else 0
    f["both_show_online"] = 1 if (u1.get("showOnlineStatus")  and u2.get("showOnlineStatus"))  else 0

    # ── GROUP 12: ACTIVITY RECENCY ─────────────────────────────────
    d1 = days_since(u1.get("lastSeen"))
    d2 = days_since(u2.get("lastSeen"))
    f["days_since_seen_u1"] = min(d1, 365)
    f["days_since_seen_u2"] = min(d2, 365)
    f["both_active_7d"]     = 1 if (d1 <= 7  and d2 <= 7)  else 0
    f["both_active_30d"]    = 1 if (d1 <= 30 and d2 <= 30) else 0
    f["either_online"]      = 1 if (u1.get("isOnline") or  u2.get("isOnline")) else 0
    f["both_online"]        = 1 if (u1.get("isOnline") and u2.get("isOnline")) else 0

    # ── GROUP 13: DEAL BREAKERS & COMPOSITE ────────────────────────
    db = sum([f["gender_match"] == 0, f["dist_pref_match"] == 0, f["age_pref_match"] == 0])
    f["deal_breaker_count"] = db
    f["has_deal_breakers"]  = 1 if db > 0 else 0
    cf = [
        f["gender_match"], f["age_pref_match"], f["dist_pref_match"],
        f["compat_style"], f["skill_similar_25"], f["dist_within_50km"],
        f["age_compatible_8"], f["has_common_game"], f["same_looking_for"],
        f["either_verified"], f["both_active_30d"],
    ]
    f["compat_factor_score"] = sum(cf) / len(cf)

    # ── PREFERENCE SCORE (mode-specific weight) ─────────────────────
    if mode == "mentor_mentee":
        f["preference_score"] = f["has_mentor_relation"] * 0.4 + (1 - min(wrd/50, 1)) * 0.6
    elif mode == "same_style":
        f["preference_score"] = (f["same_game_style"]*0.3 + f["compat_style"]*0.2 +
                                  rc*0.3 + f["skill_similar_25"]*0.2)
    elif mode == "nearby":
        f["preference_score"] = (max(0, 1 - dist/100)*0.5 +
                                  f["dist_pref_match"]*0.3 + f["age_pref_match"]*0.2)
    else:  # balanced
        f["preference_score"] = f["compat_factor_score"]

    return f


# ─────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Gamenect ML API",
        "version": "3.0.0",
        "model_features": len(feature_names),
        "endpoints": ["/health", "/recommend", "/reload_model", "/docs"]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": pairwise_model is not None,
        "features_count": len(feature_names),
        "version": "3.0.0"
    }


@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    if pairwise_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    valid_modes = ["balanced", "mentor_mentee", "same_style", "nearby"]
    if request.preference_mode not in valid_modes:
        raise HTTPException(status_code=400,
                            detail=f"preference_mode phải là: {valid_modes}")
    try:
        current_user = request.current_user.dict()
        recommendations = []

        for candidate in request.candidate_users:
            cand = candidate.dict()
            features = create_features(current_user, cand, request.preference_mode)

            # Build feature vector theo đúng thứ tự model expect
            X = pd.DataFrame([{col: features.get(col, 0) for col in feature_names}])
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            X_scaled = pairwise_scaler.transform(X)

            prob       = pairwise_model.predict_proba(X_scaled)[0]
            base_score = float(prob[1] * 100)
            pref_w     = features.get("preference_score", 0.5)
            final      = base_score * 0.7 + pref_w * 100 * 0.3

            recommendations.append({
                # ── Core (backward-compatible) ──────────────────────
                "user_id":              cand["user_id"],
                "compatibility_score":  round(final, 2),
                "base_score":           round(base_score, 2),
                "preference_score":     round(pref_w * 100, 2),
                "distance_km":          round(float(features.get("distance_km", 0)), 2),
                "age_diff":             int(features.get("age_diff", 0)),
                "win_rate_diff":        round(float(features.get("win_rate_diff", 0)), 2),
                "role_compatibility":   round(float(features.get("role_compatibility", 0)), 2),
                "has_mentor_relationship": bool(features.get("has_mentor_relation", 0)),
                "complementary_roles":  bool(features.get("complementary_roles", 0)),
                # ── New v3 fields ────────────────────────────────────
                "shared_games":         int(features.get("shared_game_count", 0)),
                "shared_game_jaccard":  round(float(features.get("shared_game_jaccard", 0)), 3),
                "shared_interests":     int(features.get("shared_interest_count", 0)),
                "both_active_30d":      bool(features.get("both_active_30d", 0)),
                "either_verified":      bool(features.get("either_verified", 0)),
                "compat_factor_score":  round(float(features.get("compat_factor_score", 0)), 3),
            })

        recommendations.sort(key=lambda x: x["compatibility_score"], reverse=True)
        return {
            "recommendations":   recommendations[:request.top_k],
            "total_candidates":  len(request.candidate_users),
            "preference_mode":   request.preference_mode,
            "model_version":     "3.0.0",
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_model")
async def reload_model(authorization: str = Header(None),
                       model: UploadFile = File(None)):
    token = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2:
            token = parts[1]
    if token != os.getenv("MODEL_DOWNLOAD_TOKEN"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if model is not None:
        try:
            bio = io.BytesIO(await model.read()); bio.seek(0)
            new = joblib.load(bio)
            global pairwise_model, pairwise_scaler, feature_names
            if isinstance(new, dict):
                pairwise_model  = new.get("model",         pairwise_model)
                pairwise_scaler = new.get("scaler",         pairwise_scaler)
                feature_names   = new.get("feature_names",  feature_names)
            else:
                pairwise_model = new
            load_models()  # reload từ disk nếu file đã được update
            return {"status": "ok", "message": "model reloaded", "features": len(feature_names)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"reload failed: {e}")

    raise HTTPException(status_code=400, detail="No model file provided")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)