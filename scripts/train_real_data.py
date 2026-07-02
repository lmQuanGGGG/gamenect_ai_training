"""
train_real_data.py  —  v3.0
============================
Dùng TẤT CẢ signals từ Firebase để train model:

LABELS (weighted multi-signal):
  Signal 1: swipe_history    → like=1, dislike=0        (335 pairs)
  Signal 2: swipe_latest     → bổ sung / cập nhật mới   (56 pairs)
  Signal 3: matches confirmed → strongest positive (1)   (14 match × 2 chiều)
  Signal 4: matches cancelled → weak positive (1)        (8 match × 2 chiều)
  Signal 5: match quality    → confirmed=weight 2, cancelled=weight 1

FEATURES (62 features từ tất cả profile data):
  Group 1:  Tuổi & preferences    (Homophily Theory - McPherson 2001)
  Group 2:  Khoảng cách GPS       (Proximity Effect - Festinger 1950)
  Group 3:  Giới tính             (User preferences)
  Group 4:  Kỹ năng game         (SBMM - Elo 1978)
  Group 5:  Phong cách chơi      (Self-Categorization - Turner 1987)
  Group 6:  Game & sở thích chung (Jaccard 1912)
  Group 7:  Mục đích tìm kiếm    (Goal alignment)
  Group 8:  Hoạt động & thời gian (Activity matching)
  Group 9:  Profile completeness  (Signaling Theory - Spence 1973)
  Group 10: Popularity & social   (Social proof)
  Group 11: Premium / Verified    (Commitment signals)
  Group 12: Activity recency      (Online status)
  Group 13: Deal breakers & composite

PIPELINE:
  ① Load all signals → merge → weight
  ② Feature engineering (62 features)
  ③ Hybrid dataset: real (weighted) + synthetic augment → ~8000 pairs
  ④ Compare 4 models với 5-Fold Stratified CV
  ⑤ Hyperparameter tuning (RandomizedSearchCV)
  ⑥ Final evaluation + SHAP + all plots
  ⑦ Save model + metadata
"""

import os, sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from datetime import datetime
from collections import Counter

import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     RandomizedSearchCV, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, ConfusionMatrixDisplay,
                              RocCurveDisplay, PrecisionRecallDisplay,
                              accuracy_score, f1_score, precision_score, recall_score)
from geopy.distance import geodesic

warnings.filterwarnings('ignore')

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / 'data' / 'raw'
MODEL_DIR  = BASE_DIR / 'models'
REPORT_DIR = BASE_DIR / 'reports'
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════

def load_all_data():
    print("=" * 65)
    print("📦 LOADING ALL DATA FROM FIREBASE COLLECTIONS")
    print("=" * 65)

    with open(DATA_DIR / 'users.json', encoding='utf-8') as f:
        users = json.load(f)
    print(f"  users           : {len(users):>5} records | {len(users[0]) if users else 0} fields/user")

    with open(DATA_DIR / 'swipe_history.json', encoding='utf-8') as f:
        swipe_history = json.load(f)
    print(f"  swipe_history   : {len(swipe_history):>5} records")

    swipe_latest = []
    if (DATA_DIR / 'swipe_latest.json').exists():
        with open(DATA_DIR / 'swipe_latest.json', encoding='utf-8') as f:
            swipe_latest = json.load(f)
        print(f"  swipe_latest    : {len(swipe_latest):>5} records")

    with open(DATA_DIR / 'matches.json', encoding='utf-8') as f:
        matches = json.load(f)
    conf = sum(1 for m in matches if m.get('status') == 'confirmed')
    canc = sum(1 for m in matches if m.get('status') == 'cancelled')
    print(f"  matches         : {len(matches):>5} records (confirmed={conf}, cancelled={canc})")

    return users, swipe_history, swipe_latest, matches


def build_user_dict(users):
    d = {}
    for u in users:
        uid = u.get('uid') or u.get('__id', '')
        if uid:
            d[uid] = u
    return d


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: MERGE ALL INTERACTION SIGNALS
# ══════════════════════════════════════════════════════════════════════

def merge_all_signals(swipe_history, swipe_latest, matches):
    """
    Merge TẤT CẢ signals tương tác với weight khác nhau:

    Thứ tự ưu tiên (signal mạnh hơn ghi đè / cộng thêm):
      - swipe like      → label=1, weight=1.0
      - swipe dislike   → label=0, weight=1.0
      - match confirmed → label=1, weight=2.0 (mutual → mạnh nhất)
      - match cancelled → label=1, weight=0.8 (họ vẫn matched rồi mới cancel)

    Sample weights được dùng để train_test_split và trong model fit.
    Cơ sở: Weighted learning (Freund & Schapire 1997 - AdaBoost paper)
    """
    labels   = {}  # (u1, u2) → 0/1
    weights  = {}  # (u1, u2) → float weight

    # ── Signal 1+2: Swipe History + Latest ──────────────────────────
    for s in swipe_history + swipe_latest:
        key    = (s['userId'], s['targetUserId'])
        action = s['action']
        lbl    = 1 if action == 'like' else 0
        # Swipe history = base signal
        if key not in labels:
            labels[key]  = lbl
            weights[key] = 1.0
        elif lbl == 1:  # like ghi đè dislike (cập nhật tích cực)
            labels[key]  = 1
            weights[key] = max(weights[key], 1.0)

    # ── Signal 3+4: Matches (Confirmed & Cancelled) ──────────────────
    for m in matches:
        user_ids = m.get('userIds', [])
        if len(user_ids) != 2:
            continue
        u1, u2 = user_ids[0], user_ids[1]
        status = m.get('status', '')
        w = 2.0 if status == 'confirmed' else 0.8  # Confirmed = stronger

        for key in [(u1, u2), (u2, u1)]:  # Cả 2 chiều
            labels[key]  = 1
            weights[key] = max(weights.get(key, 0), w)

    # Stats
    cnts = Counter(labels.values())
    print(f"\n  Merged signals: {len(labels)} unique pairs")
    print(f"    Like  (1): {cnts[1]} pairs")
    print(f"    Dislike(0): {cnts[0]} pairs")
    w_vals = list(weights.values())
    print(f"    Weight dist: min={min(w_vals):.1f} max={max(w_vals):.1f} mean={np.mean(w_vals):.2f}")

    return labels, weights


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: FEATURE ENGINEERING — 62 FEATURES
# ══════════════════════════════════════════════════════════════════════

def calc_distance(u1, u2):
    def get_loc(u):
        loc = u.get('location', {})
        if isinstance(loc, dict):
            return loc.get('latitude'), loc.get('longitude')
        return None, None
    lat1, lon1 = get_loc(u1)
    lat2, lon2 = get_loc(u2)
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return 500.0
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except Exception:
        return 500.0


def days_since(date_str):
    """Tính số ngày kể từ lastSeen"""
    if not date_str:
        return 999
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        return max((now - dt).days, 0)
    except Exception:
        return 999


def create_features(u1, u2):
    """
    Tạo 62 features cho cặp (u1, u2).
    Dùng TẤT CẢ thông tin có trong user profile từ Firebase.
    """
    f = {}

    # ── GROUP 1: TUỔI (Homophily Theory - McPherson 2001) ──────────
    age1 = float(u1.get('age', 25))
    age2 = float(u2.get('age', 25))
    ad   = abs(age1 - age2)
    f['age_diff']          = ad
    f['age_compatible_5']  = 1 if ad <= 5  else 0
    f['age_compatible_8']  = 1 if ad <= 8  else 0
    f['age_compatible_12'] = 1 if ad <= 12 else 0
    mn1 = float(u1.get('minAge', 18)); mx1 = float(u1.get('maxAge', 60))
    mn2 = float(u2.get('minAge', 18)); mx2 = float(u2.get('maxAge', 60))
    f['age_pref_match']   = 1 if (mn1 <= age2 <= mx1 and mn2 <= age1 <= mx2) else 0
    f['age_pref_one_way'] = 1 if (mn1 <= age2 <= mx1 or  mn2 <= age1 <= mx2) else 0

    # ── GROUP 2: KHOẢNG CÁCH (Proximity Effect - Festinger 1950) ───
    dist = calc_distance(u1, u2)
    f['distance_km']       = min(dist, 1000)
    f['dist_within_5km']   = 1 if dist <= 5   else 0
    f['dist_within_20km']  = 1 if dist <= 20  else 0
    f['dist_within_50km']  = 1 if dist <= 50  else 0
    f['dist_within_100km'] = 1 if dist <= 100 else 0
    md1 = float(u1.get('maxDistance', 50))
    md2 = float(u2.get('maxDistance', 50))
    f['dist_pref_match']   = 1 if (dist <= md1 and dist <= md2) else 0
    f['dist_pref_one_way'] = 1 if (dist <= md1 or  dist <= md2) else 0

    # ── GROUP 3: GIỚI TÍNH ─────────────────────────────────────────
    g1 = u1.get('gender', 'Unknown')
    g2 = u2.get('gender', 'Unknown')
    p1 = u1.get('interestedInGender', 'Tất cả') or 'Tất cả'
    p2 = u2.get('interestedInGender', 'Tất cả') or 'Tất cả'
    m12 = (p1 == 'Tất cả' or p1 == g2)
    m21 = (p2 == 'Tất cả' or p2 == g1)
    f['gender_match']      = 1 if (m12 and m21) else 0
    f['gender_one_way']    = 1 if (m12 or m21)  else 0
    f['same_gender']       = 1 if g1 == g2       else 0

    # ── GROUP 4: KỸ NĂNG GAME (SBMM - Elo 1978) ───────────────────
    wr1 = float(u1.get('winRate', 50))
    wr2 = float(u2.get('winRate', 50))
    wrd = abs(wr1 - wr2)
    f['win_rate_diff']     = wrd
    f['skill_similar_15']  = 1 if wrd <= 15 else 0
    f['skill_similar_25']  = 1 if wrd <= 25 else 0
    f['same_rank']         = 1 if u1.get('rank', '') == u2.get('rank', '') else 0
    sg  = wr1 - wr2
    pt1 = float(u1.get('playTime', 0))
    pt2 = float(u2.get('playTime', 0))
    xg  = pt1 - pt2
    f['can_mentor_12']       = 1 if (sg > 15  and xg > 500)  else 0
    f['can_mentor_21']       = 1 if (sg < -15 and xg < -500) else 0
    f['has_mentor_relation'] = 1 if (f['can_mentor_12'] or f['can_mentor_21']) else 0

    # ── GROUP 5: PHONG CÁCH CHƠI (Self-Categorization - Turner 1987)
    s1 = u1.get('gameStyle', 'Unknown')
    s2 = u2.get('gameStyle', 'Unknown')
    f['same_game_style'] = 1 if s1 == s2 else 0
    cm = {
        'Casual':            {'Casual', 'Vừa chơi vừa học'},
        'Competitive':       {'Competitive', 'Pro Player'},
        'Pro Player':        {'Pro Player', 'Competitive'},
        'Streamer':          {'Streamer', 'Casual', 'Vừa chơi vừa học'},
        'Vừa chơi vừa học': {'Casual', 'Vừa chơi vừa học', 'Streamer'},
    }
    f['compat_style'] = 1 if (s1 in cm and s2 in cm.get(s1, set())) else 0
    rc = 0.0
    def sty(a, b): return (s1==a and s2==b) or (s1==b and s2==a)
    if sty('Pro Player', 'Competitive'):          rc = 0.9
    if sty('Vừa chơi vừa học', 'Competitive'):   rc = max(rc, 0.85)
    if sty('Streamer', 'Casual'):                 rc = max(rc, 0.8)
    if sty('Competitive', 'Casual'):              rc = max(rc, 0.7)
    if s1 == s2:                                  rc = max(rc, 0.6)
    f['role_compatibility']  = rc
    f['complementary_roles'] = 1 if rc >= 0.75 else 0

    # ── GROUP 6: GAME & SỞ THÍCH CHUNG (Jaccard 1912) ─────────────
    gs1 = set(u1.get('favoriteGames', []) or [])
    gs2 = set(u2.get('favoriteGames', []) or [])
    ig  = gs1 & gs2; ug = gs1 | gs2
    f['shared_game_count']   = len(ig)
    f['shared_game_jaccard'] = len(ig) / max(len(ug), 1)
    f['has_common_game']     = 1 if ig else 0

    i1 = set(u1.get('interests', []) or [])
    i2 = set(u2.get('interests', []) or [])
    ii = i1 & i2; ui = i1 | i2
    f['shared_interest_count']   = len(ii)
    f['shared_interest_jaccard'] = len(ii) / max(len(ui), 1)
    f['has_common_interest']     = 1 if ii else 0

    # ── GROUP 7: MỤC ĐÍCH TÌM KIẾM ────────────────────────────────
    f['same_looking_for'] = 1 if u1.get('lookingFor', '') == u2.get('lookingFor', '') else 0

    # ── GROUP 8: HOẠT ĐỘNG ─────────────────────────────────────────
    f['play_time_ratio']   = min(pt1, pt2) / max(pt1, pt2, 1)
    f['both_active_1000h'] = 1 if (pt1 > 1000 and pt2 > 1000) else 0
    f['activity_gap_log']  = abs(np.log1p(pt1) - np.log1p(pt2))

    # ── GROUP 9: PROFILE COMPLETENESS (Signaling - Spence 1973) ────
    def completeness(u):
        sc = 0.0
        if u.get('bio'):                                      sc += 0.20
        if len(u.get('additionalPhotos', []) or []) > 0:     sc += 0.20
        if len(u.get('favoriteGames',    []) or []) > 0:     sc += 0.15
        if len(u.get('interests',        []) or []) > 0:     sc += 0.15
        if u.get('isVerified'):                               sc += 0.15
        if isinstance(u.get('location'), dict):               sc += 0.15
        return round(sc, 2)
    c1 = completeness(u1); c2 = completeness(u2)
    f['avg_completeness']  = (c1 + c2) / 2
    f['completeness_diff'] = abs(c1 - c2)

    # ── GROUP 10: POPULARITY & SOCIAL PROOF ────────────────────────
    # Normalize bằng log để giảm ảnh hưởng outliers
    lc1 = np.log1p(float(u1.get('likeCount',    0)))
    lc2 = np.log1p(float(u2.get('likeCount',    0)))
    mc1 = np.log1p(float(u1.get('matchCount',   0)))
    mc2 = np.log1p(float(u2.get('matchCount',   0)))
    pv1 = np.log1p(float(u1.get('profileViews', 0)))
    pv2 = np.log1p(float(u2.get('profileViews', 0)))
    fc1 = np.log1p(float(u1.get('friendCount',  0)))
    fc2 = np.log1p(float(u2.get('friendCount',  0)))

    f['avg_like_count']      = (lc1 + lc2) / 2
    f['avg_match_count']     = (mc1 + mc2) / 2
    f['avg_profile_views']   = (pv1 + pv2) / 2
    f['avg_friend_count']    = (fc1 + fc2) / 2
    f['popularity_gap']      = abs(lc1 - lc2)
    f['match_rate_u1']       = float(u1.get('matchCount', 0)) / max(float(u1.get('likeCount', 1)), 1)
    f['match_rate_u2']       = float(u2.get('matchCount', 0)) / max(float(u2.get('likeCount', 1)), 1)
    f['avg_match_rate']      = (f['match_rate_u1'] + f['match_rate_u2']) / 2

    # Super likes = high-intent signal
    sl1 = float(u1.get('superLikeCount', 0))
    sl2 = float(u2.get('superLikeCount', 0))
    f['avg_super_likes']     = (sl1 + sl2) / 2

    # ── GROUP 11: PREMIUM / VERIFIED (Commitment signal) ───────────
    f['both_verified']   = 1 if (u1.get('isVerified') and u2.get('isVerified'))   else 0
    f['either_verified'] = 1 if (u1.get('isVerified') or  u2.get('isVerified'))   else 0
    f['both_premium']    = 1 if (u1.get('isPremium')  and u2.get('isPremium'))     else 0
    f['either_premium']  = 1 if (u1.get('isPremium')  or  u2.get('isPremium'))     else 0
    # showOnlineStatus = user muốn hiện trạng thái → tích cực hơn
    f['both_show_online'] = 1 if (u1.get('showOnlineStatus') and u2.get('showOnlineStatus')) else 0

    # ── GROUP 12: ACTIVITY RECENCY (Online & LastSeen) ─────────────
    d1 = days_since(u1.get('lastSeen', ''))
    d2 = days_since(u2.get('lastSeen', ''))
    f['days_since_seen_u1']  = min(d1, 365)
    f['days_since_seen_u2']  = min(d2, 365)
    f['both_active_7d']      = 1 if (d1 <= 7  and d2 <= 7)  else 0
    f['both_active_30d']     = 1 if (d1 <= 30 and d2 <= 30) else 0
    f['either_online']       = 1 if (u1.get('isOnline') or u2.get('isOnline')) else 0
    f['both_online']         = 1 if (u1.get('isOnline') and u2.get('isOnline')) else 0

    # ── GROUP 13: DEAL BREAKERS & COMPOSITE ────────────────────────
    db = sum([
        f['gender_match'] == 0,
        f['dist_pref_match'] == 0,
        f['age_pref_match'] == 0,
    ])
    f['deal_breaker_count'] = db
    f['has_deal_breakers']  = 1 if db > 0 else 0

    cf_list = [
        f['gender_match'], f['age_pref_match'], f['dist_pref_match'],
        f['compat_style'], f['skill_similar_25'], f['dist_within_50km'],
        f['age_compatible_8'], f['has_common_game'], f['same_looking_for'],
        f['either_verified'], f['both_active_30d'],
    ]
    f['compat_factor_score'] = sum(cf_list) / len(cf_list)

    return f


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: BUILD HYBRID DATASET (~8000 pairs)
# ══════════════════════════════════════════════════════════════════════

def build_dataset(user_dict, labels, weights, target_total=8000):
    """
    Phase 1: Real labeled pairs từ TẤT CẢ signals (weighted)
    Phase 2: Synthetic augmentation → tổng ~target_total pairs
    """
    pairs, y_labels, y_weights = [], [], []
    real_keys = set()

    # ── PHASE 1: REAL LABELED PAIRS ────────────────────────────────
    real_count = 0
    for (uid1, uid2), lbl in labels.items():
        u1 = user_dict.get(uid1); u2 = user_dict.get(uid2)
        if u1 is None or u2 is None:
            continue
        feats = create_features(u1, u2)
        pairs.append(feats)
        y_labels.append(lbl)
        y_weights.append(weights.get((uid1, uid2), 1.0))
        real_count += 1
        real_keys.add((uid1, uid2))
        real_keys.add((uid2, uid1))

    like_r    = sum(y_labels)
    dislike_r = len(y_labels) - like_r
    print(f"\n  Phase 1 — Real pairs: {real_count}")
    print(f"    Like (1)   : {like_r}  (avg weight={np.mean([y_weights[i] for i in range(len(y_labels)) if y_labels[i]==1]):.2f})")
    print(f"    Dislike (0): {dislike_r}  (avg weight={np.mean([y_weights[i] for i in range(len(y_labels)) if y_labels[i]==0]):.2f})")

    # ── PHASE 2: SYNTHETIC AUGMENTATION ────────────────────────────
    user_list  = list(user_dict.values())
    aug_needed = max(0, target_total - real_count)
    aug_each   = aug_needed // 2
    aug_c = 0; aug_i = 0

    print(f"\n  Phase 2 — Target {target_total} total ({aug_needed} synthetic)")

    rng = np.random.RandomState(42)
    idx = list(range(len(user_list))); rng.shuffle(idx)

    done = False
    for i in range(len(idx)):
        if done: break
        for j in range(i + 1, len(idx)):
            if aug_c >= aug_each and aug_i >= aug_each:
                done = True; break
            u1 = user_list[idx[i]]; u2 = user_list[idx[j]]
            uid1 = u1.get('uid', u1.get('__id', ''))
            uid2 = u2.get('uid', u2.get('__id', ''))
            if (uid1, uid2) in real_keys or uid1 == uid2:
                continue

            feats = create_features(u1, u2)

            clearly_compat = (
                feats['deal_breaker_count'] == 0 and
                feats['compat_factor_score'] >= 0.75 and
                feats['has_common_game'] == 1 and
                feats['distance_km'] <= 50
            )
            clearly_incompat = (
                feats['deal_breaker_count'] >= 2 or
                feats['distance_km'] > 300 or
                feats['age_diff'] > 20 or
                (feats['win_rate_diff'] > 45 and feats['has_mentor_relation'] == 0)
            )

            if clearly_compat and aug_c < aug_each:
                pairs.append(feats); y_labels.append(1); y_weights.append(0.5); aug_c += 1
            elif clearly_incompat and aug_i < aug_each:
                pairs.append(feats); y_labels.append(0); y_weights.append(0.5); aug_i += 1

    total_aug = aug_c + aug_i
    print(f"    Compatible  : {aug_c} | Incompatible: {aug_i}")
    print(f"\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │ Total pairs    : {len(pairs):>6}                        │")
    print(f"  │ Real (weighted): {real_count:>6} ({real_count/len(pairs)*100:.1f}%)              │")
    print(f"  │ Synthetic      : {total_aug:>6} ({total_aug/len(pairs)*100:.1f}%) weight=0.5     │")
    print(f"  │ Label 1 (like) : {sum(y_labels):>6}                        │")
    print(f"  │ Label 0 (dis)  : {len(y_labels)-sum(y_labels):>6}                        │")
    print(f"  └──────────────────────────────────────────────────┘")

    return (pd.DataFrame(pairs).fillna(0),
            np.array(y_labels),
            np.array(y_weights))


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════

def compare_models(X, y, sample_weight):
    print("\n" + "=" * 65)
    print("🏆 MODEL COMPARISON (5-Fold Stratified CV)")
    print("=" * 65)

    # Note: CV không support sample_weight trực tiếp → dùng unweighted CV
    # nhưng dùng sample_weight khi fit final model
    models_map = {
        "Baseline (Majority)":   Pipeline([('sc', StandardScaler()), ('clf', DummyClassifier(strategy='most_frequent'))]),
        "Logistic Regression":   Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))]),
        "Random Forest":         Pipeline([('sc', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42))]),
        "Gradient Boosting":     Pipeline([('sc', StandardScaler()), ('clf', GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42))]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
    results = {}

    for name, pipe in models_map.items():
        print(f"\n  {name} ...", end='', flush=True)
        cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                                return_train_score=True, n_jobs=-1)
        results[name] = {m: (cv_res[f'test_{m}'].mean(), cv_res[f'test_{m}'].std())
                         for m in scoring}
        results[name]['_train_auc'] = cv_res['train_roc_auc'].mean()
        auc = results[name]['roc_auc']
        f1  = results[name]['f1']
        over = results[name]['_train_auc'] - auc[0]
        print(f" AUC={auc[0]:.4f}±{auc[1]:.4f} | F1={f1[0]:.4f} | Overfit gap={over:.3f}")

    print("\n" + "─" * 65)
    print(f"{'Model':<28} {'AUC':>10} {'F1':>10} {'Acc':>10}")
    print("─" * 65)
    for name, res in results.items():
        print(f"{name:<28} {res['roc_auc'][0]:.4f}±{res['roc_auc'][1]:.3f}"
              f" {res['f1'][0]:.4f}±{res['f1'][1]:.3f}"
              f" {res['accuracy'][0]:.4f}±{res['accuracy'][1]:.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════

def tune_model(X, y):
    print("\n" + "=" * 65)
    print("🔧 HYPERPARAMETER TUNING (RandomizedSearchCV, 40 combos)")
    print("=" * 65)

    param_dist = {
        'clf__n_estimators':      [100, 200, 300, 500],
        'clf__max_depth':         [3, 4, 5, 6, 7],
        'clf__learning_rate':     [0.01, 0.05, 0.1, 0.2],
        'clf__min_samples_split': [5, 10, 20],
        'clf__min_samples_leaf':  [3, 5, 10],
        'clf__subsample':         [0.7, 0.8, 0.9, 1.0],
    }
    pipe   = Pipeline([('sc', StandardScaler()),
                       ('clf', GradientBoostingClassifier(random_state=42))])
    search = RandomizedSearchCV(pipe, param_dist, n_iter=40,
                                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring='roc_auc', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X, y)
    print(f"\n  Best AUC (CV): {search.best_score_:.4f}")
    print(f"  Best params  : {search.best_params_}")
    return search.best_estimator_, search.best_params_


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: FINAL MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════

def train_and_evaluate(X_df, y, sample_weight, best_params):
    from sklearn.model_selection import train_test_split
    print("\n" + "=" * 65)
    print("📊 FINAL MODEL EVALUATION (with sample weights)")
    print("=" * 65)

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X_df.values, y, sample_weight,
        test_size=0.2, random_state=42, stratify=y
    )

    gbc_params = {k.replace('clf__', ''): v for k, v in best_params.items()}
    gbc_params['random_state'] = 42

    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_tr)
    X_te_sc    = scaler.transform(X_te)

    model = GradientBoostingClassifier(**gbc_params)
    model.fit(X_tr_sc, y_tr, sample_weight=w_tr)

    y_pred = model.predict(X_te_sc)
    y_prob = model.predict_proba(X_te_sc)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)

    print(f"\n  Train set: {len(X_tr)} samples")
    print(f"  Test set : {len(X_te)} samples")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Accuracy : {accuracy_score(y_te, y_pred):.4f}")
    print(f"  F1       : {f1_score(y_te, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_te, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_te, y_pred):.4f}")
    print(classification_report(y_te, y_pred, target_names=['Incompatible', 'Compatible']))

    return model, scaler, X_tr, y_tr, X_te_sc, y_te, y_pred, y_prob


# ══════════════════════════════════════════════════════════════════════
# SECTION 8: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════

def generate_plots(model, feature_names, X_tr, y_tr, X_te_sc, y_te, y_pred, y_prob):
    print("\n" + "=" * 65)
    print("📈 GENERATING ALL VISUALIZATIONS")
    print("=" * 65)

    plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})
    sns.set_style("whitegrid")

    # Dashboard 2×2
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('GameNect AI — Model Evaluation Dashboard (v3.0)', fontsize=14, fontweight='bold')

    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=['Incompatible','Compatible'], yticklabels=['Incompatible','Compatible'])
    axes[0,0].set(title='Confusion Matrix', ylabel='Actual', xlabel='Predicted')

    RocCurveDisplay.from_predictions(y_te, y_prob, ax=axes[0,1],
                                     name=f'GBC (AUC={roc_auc_score(y_te,y_prob):.3f})')
    axes[0,1].plot([0,1],[0,1],'k--',alpha=0.4,label='Random'); axes[0,1].set_title('ROC Curve'); axes[0,1].legend()

    PrecisionRecallDisplay.from_predictions(y_te, y_prob, ax=axes[1,0], name='GBC')
    axes[1,0].set_title('Precision-Recall Curve')

    imp_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})\
               .sort_values('importance').tail(20)
    axes[1,1].barh(imp_df['feature'], imp_df['importance'], color='steelblue')
    axes[1,1].set(title='Top 20 Feature Importances', xlabel='Importance')

    plt.tight_layout()
    p1 = REPORT_DIR / 'model_evaluation.png'
    plt.savefig(p1, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p1}")

    # Learning curve
    from sklearn.pipeline import Pipeline as _Pipe
    pipe_lc = _Pipe([('sc', StandardScaler()), ('clf', model)])
    sizes   = np.linspace(0.15, 1.0, 8)
    # pyrefly: ignore [bad-unpacking]
    tsz, tr_sc, va_sc = learning_curve(pipe_lc, X_tr, y_tr, train_sizes=sizes,
                                        cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                        scoring='roc_auc', n_jobs=-1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(tsz, tr_sc.mean(1), 'o-', color='royalblue', label='Train AUC')
    ax.fill_between(tsz, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.15, color='royalblue')
    ax.plot(tsz, va_sc.mean(1), 'o-', color='coral', label='Validation AUC')
    ax.fill_between(tsz, va_sc.mean(1)-va_sc.std(1), va_sc.mean(1)+va_sc.std(1), alpha=0.15, color='coral')
    ax.set(xlabel='Training Samples', ylabel='ROC-AUC', title='Learning Curve — GameNect AI v3.0'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = REPORT_DIR / 'learning_curve.png'; plt.savefig(p2, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p2}")

    # SHAP
    try:
        import shap
        print("  Generating SHAP analysis...")
        explainer = shap.TreeExplainer(model)
        idx_s     = np.random.RandomState(42).choice(len(X_te_sc), min(200, len(X_te_sc)), replace=False)
        shap_vals = explainer.shap_values(X_te_sc[idx_s])

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_te_sc[idx_s], feature_names=feature_names,
                          plot_type="bar", show=False, max_display=20)
        plt.title('SHAP Feature Importance (Top 20)', fontsize=13, fontweight='bold')
        plt.tight_layout(); p3 = REPORT_DIR / 'shap_importance.png'
        plt.savefig(p3, bbox_inches='tight'); plt.close(); print(f"  Saved: {p3}")

        shap.summary_plot(shap_vals, X_te_sc[idx_s], feature_names=feature_names, show=False, max_display=20)
        p4 = REPORT_DIR / 'shap_beeswarm.png'; plt.savefig(p4, bbox_inches='tight'); plt.close()
        print(f"  Saved: {p4}")
    except ImportError:
        print("  [shap not found — pip install shap]")

    # Score distribution
    fig, ax = plt.subplots(figsize=(9,5))
    ax.hist(y_prob[y_te==0], bins=25, alpha=0.6, color='coral',     label='Dislike (0)')
    ax.hist(y_prob[y_te==1], bins=25, alpha=0.6, color='royalblue', label='Like (1)')
    ax.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
    ax.set(xlabel='Predicted Score', ylabel='Count', title='Score Distribution by Actual Label'); ax.legend()
    plt.tight_layout(); p5 = REPORT_DIR / 'score_distribution.png'
    plt.savefig(p5, bbox_inches='tight'); plt.close(); print(f"  Saved: {p5}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 9: SAVE MODEL + FULL METADATA
# ══════════════════════════════════════════════════════════════════════

def save_model(model, scaler, feature_names, best_params, y, dataset_info, auc_score=None):
    joblib.dump(model,         MODEL_DIR / 'pairwise_compatibility_model.pkl')
    joblib.dump(scaler,        MODEL_DIR / 'pairwise_scaler.pkl')
    joblib.dump(feature_names, MODEL_DIR / 'pairwise_feature_names.pkl')

    metadata = {
        "version":            f"3.0.{os.getenv('GITHUB_RUN_NUMBER', '0')}",
        "trained_at":         datetime.now().isoformat(),
        "data_source":        "Firebase Firestore — All collections",
        "training_strategy":  "Hybrid weighted: all signals (swipe+match+profile) + clear synthetic",
        "model_type":         "GradientBoostingClassifier",
        "best_params":        best_params,
        "feature_count":      len(feature_names),
        "feature_names":      feature_names,
        "dataset":            dataset_info,
        "signals_used": {
            "swipe_history":        "335 records — like=1/dislike=0, weight=1.0",
            "swipe_latest":         "56 records — cập nhật mới nhất",
            "matches_confirmed":    "14 pairs — weight=2.0 (strongest signal)",
            "matches_cancelled":    "8 pairs  — weight=0.8 (they did match)",
            "user_profile_social":  "likeCount, matchCount, friendCount, profileViews (log-normalized)",
            "user_profile_premium": "isPremium, isVerified, subscriptionTier",
            "user_profile_activity":"lastSeen, isOnline, playTime",
            "user_profile_content": "favoriteGames (Jaccard), interests (Jaccard)"
        },
        "theoretical_basis": {
            "homophily":     "McPherson et al. (2001) — Annual Review of Sociology 27:415-444",
            "proximity":     "Festinger et al. (1950) — Social Pressures in Informal Groups",
            "sbmm":          "Elo (1978) — The Rating of Chessplayers",
            "jaccard":       "Jaccard (1912) — New Phytologist 11(2):37-50",
            "signaling":     "Spence (1973) — QJE 87(3):355-374 [Nobel 2001]",
            "implicit_fb":   "Hu, Koren & Volinsky (2008) — ICDM",
            "sample_weight": "Freund & Schapire (1997) — Weighted learning",
            "cv_method":     "Kohavi (1995) — Cross-validation and Bootstrap",
        }
    }

    with open(MODEL_DIR / 'model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    log_file = MODEL_DIR / 'training_history.log'
    with open(log_file, 'a', encoding='utf-8') as f:
        auc_str = f" | AUC: {auc_score:.4f}" if auc_score else ""
        f.write(f"[{metadata['trained_at']}] Version: {metadata['version']} | Data: {dataset_info['total_pairs']} (Train: {dataset_info.get('train_size', 0)}, Test: {dataset_info.get('test_size', 0)}){auc_str} | Status: PASSED (Auto-Gating)\n")

    print(f"\n  ✅ Model   → {MODEL_DIR / 'pairwise_compatibility_model.pkl'}")
    print(f"  ✅ Scaler  → {MODEL_DIR / 'pairwise_scaler.pkl'}")
    print(f"  ✅ Meta    → {MODEL_DIR / 'model_metadata.json'}")
    print(f"  ✅ Log     → {log_file}")
    print(f"  ✅ Reports → {REPORT_DIR}/")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
# Ghi chú: Kích hoạt chạy lại GitHub Actions

def main():
    import time
    from services.telegram_logger import telegram_logger
    
    start_time = time.time()
    
    print("\n" + "█" * 65)
    print("█  GAMENECT AI — REAL DATA PIPELINE v3.0                   █")
    print("█  Using ALL Firebase signals: swipe + match + profile      █")
    print("█" * 65 + "\n")

    telegram_logger.log_training_start("GAMENECT_AI_V3.0", config={"pipeline": "v3.0", "target_total": 8000, "signals": "swipe+match"})

    # 1. Load
    users, swipe_hist, swipe_latest, matches = load_all_data()
    user_dict = build_user_dict(users)
    print(f"\n  User dict: {len(user_dict)} users")

    # 2. Merge ALL signals
    print("\n" + "=" * 65)
    print("🔗 MERGING ALL INTERACTION SIGNALS")
    print("=" * 65)
    labels, weights = merge_all_signals(swipe_hist, swipe_latest, matches)

    # 3. Build dataset
    print("\n" + "=" * 65)
    print("🔨 BUILDING HYBRID DATASET (all signals + 62 features)")
    print("=" * 65)
    X_df, y, sample_weight = build_dataset(user_dict, labels, weights, target_total=8000)
    X = X_df.values
    feature_names = X_df.columns.tolist()
    print(f"\n  Features : {len(feature_names)}")
    print(f"  Shape    : {X.shape}")
    print(f"  Labels   : {Counter(y.tolist())}")

    # 4. Compare models
    compare_models(X_df, y, sample_weight)

    # 5. Tune
    best_pipe, best_params = tune_model(X_df, y)

    # 6. Final train + eval (with sample_weight)
    model, scaler, X_tr, y_tr, X_te_sc, y_te, y_pred, y_prob = train_and_evaluate(
        X_df, y, sample_weight, best_params
    )

    # 7. Plots
    generate_plots(model, feature_names, X_tr, y_tr, X_te_sc, y_te, y_pred, y_prob)

    # 7.5 Automated Gating (Tiêu chí thay thế model)
    from sklearn.metrics import roc_auc_score, accuracy_score
    auc_score = roc_auc_score(y_te, y_prob)
    acc_score = accuracy_score(y_te, y_pred)
    print("\n" + "=" * 65)
    print(f"🛡️ AUTOMATED GATING CHECK (Threshold AUC >= 0.75)")
    print("=" * 65)
    if auc_score < 0.75:
        print(f"❌ REJECTED: Model AUC ({auc_score:.4f}) is below threshold (0.75)!")
        print("❌ Canceled saving model and deploying. Exiting with error code 1.")
        telegram_logger.log_alert(f"Quá trình Training thất bại! Gating bị từ chối do AUC thấp ({auc_score:.4f} < 0.75)")
        sys.exit(1)
    else:
        print(f"✅ PASSED: Model AUC ({auc_score:.4f}) meets the threshold.")

    # 8. Save
    dataset_info = {
        "total_pairs":       int(len(y)),
        "train_size":        int(len(y_tr)),
        "test_size":         int(len(y_te)),
        "real_pairs":        int(sum(1 for w in sample_weight if w >= 1.0)),
        "synthetic_pairs":   int(sum(1 for w in sample_weight if w < 1.0)),
        "positive_rate":     float(y.mean()),
        "signals_count":     len(labels),
    }
    save_model(model, scaler, feature_names, best_params, y, dataset_info, auc_score)

    time_taken = time.time() - start_time
    telegram_logger.log_training_end(
        version=f"3.0.{os.getenv('GITHUB_RUN_NUMBER', '0')}",
        time_taken=time_taken,
        total_pairs=len(y),
        train_size=len(y_tr),
        test_size=len(y_te),
        auc=auc_score,
        acc=acc_score,
        status="PASSED (Auto-Gating)" if auc_score >= 0.75 else "REJECTED (Low AUC)",
        filepath="models/pairwise_compatibility_model.pkl"
    )

    print("\n" + "█" * 65)
    print("█  TRAINING v3.0 COMPLETED                                  █")
    print("█" * 65)


if __name__ == "__main__":
    main()
