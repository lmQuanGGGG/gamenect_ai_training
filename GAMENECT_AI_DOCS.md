# 🎮 GameNect AI Training — Tài Liệu Kỹ Thuật Đầy Đủ

> **Phiên bản:** 1.0.0 | **Cập nhật:** 2026-04-20  
> **Tác giả:** Gamenect Team  
> **Mục tiêu:** Hệ thống AI dự đoán độ tương thích giữa game thủ — tương tự Tinder nhưng dành cho gaming

---

## 📚 Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Kiến Trúc Hệ Thống](#2-kiến-trúc-hệ-thống)
3. [Dữ Liệu — Lấy Từ Đâu & Cấu Trúc](#3-dữ-liệu--lấy-từ-đâu--cấu-trúc)
4. [Feature Engineering — Chuyển Đổi Sang Vector](#4-feature-engineering--chuyển-đổi-sang-vector)
5. [Hai Model Trong Dự Án](#5-hai-model-trong-dự-án)
6. [Quy Trình Training Chi Tiết](#6-quy-trình-training-chi-tiết)
7. [API Serving](#7-api-serving)
8. [Cách Chạy Dự Án — Từng Bước](#8-cách-chạy-dự-án--từng-bước)
9. [Đánh Giá Model](#9-đánh-giá-model)
10. [Cấu Trúc Thư Mục](#10-cấu-trúc-thư-mục)
11. [Biến Môi Trường](#11-biến-môi-trường)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Tổng Quan Dự Án

GameNect AI là hệ thống **matching AI** cho ứng dụng kết nối game thủ. Nó hoạt động giống Tinder nhưng thay vì khớp vẻ ngoài, nó khớp dựa trên:

- 🎮 **Phong cách chơi game** (Casual, Competitive, Streamer, Pro Player)
- 📍 **Khoảng cách địa lý** thực tế (geodesic)
- 🏆 **Trình độ kỹ năng** (win rate, rank)
- 👨‍👩‍👦 **Sở thích giới tính & độ tuổi**
- ⏰ **Mức độ hoạt động** (thời gian chơi game)
- 🎯 **Mục đích tìm kiếm** (bạn chơi game, hẹn hò, v.v.)

Hệ thống gồm **2 components chính**:
1. **Training Pipeline** — Thu thập data từ Firebase → Train model → Lưu model
2. **FastAPI Server** — Load model đã train → Phục vụ recommendation qua HTTP API

---

## 2. Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                     FIREBASE FIRESTORE                          │
│     collections: users, likes, matches, swipes, swipe_history  │
└────────────────────────┬────────────────────────────────────────┘
                         │ FirestoreCollector
                         ▼
┌───────────────────────────────────────────────┐
│              DATA LAYER  (data/raw/)           │
│  users.json  |  matches.json  |  swipes.json  │
└────────────────────────┬──────────────────────┘
                         │ train_user_model.py
                         ▼
┌───────────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                             │
│                                                                │
│  1. Load & Clean Data (fillna, handle nested location)         │
│  2. Create Pairwise Training Pairs (10,000 pairs, 50/50)       │
│  3. Feature Engineering (37 features per pair)                 │
│  4. StandardScaler normalization                               │
│  5. GradientBoostingClassifier (300 trees, depth=6)            │
│  6. Evaluate (Classification Report + ROC-AUC)                 │
└────────────────────────┬──────────────────────────────────────┘
                         │ joblib.dump()
                         ▼
┌─────────────────────────────────────────┐
│            models/                      │
│  pairwise_compatibility_model.pkl       │  ← GBC model
│  pairwise_scaler.pkl                    │  ← StandardScaler
│  pairwise_feature_names.pkl             │  ← Danh sách 37 features
└────────────────────────┬────────────────┘
                         │ joblib.load() on startup
                         ▼
┌─────────────────────────────────────────────────┐
│              FastAPI Server (api/main.py)        │
│                                                  │
│  GET  /health         → Kiểm tra model loaded   │
│  POST /recommend      → Trả danh sách gợi ý     │
│  POST /reload_model   → Hot-reload model mới     │
└─────────────────────────────────────────────────┘
                         │ HTTP JSON Response
                         ▼
                   Flutter App (GameNect)
```

---

## 3. Dữ Liệu — Lấy Từ Đâu & Cấu Trúc

### 3.1 Nguồn Dữ Liệu

Toàn bộ dữ liệu được lấy từ **Firebase Firestore** của project `gamenect-9bec0` (hoặc configured project).

**Collections được thu thập:**

| Collection | Nội dung | Dùng để |
|------------|----------|---------|
| `users` | Hồ sơ người dùng | **Dữ liệu chính để train** |
| `likes` | Lịch sử lượt thích | Tham khảo (chưa dùng trong v1) |
| `matches` | Các cặp đã match | Tham khảo |
| `swipes` | Lịch sử vuốt | Tham khảo |
| `swipe_history` | Lịch sử chi tiết | Tham khảo |

### 3.2 Schema Một User Document (Firestore)

```json
{
  "__id": "userId_abc123",
  "age": 22,
  "gender": "Nam",
  "height": 175,
  "playTime": 1500,
  "winRate": 63.5,
  "totalMatches": 450,
  "friendCount": 120,
  "profileViews": 340,
  "likeCount": 89,
  "matchCount": 34,
  "gameStyle": "Competitive",
  "rank": "Chiến Binh Phèn",
  "lookingFor": "Đồng đội lâu dài",
  "isPremium": false,
  "isVerified": true,
  "bio": "Tìm đồng đội ranked!",
  "interestedInGender": "Tất cả",
  "minAge": 18,
  "maxAge": 30,
  "maxDistance": 50.0,
  "interests": ["FPS", "MOBA", "Battle Royale"],
  "favoriteGames": ["VALORANT", "Liên Quân"],
  "additionalPhotos": ["url1", "url2"],
  "location": {
    "latitude": 10.7769,
    "longitude": 106.7009
  }
}
```

### 3.3 Cách Thu Thập (FirestoreCollector)

```python
# collectors/firestore_service.py
collector = FirestoreCollector(config)
collector.export_to_json()
# → ghi ra: data/raw/users.json, matches.json, swipes.json, ...
```

**Cơ chế:**
- Kết nối Firebase qua Service Account JSON (credential)
- Stream toàn bộ documents trong từng collection (tối đa `max_samples=20,000`)
- Convert các kiểu dữ liệu đặc thù của Firestore (`DatetimeWithNanoseconds`) sang ISO 8601 string
- Gắn thêm `__id` = Firestore document ID vào mỗi record
- Xuất ra file JSON, encode UTF-8 (hỗ trợ tiếng Việt)

---

## 4. Feature Engineering — Chuyển Đổi Sang Vector

### 4.1 Có Hai Lớp Feature Engineering

Dự án có **hai cách** encode data thành vector, dùng cho hai model khác nhau:

---

#### Lớp 1 — `FeatureEngineer` (pipelines/feature_engineering.py)
> Dùng cho **Deep Learning model** (TensorFlow) — *chưa production*

**Cách hoạt động:** Encode từng user thành vector 17 chiều, sau đó ghép hai user + thêm 5 shared features → vector **39 chiều** cuối cùng.

**Bảng encode từng field:**

| Field | Phương thức | Giá trị |
|-------|------------|---------|
| `rank` | Label encoding | Gà Mờ=0 → Thượng Đế AFK=6 |
| `gender` | Label encoding | Nam=0, Nữ=1, Khác=2 |
| `age` | Numeric raw | float |
| `height` | Numeric raw | float |
| `win_rate` | Normalize ÷ 100 | [0.0–1.0] |
| `play_time` | Normalize ÷ 480 | [0.0–∞] (giờ/8h) |
| `game_style` | Label encoding | Casual=0…Vừa chơi vừa học=4 |
| `looking_for` | Label encoding | 0–4 |
| `max_distance` | Normalize ÷ 200 | [0.0–1.0] |
| `is_verified` | Binary | 0.0 / 1.0 |
| `is_online` | Binary | 0.0 / 1.0 |
| `boost_count` | Normalize ÷ 5 | [0.0–1.0] |
| `super_likes_remaining` | Normalize ÷ 5 | [0.0–1.0] |
| `favorite_games_count` | Normalize ÷ 10 | [0.0–1.0] |
| `interests_count` | Normalize ÷ 10 | [0.0–1.0] |

**Pair features bổ sung (5 features):**

```python
shared_games      = |games(u1) ∩ games(u2)| / 10.0
shared_interests  = |interests(u1) ∩ interests(u2)| / 10.0
distance          = haversine_km / 500.0  (giới hạn ở 1.0)
age_gap           = |age1 - age2| / 50.0
rank_gap          = |rank1 - rank2| / 7.0
```

**Khoảng cách tính bằng Haversine:**
```python
a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
c = 2·arcsin(√a)
km = 6371 · c
```

---

#### Lớp 2 — `create_pairwise_features()` (scripts/train_user_model.py)
> Dùng cho **GradientBoosting model** (Scikit-Learn) — **Model đang production**

**Đây là model thực sự được dùng.** Tạo **37 features theo cặp (pairwise)** — không encode riêng từng user mà so sánh trực tiếp cặp:

| # | Feature | Ý Nghĩa | Loại |
|---|---------|---------|------|
| 1 | `age_diff` | Chênh lệch tuổi tuyệt đối | Numeric |
| 2 | `age_compatible` | age_diff ≤ 5 | Binary |
| 3 | `age_within_10` | age_diff ≤ 10 | Binary |
| 4 | `distance_km` | Khoảng cách geodesic (km, max 1000) | Numeric |
| 5 | `distance_within_10km` | dist ≤ 10 km | Binary |
| 6 | `distance_within_50km` | dist ≤ 50 km | Binary |
| 7 | `distance_within_100km` | dist ≤ 100 km | Binary |
| 8 | `gender_match` | Cả 2 đều match giới tính sở thích | Binary |
| 9 | `one_way_gender_match` | Ít nhất 1 bên match | Binary |
| 10 | `same_game_style` | Cùng phong cách chơi | Binary |
| 11 | `compatible_game_style` | Phong cách tương thích (theo bảng) | Binary |
| 12 | `win_rate_diff` | Chênh lệch tỉ lệ thắng (%) | Numeric |
| 13 | `similar_skill` | win_rate_diff ≤ 20 | Binary |
| 14 | `same_rank` | Cùng rank | Binary |
| 15 | `play_time_ratio` | min/max play_time (0–1) | Numeric |
| 16 | `both_active` | Cả 2 chơi > 1000h | Binary |
| 17 | `both_premium` | Cả 2 là Premium | Binary |
| 18 | `both_verified` | Cả 2 đã Verified | Binary |
| 19 | `avg_profile_completeness` | TB độ hoàn thiện profile (0–1) | Numeric |
| 20 | `interest_count_similarity` | Tương đồng số sở thích | Numeric |
| 21 | `game_count_similarity` | Tương đồng số game yêu thích | Numeric |
| 22 | `same_looking_for` | Cùng mục đích tìm kiếm | Binary |
| 23 | `user1_popularity` | Độ phổ biến user 1 (max 10) | Numeric |
| 24 | `user2_popularity` | Độ phổ biến user 2 (max 10) | Numeric |
| 25 | `popularity_balance` | Cân bằng độ phổ biến (1-diff) | Numeric |
| 26 | `height_diff` | Chênh lệch chiều cao (cm) | Numeric |
| 27 | `height_compatible` | height_diff ≤ 15 cm | Binary |
| 28 | `age_preference_match` | Tuổi cả 2 nằm trong range của nhau | Binary |
| 29 | `one_way_age_match` | Ít nhất 1 bên tuổi trong range | Binary |
| 30 | `distance_preference_match` | Khoảng cách trong maxDistance cả 2 | Binary |
| 31 | `one_way_distance_match` | Ít nhất 1 bên trong maxDistance | Binary |
| 32 | `deal_breaker_count` | Số lượng điều kiện vi phạm (0–3) | Numeric |
| 33 | `has_deal_breakers` | deal_breaker_count > 0 | Binary |
| 34 | `compatibility_factor_score` | Tổng hợp 7 yếu tố chính (0–1) | Numeric |
| 35 | `activity_gap` | \|log(playtime1) - log(playtime2)\| | Numeric |
| 36 | `avg_engagement` | TB (views + likes + matches) / 3 | Numeric |
| 37 | `engagement_gap` | Chênh lệch engagement | Numeric |

**Bảng tương thích phong cách game:**

```
Casual      ↔ Casual, Vừa chơi vừa học
Competitive ↔ Competitive, Pro Player
Pro Player  ↔ Pro Player, Competitive
Streamer    ↔ Streamer, Casual, Vừa chơi vừa học
Vừa chơi   ↔ Casual, Vừa chơi vừa học, Streamer
```

**Deal Breakers (tự động trừ điểm):**
1. `gender_match == 0` — Giới tính không khớp
2. Khoảng cách vượt `maxDistance` của cả 2
3. `age_preference_match == 0` — Tuổi ngoài range

### 4.2 Chuẩn Hóa (Normalization)

Sau khi tạo 37 features, toàn bộ được **chuẩn hóa bằng `StandardScaler`**:

```
X_scaled = (X - μ) / σ
```

Trong đó μ là mean, σ là standard deviation tính từ tập train. Scaler được lưu vào `models/pairwise_scaler.pkl` để dùng lại khi inference.

---

## 5. Hai Model Trong Dự Án

### Model 1 — GradientBoostingClassifier ✅ (Đang Production)

**File:** `scripts/train_user_model.py`  
**Saved:** `models/pairwise_compatibility_model.pkl`

```python
GradientBoostingClassifier(
    n_estimators=300,       # 300 cây quyết định
    max_depth=6,            # Độ sâu tối đa mỗi cây
    learning_rate=0.05,     # Học chậm → chính xác hơn
    min_samples_split=10,   # Regularization: min samples để split
    min_samples_leaf=5,     # Regularization: min samples/leaf
    subsample=0.8,          # Dùng 80% data ngẫu nhiên mỗi cây
    random_state=42
)
```

**Ưu điểm của GBC:**
- Không bị ảnh hưởng bởi feature scale (nhưng vẫn scale để đồng nhất)
- Hiệu quả với dữ liệu tabular nhỏ-vừa
- Dễ giải thích qua `feature_importances_`
- Train nhanh hơn Deep Learning

**Output:** Probability [0..1] → nhân 100 → `compatibility_score` (%)

---

### Model 2 — Deep Learning (TensorFlow/Keras) 🚧 (Chưa Production)

**File:** `pipelines/model_architecture.py`  

Kiến trúc:
```
Input (39 features)
    ↓
Dense(256, relu) → BatchNorm → Dropout(0.2)
    ↓
ResidualBlock(256): Dense→BN→Dropout→Dense→BN + Shortcut → ReLU
    ↓
ResidualBlock(128)
    ↓
ResidualBlock(64)
    ↓
Reshape(8, 8) → MultiHeadAttention(4 heads, key_dim=16) → Flatten
    ↓
Concatenate([ResidualOutput, AttentionOutput])
    ↓
Dense(128, relu) → Dropout(0.2)
Dense(64, relu) → Dropout(0.1)
    ↓
Output: Dense(1, sigmoid) → match_probability [0..1]
```

**Loss:** Binary Cross-Entropy  
**Optimizer:** Adam(lr=1e-3)  
**Metrics:** Accuracy, AUC, Precision, Recall

> ⚠️ Model này chưa được tích hợp vào API. `scripts/evaluate_model.py` đánh giá model TF này nhưng cần file `gamenect_matching_model.h5` chưa có.

---

## 6. Quy Trình Training Chi Tiết

### 6.1 Tạo Cặp Training (Pairwise Sampling)

Vì dữ liệu thực (lượt like/swipe thực) chưa được dùng, model **tự sinh** cặp training từ user profiles:

**Cân bằng 50% Compatible / 50% Incompatible (10,000 pairs):**

```
Compatible pairs (5,000):
  - Chọn user1 ngẫu nhiên
  - Tìm user2 thỏa: |age_diff| ≤ 7 VÀ |win_rate_diff| ≤ 25
  - Tính features → chấp nhận nếu deal_breaker_count ≤ 1
  - Label = 1

Incompatible pairs (5,000):
  - Chọn cả 2 ngẫu nhiên
  - Nếu vi phạm ≥2 deal breakers → Label = 0
  - Nếu compatibility_factor_score ≥ 0.5 → Label = 1 (diversity)
  - Còn lại → 70% label=0, 30% label=1 (ngẫu nhiên)
```

> **Lưu ý quan trọng:** Đây là synthetic labels — không dựa trên hành vi thực. Khi có đủ dữ liệu swipe/like thực, cần thay thế bằng real labels.

### 6.2 Train/Test Split

- **80% Train**, **20% Test**
- **Stratified split** — giữ nguyên tỉ lệ class

### 6.3 Training Flow

```
users.json (data/raw/)
    ↓
DataFrame (fillna, handle nested location)
    ↓
create_training_pairs(df, num_pairs=10000)  → X (37 cols), y (10000 labels)
    ↓
fillna(0), replace(inf, 0)
    ↓
train_test_split(0.2, stratify=y)
    ↓
StandardScaler.fit_transform(X_train)   ← fit chỉ trên train
StandardScaler.transform(X_test)        ← chỉ transform test
    ↓
GradientBoostingClassifier.fit(X_train_scaled, y_train)
    ↓
Predict → Classification Report + Confusion Matrix + ROC-AUC
    ↓
joblib.dump(model/scaler/feature_names → models/)
```

---

## 7. API Serving

### 7.1 Endpoint Chính: `POST /recommend`

**Request body:**
```json
{
  "current_user": { ...UserProfile },
  "candidate_users": [ ...UserProfile[] ],
  "top_k": 10,
  "preference_mode": "balanced"
}
```

**4 chế độ `preference_mode`:**

| Mode | Trọng số | Mô tả |
|------|---------|-------|
| `balanced` | `compatibility_factor_score` | Cân bằng tất cả yếu tố |
| `mentor_mentee` | skill/XP gap | Tìm người dạy/học |
| `same_style` | game style | Cùng phong cách chơi |
| `nearby` | distance | Ưu tiên người gần |

**Công thức tính `final_score`:**
```
base_score = model.predict_proba()[1] × 100
preference_weight = f(preference_mode)
final_score = base_score × 0.7 + preference_weight × 100 × 0.3
```

**Response:**
```json
{
  "recommendations": [
    {
      "user_id": "abc123",
      "compatibility_score": 87.5,
      "base_score": 82.3,
      "preference_score": 100.0,
      "distance_km": 3.2,
      "age_diff": 2,
      "win_rate_diff": 5.5,
      "role_compatibility": 0.9,
      "has_mentor_relationship": false,
      "complementary_roles": true
    }
  ],
  "total_candidates": 150,
  "preference_mode": "balanced"
}
```

### 7.2 Endpoint Khác

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `GET /` | GET | Thông tin API |
| `GET /health` | GET | Kiểm tra model đã load chưa |
| `POST /reload_model` | POST | Hot-reload model mới (cần token) |

---

## 8. Cách Chạy Dự Án — Từng Bước

### Bước 0 — Yêu Cầu Môi Trường

```bash
Python >= 3.9
macOS (tensorflow-macos), hoặc Linux (đổi tensorflow thường)
```

### Bước 1 — Cài Đặt Dependencies

```bash
cd /Users/wang04/Downloads/gamenect_ai_training

# Tạo venv mới (khuyến nghị Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate

# Cài từ pyproject.toml (đầy đủ, có TensorFlow Metal cho Mac)
pip install -e ".[dev]"

# HOẶC cài nhanh từ requirements.txt (không có TF)
pip install -r requirements.txt
```

> **macOS Apple Silicon:** Dùng `tensorflow-macos` + `tensorflow-metal` (đã khai báo trong `pyproject.toml`) để GPU acceleration.

### Bước 2 — Cấu Hình Biến Môi Trường

```bash
cp .env.example .env
# Sau đó chỉnh sửa .env:
```

```env
FIREBASE_CREDENTIALS_PATH=/đường/dẫn/tới/firebase-service-account.json
FIREBASE_PROJECT_ID=gamenect-9bec0
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
TRAINING_OUTPUT_DIR=data/artifacts
MODEL_DOWNLOAD_TOKEN=your-secret-token-here
```

> 🔑 **Lấy Firebase Service Account:**  
> Firebase Console → Project Settings → Service Accounts → Generate new private key → Tải file JSON

### Bước 3 — Thu Thập Dữ Liệu Từ Firebase

```bash
# Cách 1: Script trực tiếp
python scripts/collect_from_firebase.py

# Cách 2: Qua collector module
python -m collectors.firestore_service

# Output: data/raw/users.json, data/raw/matches.json, ...
# Sẽ tự backup file cũ với timestamp: data/raw/users_backup_20260420_210000.json
```

### Bước 4 — Tiền Xử Lý Dữ Liệu (Optional)

```bash
python scripts/preprocess_users.py

# Output: data/processed/users_features.csv
# Hiển thị thống kê cơ bản về dữ liệu
```

### Bước 5 — Train Model

```bash
python scripts/train_user_model.py

# Sẽ:
# 1. Load data/raw/users.json
# 2. Tạo 10,000 cặp training
# 3. Train GradientBoostingClassifier
# 4. In Classification Report + ROC-AUC
# 5. Lưu vào models/
```

**Ví dụ output khi train:**
```
Loading users data...
Loaded 450 users
Creating 10000 training pairs...
Created 5000 compatible pairs
Created 5000 incompatible pairs

Label distribution BEFORE split:
Compatible (1): 6234 (62.3%)
Incompatible (0): 3766 (37.7%)

Training pairwise compatibility model...

Classification Report:
              precision    recall  f1-score   support
           0       0.82      0.79      0.80       753
           1       0.86      0.88      0.87      1247
    accuracy                           0.84      2000

ROC-AUC Score: 0.9123

Top 15 Important Features:
                    feature  importance
0      compatibility_factor_score   0.1823
1                 gender_match   0.1245
2          age_preference_match   0.0934
...

Model saved to models/
```

### Bước 6 — Chạy Pipeline Đầy Đủ (Tất Cả Bước)

```bash
bash scripts/run_training_pipeline.sh

# Chạy tuần tự:
# Step 1/3: preprocess_users.py
# Step 2/3: train_user_model.py
# Step 3/3: test_user_model.py
```

### Bước 7 — Test Model

```bash
# Test pairwise model (model production)
python scripts/test_pairwise_model.py

# Output: 5 cặp ngẫu nhiên với điểm tương thích
# 🔥 Compatibility Score: 87.3%  ← >= 70%
# 👍 Compatibility Score: 55.2%  ← 50-70%
# 👎 Compatibility Score: 31.1%  ← < 50%
```

### Bước 8 — Chạy API Server

```bash
# Development
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Heroku-style (Procfile)
# web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**Kiểm tra API hoạt động:**
```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true,"features_count":37}

# Xem Swagger docs
open http://localhost:8000/docs
```

**Gọi API recommend:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "current_user": {
      "user_id": "user123",
      "age": 22,
      "gender": "Nam",
      "height": 175.0,
      "play_time": 1500,
      "win_rate": 65.0,
      "game_style": "Competitive",
      "rank": "Chiến Binh Phèn",
      "looking_for": "Đồng đội lâu dài",
      "latitude": 10.7769,
      "longitude": 106.7009,
      "interestedInGender": "Tất cả",
      "minAge": 18,
      "maxAge": 30,
      "maxDistance": 50.0,
      "num_interests": 4,
      "num_games": 3,
      "num_photos": 5
    },
    "candidate_users": [
      { "user_id": "candidate1", "age": 24, "gender": "Nam", ... }
    ],
    "top_k": 10,
    "preference_mode": "balanced"
  }'
```

### Bước 9 — Hot Reload Model (Sau Khi Train Lại)

```bash
# Upload model mới mà không cần restart server
curl -X POST http://localhost:8000/reload_model \
  -H "Authorization: Bearer YOUR_MODEL_DOWNLOAD_TOKEN" \
  -F "model=@models/pairwise_compatibility_model.pkl"
```

---

## 9. Đánh Giá Model

### 9.1 Các Metric Đánh Giá Tự Động Khi Train

```bash
python scripts/train_user_model.py
```

Trong quá trình training, các metric sau được tự động tính:

| Metric | Mô tả | Ngưỡng tốt |
|--------|--------|-----------|
| **Accuracy** | Tỉ lệ dự đoán đúng | > 80% |
| **Precision** | Trong số predicted compatible, bao nhiêu thực sự compatible | > 80% |
| **Recall** | Trong số thực sự compatible, model tìm được bao nhiêu | > 75% |
| **F1-Score** | TB điều hòa Precision & Recall | > 80% |
| **ROC-AUC** | Khả năng phân biệt 2 class | > 0.85 |

**Confusion Matrix:**
```
                Predicted Incompatible  Predicted Compatible
Actual Incompatible       TN                    FP
Actual Compatible         FN                    TP
```

> ⚠️ **Nếu ROC-AUC > 0.95 → WARNING:** Model có thể đang **overfitting** do labels được tạo synthetic từ chính các rules, không phải từ hành vi người dùng thực.

### 9.2 Đánh Giá Bổ Sung — Feature Importance

```bash
python scripts/train_user_model.py
# Xem "Top 15 Important Features" cuối output
```

Features quan trọng nhất thường là:
1. `compatibility_factor_score` — Tổng hợp 7 yếu tố
2. `gender_match` — Giới tính khớp
3. `age_preference_match` — Độ tuổi trong range
4. `distance_preference_match` — Khoảng cách phù hợp
5. `deal_breaker_count` — Số deal breakers

### 9.3 Đánh Giá Model TF (Legacy)

```bash
# Chỉ chạy được nếu đã train TF model và có test_samples.json
python scripts/evaluate_model.py

# Output: logs/training_history/evaluation_report.json
```

Metrics gồm: accuracy, precision, recall, f1_score, auc, confusion_matrix.

### 9.4 Đánh Giá Thủ Công — Test Cặp Cụ Thể

```python
# Tạo script test thủ công
from scripts.test_pairwise_model import load_model, predict_compatibility

model, scaler, feature_names = load_model('models/')

user_gamer = {
    'user_id': 'a1',
    'age': 22, 'gender': 'Nam', 'height': 175,
    'play_time': 2000, 'win_rate': 70.0,
    'game_style': 'Competitive', 'rank': 'Quái vật cân team',
    'looking_for': 'Đồng đội lâu dài',
    'latitude': 10.77, 'longitude': 106.70,
    'interestedInGender': 'Tất cả',
    'minAge': 18, 'maxAge': 30, 'maxDistance': 50.0,
    'isPremium': False, 'isVerified': True,
    'num_interests': 5, 'num_games': 4, 'num_photos': 6,
    'bio_length': 120, 'profile_views': 200,
    'like_count': 50, 'match_count': 20,
}

user_casual = {
    'user_id': 'b2',
    'age': 25, 'gender': 'Nam', 'height': 172,
    'play_time': 500, 'win_rate': 40.0,
    'game_style': 'Casual', 'rank': 'Gà Mờ',
    'looking_for': 'Bạn chơi game',
    'latitude': 10.78, 'longitude': 106.71,
    'interestedInGender': 'Tất cả',
    'minAge': 18, 'maxAge': 35, 'maxDistance': 100.0,
    'isPremium': False, 'isVerified': False,
    'num_interests': 2, 'num_games': 1, 'num_photos': 2,
    'bio_length': 30, 'profile_views': 50,
    'like_count': 10, 'match_count': 5,
}

prediction, score = predict_compatibility(user_gamer, user_casual, model, scaler, feature_names)
print(f"Score: {score:.1f}% - {'✅ COMPATIBLE' if prediction else '❌ INCOMPATIBLE'}")
```

### 9.5 Phân Tích SHAP (Giải Thích Quyết Định)

```bash
pip install shap

python -c "
import joblib, shap, pandas as pd, json
from pathlib import Path

model = joblib.load('models/pairwise_compatibility_model.pkl')
feature_names = joblib.load('models/pairwise_feature_names.pkl')

# Load một vài samples
# ... (tạo X_sample với features)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
"
```

---

## 10. Cấu Trúc Thư Mục

```
gamenect_ai_training/
│
├── 📁 api/
│   └── main.py                 ← FastAPI server (endpoints /recommend, /health, /reload_model)
│
├── 📁 collectors/
│   ├── __init__.py
│   └── firestore_service.py    ← Thu thập data từ Firebase Firestore
│
├── 📁 data/
│   └── raw/                    ← JSON files từ Firebase (users.json, matches.json, ...)
│
├── 📁 models/                  ← Model đã train (binary pkl files)
│   ├── pairwise_compatibility_model.pkl  ← GradientBoosting model (~2.4 MB)
│   ├── pairwise_scaler.pkl               ← StandardScaler
│   └── pairwise_feature_names.pkl        ← List 37 feature names
│
├── 📁 pipelines/
│   ├── __init__.py
│   ├── dataset_builder.py      ← Build dataset từ JSON cho TF model
│   ├── feature_engineering.py  ← FeatureEngineer class (encode user → vector)
│   └── model_architecture.py   ← TensorFlow Deep Learning model (ResNet + Attention)
│
├── 📁 scripts/
│   ├── collect_from_firebase.py ← Chạy thu thập data
│   ├── preprocess_users.py      ← Tiền xử lý, xuất CSV
│   ├── train_user_model.py      ← ⭐ Train GBC model (main training script)
│   ├── test_pairwise_model.py   ← Test model với random pairs
│   ├── test_user_model.py       ← Test model đơn giản hơn
│   ├── evaluate_model.py        ← Đánh giá TF model
│   └── run_training_pipeline.sh ← Shell script chạy toàn bộ pipeline
│
├── 📁 services/
│   └── data_collector_api.py   ← FastAPI mini-service để trigger data collection
│
├── .env                        ← Biến môi trường (KHÔNG commit)
├── .env.example                ← Template biến môi trường
├── requirements.txt            ← Dependencies cơ bản
├── pyproject.toml              ← Dependencies đầy đủ + metadata dự án
├── Procfile                    ← Deploy lệnh cho Heroku/Railway
└── GAMENECT_AI_DOCS.md         ← File này
```

---

## 11. Biến Môi Trường

| Biến | Bắt buộc | Giá trị mẫu | Mô tả |
|------|---------|------------|-------|
| `FIREBASE_CREDENTIALS_PATH` | ✅ | `/path/to/firebase.json` | Service Account key JSON |
| `FIREBASE_PROJECT_ID` | ✅ | `gamenect-9bec0` | Firebase project ID |
| `RAW_DATA_DIR` | ❌ | `data/raw` | Thư mục lưu data thô |
| `PROCESSED_DATA_DIR` | ❌ | `data/processed` | Thư mục lưu data đã xử lý |
| `TRAINING_OUTPUT_DIR` | ❌ | `data/artifacts` | Thư mục lưu TF model |
| `TENSORBOARD_LOG_DIR` | ❌ | `logs/tensorboard` | TensorBoard logs |
| `TRAINING_HISTORY_DIR` | ❌ | `logs/training_history` | Lịch sử training |
| `REMOTE_MODEL_REGISTRY` | ❌ | `http://127.0.0.1:7000` | Model registry URL |
| `MODEL_DOWNLOAD_TOKEN` | ✅ (API) | `your-secret-token` | Token cho /reload_model |

---

## 12. Troubleshooting

### ❌ Lỗi: `FileNotFoundError: data/raw/users.json`

```bash
# Chưa thu thập data từ Firebase
python scripts/collect_from_firebase.py
```

### ❌ Lỗi: `Error loading models: No such file or directory: 'pairwise_compatibility_model.pkl'`

```bash
# Chưa train model, cần train trước:
python scripts/train_user_model.py
```

### ❌ Lỗi: Firebase credentials

```bash
# Kiểm tra file .env
cat .env | grep FIREBASE

# Kiểm tra file credentials tồn tại
ls -la $(cat .env | grep FIREBASE_CREDENTIALS_PATH | cut -d= -f2)
```

### ❌ Lỗi: `Module not found: collectors`

```bash
# Cài package ở development mode
pip install -e .

# Hoặc thêm project root vào PYTHONPATH
export PYTHONPATH=/Users/wang04/Downloads/gamenect_ai_training:$PYTHONPATH
```

### ⚠️ WARNING: ROC-AUC > 0.95 — Overfitting

Nguyên nhân: Labels được sinh tự động từ rule-based logic, nên model dễ "học thuộc" rules.

**Giải pháp:**
1. Thu thập thêm **real interaction data** (swipe, like thực tế)
2. Giảm `n_estimators` hoặc `max_depth`
3. Tăng `min_samples_split` và `min_samples_leaf`
4. Thêm noise vào labels

### ❌ TensorFlow không install được trên macOS

```bash
# Dùng đúng package cho Apple Silicon
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal==1.1.0

# Kiểm tra
python -c "import tensorflow as tf; print(tf.__version__)"
```

### ❌ Port 8000 đã bị chiếm

```bash
# Dùng port khác
uvicorn api.main:app --port 8080

# Hoặc kill process cũ
lsof -ti:8000 | xargs kill -9
```

---

## 📊 Tóm Tắt Flow Hoàn Chỉnh

```
Firebase Firestore
    │
    ▼ collect_from_firebase.py
data/raw/users.json
    │
    ▼ train_user_model.py
[Load → Clean → Create 10K pairs → 37 features/pair → Scale → GBC]
    │
    ▼
models/ (pkl files)
    │
    ▼ uvicorn api.main:app
FastAPI Server :8000
    │
    ▼ POST /recommend
Flutter App → [sorted recommendations by compatibility_score]
```

---

*Tài liệu được tạo tự động từ phân tích source code. Cập nhật lần cuối: 2026-04-20.*
