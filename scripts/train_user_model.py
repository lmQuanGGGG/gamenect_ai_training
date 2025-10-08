import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json
import warnings
from itertools import combinations
from geopy.distance import geodesic

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

class PairwiseCompatibilityModel:
    """Model dự đoán độ tương thích giữa 2 người dùng (như Tinder/Bumble)"""
    
    def __init__(self):
        # Sử dụng Gradient Boosting - thuật toán mạnh cho classification
        self.model = GradientBoostingClassifier(
            n_estimators=300,          # Tăng từ 200 → 300
            max_depth=6,               # Tăng từ 5 → 6
            learning_rate=0.05,        # Giảm từ 0.1 → 0.05 (học chậm hơn, chính xác hơn)
            min_samples_split=10,      # Thêm regularization
            min_samples_leaf=5,        # Thêm regularization
            subsample=0.8,             # Thêm randomness
            random_state=42
        )
        self.scaler = StandardScaler()  # Chuẩn hóa dữ liệu
        self.feature_names = None
        
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Tính khoảng cách thực tế giữa 2 tọa độ (km)"""
        # Nếu thiếu tọa độ, trả về khoảng cách xa (1000km)
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return 1000
        try:
            # Dùng geodesic để tính khoảng cách chính xác trên bề mặt trái đất
            return geodesic((lat1, lon1), (lat2, lon2)).kilometers
        except:
            return 1000
    
    def create_pairwise_features(self, user1, user2):
        """
        Tạo các đặc trưng (features) cho cặp người dùng
        Đây là phần QUAN TRỌNG NHẤT - quyết định model học được gì
        """
        features = {}
        
        # ========== 1. ĐỘ TUỔI TƯƠNG THÍCH ==========
        age_diff = abs(user1['age'] - user2['age'])
        features['age_diff'] = age_diff                           # Chênh lệch tuổi
        features['age_compatible'] = 1 if age_diff <= 5 else 0   # Gần tuổi (<=5)
        features['age_within_10'] = 1 if age_diff <= 10 else 0   # Trong phạm vi 10 tuổi
        
        # ========== 2. KHOẢNG CÁCH ĐỊA LÝ ==========
        distance = self.calculate_distance(
            user1.get('latitude'), user1.get('longitude'),
            user2.get('latitude'), user2.get('longitude')
        )
        features['distance_km'] = min(distance, 1000)              # Giới hạn tối đa 1000km
        features['distance_within_10km'] = 1 if distance <= 10 else 0    # Trong 10km
        features['distance_within_50km'] = 1 if distance <= 50 else 0    # Trong 50km
        features['distance_within_100km'] = 1 if distance <= 100 else 0  # Trong 100km
        
        # ========== 3. GIỚI TÍNH & SỞ THÍCH ==========
        user1_gender = user1.get('gender', 'Unknown')
        user2_gender = user2.get('gender', 'Unknown')
        user1_looking_for = user1.get('interestedInGender', 'Tất cả')  # User 1 muốn tìm giới tính gì
        user2_looking_for = user2.get('interestedInGender', 'Tất cả')  # User 2 muốn tìm giới tính gì
        
        # Kiểm tra xem giới tính có match với sở thích không
        gender_match_1_to_2 = (user1_looking_for == 'Tất cả' or 
                               user1_looking_for == user2_gender)  # User 1 có thích giới tính của User 2?
        gender_match_2_to_1 = (user2_looking_for == 'Tất cả' or 
                               user2_looking_for == user1_gender)  # User 2 có thích giới tính của User 1?
        
        features['gender_match'] = 1 if (gender_match_1_to_2 and gender_match_2_to_1) else 0  # CẢ 2 đều match
        features['one_way_gender_match'] = 1 if (gender_match_1_to_2 or gender_match_2_to_1) else 0  # Chỉ 1 bên match
        
        # ========== 4. PHONG CÁCH CHƠI GAME ==========
        style1 = user1.get('game_style', 'Unknown')
        style2 = user2.get('game_style', 'Unknown')
        features['same_game_style'] = 1 if style1 == style2 else 0  # Cùng phong cách
        
        # Định nghĩa các phong cách tương thích với nhau
        compatible_styles = {
            'Casual': ['Casual', 'Vừa chơi vừa học'],                    # Casual phù hợp với Casual, Vừa học
            'Competitive': ['Competitive', 'Pro Player'],                # Competitive phù hợp với Pro
            'Pro Player': ['Pro Player', 'Competitive'],
            'Streamer': ['Streamer', 'Casual', 'Vừa chơi vừa học'],    # Streamer linh hoạt
            'Vừa chơi vừa học': ['Casual', 'Vừa chơi vừa học', 'Streamer']
        }
        
        is_compatible = False
        if style1 in compatible_styles and style2 in compatible_styles.get(style1, []):
            is_compatible = True
        features['compatible_game_style'] = 1 if is_compatible else 0
        
        # ========== 5. TRÌNH ĐỘ CHƠI GAME ==========
        win_rate_diff = abs(user1.get('win_rate', 50) - user2.get('win_rate', 50))
        features['win_rate_diff'] = win_rate_diff                       # Chênh lệch tỉ lệ thắng
        features['similar_skill'] = 1 if win_rate_diff <= 20 else 0    # Trình độ tương đương
        features['same_rank'] = 1 if user1.get('rank', '') == user2.get('rank', '') else 0  # Cùng rank
        
        # ========== 6. MỨC ĐỘ HOẠT ĐỘNG ==========
        play_time1 = user1.get('play_time', 0)
        play_time2 = user2.get('play_time', 0)
        # Tính tỉ lệ thời gian chơi (0-1, 1 là giống nhau hoàn toàn)
        play_time_ratio = min(play_time1, play_time2) / max(play_time1, play_time2, 1)
        features['play_time_ratio'] = play_time_ratio
        features['both_active'] = 1 if (play_time1 > 1000 and play_time2 > 1000) else 0  # Cả 2 đều active
        
        # ========== 7. ĐỘ CAM KẾT (PREMIUM/VERIFIED) ==========
        features['both_premium'] = 1 if (user1.get('isPremium', False) and user2.get('isPremium', False)) else 0
        features['both_verified'] = 1 if (user1.get('isVerified', False) and user2.get('isVerified', False)) else 0
        
        # Tính độ hoàn thiện profile trung bình
        avg_profile_completeness = (
            (user1.get('num_interests', 0) + user2.get('num_interests', 0)) / 10 +      # Số sở thích
            (user1.get('num_photos', 0) + user2.get('num_photos', 0)) / 6 +             # Số ảnh
            (user1.get('bio_length', 0) + user2.get('bio_length', 0)) / 200             # Độ dài bio
        ) / 3
        features['avg_profile_completeness'] = min(avg_profile_completeness, 1.0)
        
        # ========== 8. SỞ THÍCH CHUNG ==========
        # Tính độ tương đồng về số lượng sở thích (0-1)
        features['interest_count_similarity'] = 1 - abs(user1.get('num_interests', 0) - user2.get('num_interests', 0)) / 10
        features['game_count_similarity'] = 1 - abs(user1.get('num_games', 0) - user2.get('num_games', 0)) / 5
        
        # ========== 9. MỤC ĐÍCH TÌM KIẾM ==========
        looking_for_match = user1.get('looking_for', '') == user2.get('looking_for', '')
        features['same_looking_for'] = 1 if looking_for_match else 0  # Cùng mục đích (hẹn hò, bạn bè, etc)
        
        # ========== 10. ĐỘ PHỔ BIẾN ==========
        user1_pop = (user1.get('profile_views', 0) + user1.get('like_count', 0)) / 100
        user2_pop = (user2.get('profile_views', 0) + user2.get('like_count', 0)) / 100
        features['user1_popularity'] = min(user1_pop, 10)  # Giới hạn tối đa = 10
        features['user2_popularity'] = min(user2_pop, 10)
        features['popularity_balance'] = 1 - min(abs(user1_pop - user2_pop), 1)  # Cân bằng độ phổ biến
        
        # ========== 11. CHIỀU CAO ==========
        height_diff = abs(user1.get('height', 170) - user2.get('height', 170))
        features['height_diff'] = height_diff
        features['height_compatible'] = 1 if height_diff <= 15 else 0  # Chênh lệch <=15cm
        
        # ========== 12. SỞ THÍCH ĐỘ TUỔI ==========
        user1_min_age = user1.get('minAge', 18)
        user1_max_age = user1.get('maxAge', 99)
        user2_min_age = user2.get('minAge', 18)
        user2_max_age = user2.get('maxAge', 99)
        
        # Kiểm tra tuổi có nằm trong khoảng sở thích không
        age_pref_match_1_to_2 = (user1_min_age <= user2['age'] <= user1_max_age)  # User 2 trong range của User 1
        age_pref_match_2_to_1 = (user2_min_age <= user1['age'] <= user2_max_age)  # User 1 trong range của User 2
        
        features['age_preference_match'] = 1 if (age_pref_match_1_to_2 and age_pref_match_2_to_1) else 0  # CẢ 2 đều OK
        features['one_way_age_match'] = 1 if (age_pref_match_1_to_2 or age_pref_match_2_to_1) else 0      # Chỉ 1 bên OK
        
        # ========== 13. SỞ THÍCH KHOẢNG CÁCH ==========
        user1_max_distance = user1.get('maxDistance', 50)
        user2_max_distance = user2.get('maxDistance', 50)
        
        distance_pref_match_1 = distance <= user1_max_distance  # Khoảng cách OK với User 1
        distance_pref_match_2 = distance <= user2_max_distance  # Khoảng cách OK với User 2
        
        features['distance_preference_match'] = 1 if (distance_pref_match_1 and distance_pref_match_2) else 0
        features['one_way_distance_match'] = 1 if (distance_pref_match_1 or distance_pref_match_2) else 0
        
        # ========== THÊM FEATURES MỚI ==========
        
        # 14. SCORE TỔNG HỢP (Deal breakers)
        # Nếu vi phạm bất kỳ điều kiện quan trọng → điểm 0
        deal_breakers = 0
        if features['gender_match'] == 0:
            deal_breakers += 1
        if features['distance_km'] > user1.get('maxDistance', 50) and features['distance_km'] > user2.get('maxDistance', 50):
            deal_breakers += 1
        if features['age_preference_match'] == 0:
            deal_breakers += 1
        
        features['deal_breaker_count'] = deal_breakers
        features['has_deal_breakers'] = 1 if deal_breakers > 0 else 0
        
        # 15. COMPATIBILITY SCORE (tổng hợp các yếu tố)
        compatibility_factors = [
            features['gender_match'],
            features['age_preference_match'],
            features['distance_preference_match'],
            features['compatible_game_style'],
            features['similar_skill'],
            1 if features['distance_km'] <= 50 else 0,
            1 if features['age_diff'] <= 7 else 0,
        ]
        features['compatibility_factor_score'] = sum(compatibility_factors) / len(compatibility_factors)
        
        # 16. MỨC ĐỘ HOẠT ĐỘNG GAP
        activity_gap = abs(np.log1p(user1.get('play_time', 0)) - np.log1p(user2.get('play_time', 0)))
        features['activity_gap'] = activity_gap
        
        # 17. ENGAGEMENT SCORE
        engagement1 = (user1.get('profile_views', 0) + user1.get('like_count', 0) + user1.get('match_count', 0)) / 3
        engagement2 = (user2.get('profile_views', 0) + user2.get('like_count', 0) + user2.get('match_count', 0)) / 3
        features['avg_engagement'] = (engagement1 + engagement2) / 2
        features['engagement_gap'] = abs(engagement1 - engagement2)
        
        return features
    
    def create_training_pairs(self, df, num_pairs=10000):
        """
        Tạo các cặp user để train model - CÂN BẰNG HƠN
        """
        print(f"Creating {num_pairs} training pairs...")
        
        pairs = []
        labels = []
        
        # ========== CÂN BẰNG 50-50 ==========
        num_compatible = num_pairs // 2
        num_incompatible = num_pairs - num_compatible
        
        # ========== TẠO CẶP COMPATIBLE (50%) ==========
        compatible_count = 0
        attempts = 0
        max_attempts = num_compatible * 3  # Tối đa thử 3 lần số lượng cần
        
        while compatible_count < num_compatible and attempts < max_attempts:
            attempts += 1
            user1 = df.sample(1).iloc[0]
            
            # Tìm user tương thích - CÓ THỂ MATCH
            candidates = df[
                (df['user_id'] != user1['user_id']) &
                (df['age'] >= user1['age'] - 7) &   # Gần tuổi hơn
                (df['age'] <= user1['age'] + 7) &
                (df['win_rate'] >= user1['win_rate'] - 25) &  # Skill tương đương
                (df['win_rate'] <= user1['win_rate'] + 25)
            ]
            
            if len(candidates) > 0:
                user2 = candidates.sample(1).iloc[0]
                features = self.create_pairwise_features(user1, user2)
                
                # ✅ QUAN TRỌNG: Không quá strict, chấp nhận flexible matches
                # Chỉ cần KHÔNG vi phạm deal breakers nghiêm trọng
                if features['deal_breaker_count'] <= 1:  # Cho phép 1 deal breaker nhỏ
                    pairs.append(features)
                    labels.append(1)
                    compatible_count += 1
        
        print(f"Created {compatible_count} compatible pairs")
        
        # ========== TẠO CẶP INCOMPATIBLE (50%) ==========
        incompatible_count = 0
        
        while incompatible_count < num_incompatible:
            user1 = df.sample(1).iloc[0]
            user2 = df.sample(1).iloc[0]
            
            # Đảm bảo không phải cùng người
            while user1['user_id'] == user2['user_id']:
                user2 = df.sample(1).iloc[0]
            
            features = self.create_pairwise_features(user1, user2)
            
            # ✅ Label incompatible với tiêu chí RÕ RÀNG
            is_clear_incompatible = (
                features['deal_breaker_count'] >= 2 or  # Vi phạm 2+ deal breakers
                (features['age_diff'] > 15) or           # Quá chênh lệch tuổi
                (features['distance_km'] > 200) or       # Quá xa
                (features['win_rate_diff'] > 40)         # Skill gap quá lớn
            )
            
            # ✅ Tạo diversity: một số cặp có thể compatible
            is_maybe_compatible = (
                features['deal_breaker_count'] == 0 and
                features['compatibility_factor_score'] >= 0.5
            )
            
            pairs.append(features)
            if is_clear_incompatible:
                labels.append(0)
            elif is_maybe_compatible:
                labels.append(1)
            else:
                # Random để tạo diversity
                labels.append(np.random.choice([0, 1], p=[0.7, 0.3]))
            
            incompatible_count += 1
        
        print(f"Created {incompatible_count} incompatible pairs")
        
        return pd.DataFrame(pairs), np.array(labels)
    
    def train(self, X, y):
        """Train model với dữ liệu đã chuẩn bị"""
        # Xử lý giá trị thiếu (NaN) - thay bằng 0
        X = X.fillna(0)
        
        # Xử lý giá trị vô cực (inf)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Lưu tên các features
        self.feature_names = X.columns.tolist()
        
        # ✅ CHECK: In ra label distribution
        print(f"\nLabel distribution BEFORE split:")
        print(f"Compatible (1): {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"Incompatible (0): {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
        
        # Chia data thành train (80%) và test (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Chuẩn hóa dữ liệu (scale về mean=0, std=1)
        self.scaler.set_output(transform="pandas")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\nTraining pairwise compatibility model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Đánh giá model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {auc_score:.4f}")
        
        # ✅ WARNING nếu overfitting
        if auc_score > 0.95:
            print("\n⚠️  WARNING: AUC > 0.95 - Model có thể đang overfitting!")
            print("Xem xét giảm model complexity hoặc tăng regularization")
        
        return X_train, X_test, y_train, y_test
    
    def save_model(self, output_dir):
        """Lưu model và các preprocessor"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, output_dir / 'pairwise_compatibility_model.pkl')
        joblib.dump(self.scaler, output_dir / 'pairwise_scaler.pkl')
        joblib.dump(self.feature_names, output_dir / 'pairwise_feature_names.pkl')
        
        print(f"\nModel saved to {output_dir}")

def main():
    """Hàm chính - load data, train và save model"""
    # Đường dẫn file
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'raw' / 'users.json'
    model_dir = base_dir / 'models'
    
    # Load dữ liệu users
    print("Loading users data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        users_data = json.load(f)
    
    df_users = pd.DataFrame(users_data)
    
    # Trích xuất và làm sạch dữ liệu
    df = pd.DataFrame({
        'user_id': df_users['__id'].fillna(df_users.get('uid', df_users.get('id', ''))),
        'age': df_users['age'].fillna(25),
        'gender': df_users['gender'].fillna('Unknown'),
        'height': df_users['height'].fillna(170),
        'play_time': df_users.get('playTime', df_users.get('play_time', 0)).fillna(0),
        'win_rate': df_users.get('winRate', df_users.get('win_rate', 50)).fillna(50),
        'total_matches': df_users.get('totalMatches', df_users.get('total_matches', 0)).fillna(0),
        'friend_count': df_users.get('friendCount', df_users.get('friend_count', 0)).fillna(0),
        'profile_views': df_users.get('profileViews', df_users.get('profile_views', 0)).fillna(0),
        'like_count': df_users.get('likeCount', df_users.get('like_count', 0)).fillna(0),
        'match_count': df_users.get('matchCount', df_users.get('match_count', 0)).fillna(0),
        'game_style': df_users.get('gameStyle', df_users.get('game_style', 'Casual')).fillna('Casual'),
        'rank': df_users.get('rank', 'Unknown').fillna('Unknown'),
        'looking_for': df_users.get('lookingFor', df_users.get('looking_for', 'Bạn chơi game')).fillna('Bạn chơi game'),
        'isPremium': df_users.get('isPremium', df_users.get('is_premium', False)).fillna(False),
        'isVerified': df_users.get('isVerified', df_users.get('is_verified', False)).fillna(False),
        'num_interests': df_users.get('interests', []).apply(lambda x: len(x) if isinstance(x, list) else 0),
        'num_games': df_users.get('favoriteGames', []).apply(lambda x: len(x) if isinstance(x, list) else 0),
        'num_photos': df_users.get('additionalPhotos', []).apply(lambda x: len(x) if isinstance(x, list) else 0),
        'bio_length': df_users.get('bio', '').apply(lambda x: len(str(x)) if pd.notna(x) else 0),
        
        # Lấy latitude/longitude từ nested object 'location'
        'latitude': df_users['location'].apply(lambda x: x.get('latitude') if isinstance(x, dict) else None),
        'longitude': df_users['location'].apply(lambda x: x.get('longitude') if isinstance(x, dict) else None),
        
        'interestedInGender': df_users.get('interestedInGender', 'Tất cả').fillna('Tất cả'),
        'minAge': df_users.get('minAge', 18).fillna(18),
        'maxAge': df_users.get('maxAge', 99).fillna(99),
        'maxDistance': df_users.get('maxDistance', 50.0).fillna(50.0)
    })
    
    print(f"Loaded {len(df)} users")
    print(f"Missing latitude: {df['latitude'].isna().sum()}")
    print(f"Missing longitude: {df['longitude'].isna().sum()}")
    
    # Khởi tạo model
    model = PairwiseCompatibilityModel()
    
    # Tạo cặp training data
    X, y = model.create_training_pairs(df, num_pairs=10000)
    
    # Kiểm tra và xử lý NaN
    print(f"\nChecking for NaN values in features...")
    nan_columns = X.columns[X.isna().any()].tolist()
    if nan_columns:
        print(f"Columns with NaN: {nan_columns}")
        print(f"Filling NaN values with 0...")
        X = X.fillna(0)
    
    print(f"\nCreated {len(X)} training pairs")
    print(f"Label distribution: Compatible={sum(y)}, Incompatible={len(y)-sum(y)}")
    
    # Train model
    X_train, X_test, y_train, y_test = model.train(X, y)
    
    # Xem feature nào quan trọng nhất
    print("\nTop 15 Important Features:")
    feature_importance = pd.DataFrame({
        'feature': model.feature_names,
        'importance': model.model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(15))
    
    # Lưu model
    model.save_model(model_dir)
    
    print("\nTraining completed successfully!")
    print("\nKey Features Used:")
    print("✓ Age compatibility & preferences")
    print("✓ Distance & location preferences")
    print("✓ Gender matching & preferences")
    print("✓ Game style compatibility")
    print("✓ Skill level similarity")
    print("✓ Activity level matching")
    print("✓ Profile completeness")
    print("✓ Looking for compatibility")
    print("✓ Height preferences")
    print("✓ Popularity balance")

if __name__ == "__main__":
    main()
