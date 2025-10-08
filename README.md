# Gamenect AI Training Workspace

## Các bước khởi tạo
1. python3 -m venv venv
2. source venv/bin/activate
3. pip install --upgrade pip
4. pip install -e .

## Thu thập dữ liệu
python scripts/collect_from_firebase.py

## Huấn luyện
python scripts/train_user_model.py

## Đánh giá
python scripts/evaluate_model.py
# gamenect_ai_training
