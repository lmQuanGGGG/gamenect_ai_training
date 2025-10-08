#!/bin/bash

echo "=========================================="
echo "GameNect AI Training Pipeline"
echo "=========================================="

# Step 1: Preprocess data
echo -e "\n[Step 1/3] Preprocessing users data..."
python scripts/preprocess_users.py

# Step 2: Train model
echo -e "\n[Step 2/3] Training compatibility model..."
python scripts/train_user_model.py

# Step 3: Test model
echo -e "\n[Step 3/3] Testing model..."
python scripts/test_user_model.py

echo -e "\n=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
