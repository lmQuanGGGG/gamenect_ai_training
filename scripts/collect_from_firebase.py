"""
Script tự động thu thập data từ Firebase
Sử dụng FirestoreCollector đã có sẵn
"""
import sys
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.firestore_service import FirestoreCollector, FirestoreCollectorConfig

load_dotenv()

def main():
    print("="*60)
    print("AUTO COLLECT DATA FROM FIREBASE")
    print("="*60)
    
    # Setup config
    base_dir = Path(__file__).parent.parent
    credentials_path = Path(os.getenv("FIREBASE_CREDENTIALS_PATH"))
    raw_dir = base_dir / "data" / "raw"
    
    config = FirestoreCollectorConfig(
        credentials_path=credentials_path,
        project_id=os.getenv("FIREBASE_PROJECT_ID"),
        raw_output_dir=raw_dir,
        max_samples=20000
    )
    
    # Collect data
    print("\nInitializing Firebase collector...")
    collector = FirestoreCollector(config)
    
    print("\nCollecting data from Firebase...")
    outputs = collector.export_to_json()
    
    # Backup with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = raw_dir / f'users_backup_{timestamp}.json'
    
    import shutil
    if (raw_dir / 'users.json').exists():
        shutil.copy(raw_dir / 'users.json', backup_file)
        print(f"\nBackup saved to {backup_file}")
    
    import json
    import time
    from services.telegram_logger import telegram_logger
    
    users_count = 0
    matches_count = 0
    total_records = 0
    
    if (raw_dir / 'users.json').exists():
        with open(raw_dir / 'users.json', 'r') as f:
            data = json.load(f)
            users_count = len(data)
            total_records += users_count
            
    if (raw_dir / 'matches.json').exists():
        with open(raw_dir / 'matches.json', 'r') as f:
            data = json.load(f)
            matches_count = len(data)
            total_records += matches_count
            
    if (raw_dir / 'swipes.json').exists():
        with open(raw_dir / 'swipes.json', 'r') as f:
            data = json.load(f)
            matches_count += len(data)
            total_records += len(data)

    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETED")
    print("Files created:")
    for name, path in outputs.items():
        print(f"  - {name}: {path}")
    print("="*60)
    
    # Gửi Telegram Report
    try:
        global start_time
        time_taken = time.time() - start_time
    except NameError:
        time_taken = 0.0
    telegram_logger.log_data_sync(users_count, matches_count, time_taken, total_records)

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()