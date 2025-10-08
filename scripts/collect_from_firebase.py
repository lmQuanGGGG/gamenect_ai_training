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
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETED")
    print("Files created:")
    for name, path in outputs.items():
        print(f"  - {name}: {path}")
    print("="*60)

if __name__ == "__main__":
    main()