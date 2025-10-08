from collectors.firestore_service import FirestoreCollector, FirestoreCollectorConfig
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    config = FirestoreCollectorConfig(
        credentials_path=Path(os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase.json")),
        project_id=os.getenv("FIREBASE_PROJECT_ID", ""),
        raw_output_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
        max_samples=20000,
    )
    collector = FirestoreCollector(config)
    collector.export_to_json()


if __name__ == "__main__":
    main()
