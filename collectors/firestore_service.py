from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import os
from rich.console import Console
from datetime import datetime
from google.cloud.firestore_v1._helpers import DatetimeWithNanoseconds

console = Console()
load_dotenv()


@dataclass
class FirestoreCollectorConfig:
    credentials_path: Path
    project_id: str
    raw_output_dir: Path
    batch_size: int = 500
    max_samples: int = 10000


def convert_firestore_data(obj):
    """Convert Firestore-specific types to JSON-serializable types"""
    if isinstance(obj, (DatetimeWithNanoseconds, datetime)):
        return obj.isoformat()  # Convert to ISO8601 string
    elif isinstance(obj, dict):
        return {key: convert_firestore_data(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_firestore_data(item) for item in obj]
    return obj


class FirestoreCollector:
    """Thu thập dữ liệu người dùng và tương tác từ Firestore."""

    def __init__(self, config: FirestoreCollectorConfig) -> None:
        self.config = config
        self._init_firestore()

    def _init_firestore(self) -> None:
        if not firebase_admin._apps:
            cred = credentials.Certificate(self.config.credentials_path)
            firebase_admin.initialize_app(cred, {"projectId": self.config.project_id})
            console.log("Da ket noi Firebase thanh cong")
        self.db = firestore.client()

    def _collect_collection(self, collection_name: str) -> List[Dict[str, Any]]:
        collection_ref = self.db.collection(collection_name)
        docs = collection_ref.stream()
        results = []
        for idx, doc in enumerate(docs):
            if self.config.max_samples and idx >= self.config.max_samples:
                break
            data = doc.to_dict()
            data["__id"] = doc.id
            results.append(data)
        console.log(f"Da thu thap {len(results)} documents tu {collection_name}")
        return results

    def collect_users(self) -> List[Dict[str, Any]]:
        return self._collect_collection("users")

    def collect_interactions(self):
        """Thu thập likes, matches, swipes từ Firebase"""
        try:
            # Collect likes
            likes_ref = self.db.collection('likes')
            likes = []
            for doc in likes_ref.stream():
                like = doc.to_dict()
                like['id'] = doc.id
                likes.append(like)
            
            # Collect matches
            matches_ref = self.db.collection('matches')
            matches = []
            for doc in matches_ref.stream():
                match = doc.to_dict()
                match['id'] = doc.id
                matches.append(match)
            
            # Collect swipes (nếu có)
            try:
                swipes_ref = self.db.collection('swipes')
                swipes = []
                for doc in swipes_ref.stream():
                    swipe = doc.to_dict()
                    swipe['id'] = doc.id
                    swipes.append(swipe)
            except:
                swipes = []
            
            # Collect swipe_history
            try:
                swipe_history_ref = self.db.collection('swipe_history')
                swipe_history = []
                for doc in swipe_history_ref.stream():
                    swipe = doc.to_dict()
                    swipe['id'] = doc.id
                    swipe_history.append(swipe)
                console.log(f"Da thu thap {len(swipe_history)} swipe_history")
            except Exception as e:
                console.log(f"Khong thu thap duoc swipe_history: {e}")
                swipe_history = []
            
            return {
                'likes': likes,
                'matches': matches,
                'swipes': swipes,
                'swipe_history': swipe_history,
                'collected_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error collecting interactions: {e}")
            return {'likes': [], 'matches': [], 'swipes': [], 'swipe_history': []}

    def collect_matches(self) -> List[Dict[str, Any]]:
        return self._collect_collection("matches")
    
    def collect_swipe_history(self) -> List[Dict[str, Any]]:
        return self._collect_collection("swipe_history")

    def collect_swipe_latest(self) -> List[Dict[str, Any]]:
        """Collect latest swipe snapshots from Firestore if collection exists."""
        try:
            return self._collect_collection("swipe_latest")
        except Exception as e:
            console.log(f"Khong thu thap duoc swipe_latest: {e}")
            return []

    def export_to_json(self) -> Dict[str, Path]:
        self.config.raw_output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        datasets = {
            "users": self.collect_users(),
            "interactions": self.collect_interactions(),
            "matches": self.collect_matches(),
            "swipe_history": self.collect_swipe_history(),
            "swipe_latest": self.collect_swipe_latest(),
        }
        for name, data in datasets.items():
            output_path = self.config.raw_output_dir / f"{name}.json"
            
            # Convert Firestore data before dumping to JSON
            converted_data = convert_firestore_data(data)
            
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            outputs[name] = output_path
            console.log(f"Da luu du lieu {name} tai {output_path}")
        return outputs


def main() -> None:
    credentials_path = Path(os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase.json"))
    raw_dir = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
    config = FirestoreCollectorConfig(
        credentials_path=credentials_path,
        project_id=os.getenv("FIREBASE_PROJECT_ID", ""),
        raw_output_dir=raw_dir,
        max_samples=20000,
    )
    collector = FirestoreCollector(config)
    collector.export_to_json()


if __name__ == "__main__":
    main()