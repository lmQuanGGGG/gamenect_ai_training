from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any
from collectors.firestore_service import FirestoreCollector, FirestoreCollectorConfig
import os
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()
app = FastAPI(title="Gamenect Training Data Collector")


class CollectRequest(BaseModel):
    max_samples: int | None = 10000


@app.post("/collect")
def collect_data(request: CollectRequest) -> Dict[str, Any]:
    try:
        config = FirestoreCollectorConfig(
            credentials_path=Path(os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase.json")),
            project_id=os.getenv("FIREBASE_PROJECT_ID", ""),
            raw_output_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
            max_samples=request.max_samples or 10000,
        )
        collector = FirestoreCollector(config)
        outputs = collector.export_to_json()
        return {
            "status": "success",
            "files": {name: str(path) for name, path in outputs.items()},
        }
    except Exception as exc:
        console.print_exception()
        raise HTTPException(status_code=500, detail=str(exc))
