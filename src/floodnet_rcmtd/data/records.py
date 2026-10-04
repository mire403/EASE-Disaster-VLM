from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class FloodNetRecord:
    image_id: str
    original_split: str
    image_path: Path
    mask_path: Path
    timestamp: datetime
    camera_model: str
