from datetime import datetime
from pathlib import Path

from PIL import Image

from floodnet_rcmtd.data.records import FloodNetRecord


def _metadata(path: Path) -> tuple[datetime, str]:
    with Image.open(path) as image:
        exif = image.getexif()

    timestamp_value = exif.get(36867) or exif.get(306)
    if not timestamp_value:
        raise ValueError(f"missing EXIF timestamp: {path}")

    timestamp = datetime.strptime(str(timestamp_value), "%Y:%m:%d %H:%M:%S")
    camera_model = str(exif.get(272, "unknown")).rstrip("\x00")
    return timestamp, camera_model


def discover_pairs(root: Path) -> list[FloodNetRecord]:
    records = []
    for split in ("train", "val", "test"):
        image_dir = root / split / f"{split}-org-img"
        mask_dir = root / split / f"{split}-label-img"
        image_paths = sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem))

        for image_path in image_paths:
            image_id = image_path.stem
            mask_path = mask_dir / f"{image_id}_lab.png"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)

            timestamp, camera_model = _metadata(image_path)
            records.append(
                FloodNetRecord(
                    image_id=image_id,
                    original_split=split,
                    image_path=image_path,
                    mask_path=mask_path,
                    timestamp=timestamp,
                    camera_model=camera_model,
                )
            )

    return sorted(records, key=lambda record: int(record.image_id))
