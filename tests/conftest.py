from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tiny_floodnet_root(tmp_path: Path) -> Path:
    samples = {
        "train": [
            ("100", "2026:06:15 12:00:00", "TinyCam"),
            ("101", "2020:08:01 09:01:00", "FloodCam-A"),
        ],
        "val": [("200", "2020:08:02 12:00:00", "FloodCam-B")],
        "test": [],
    }

    for split, split_samples in samples.items():
        image_dir = tmp_path / split / f"{split}-org-img"
        mask_dir = tmp_path / split / f"{split}-label-img"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        for image_id, timestamp, camera_model in split_samples:
            exif = Image.Exif()
            exif[306] = timestamp
            exif[36867] = timestamp
            exif[272] = camera_model

            image = Image.new("RGB", (8, 8), color=(32, 64, 96))
            image.save(image_dir / f"{image_id}.jpg", exif=exif)

            mask = Image.new("L", (8, 8), color=1)
            mask.save(mask_dir / f"{image_id}_lab.png")

    return tmp_path
