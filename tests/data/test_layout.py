from datetime import datetime
from pathlib import Path

from floodnet_rcmtd.data.layout import discover_pairs


def test_discover_pairs_matches_image_and_mask_ids(tiny_floodnet_root: Path) -> None:
    records = discover_pairs(tiny_floodnet_root)
    assert [record.image_id for record in records] == ["100", "101", "200"]
    assert records[0].original_split == "train"
    assert records[0].mask_path.name == "100_lab.png"
    assert records[0].timestamp == datetime(2026, 6, 15, 12, 0, 0)
    assert records[0].camera_model == "TinyCam"

    val_record = records[2]
    assert val_record.original_split == "val"
    assert val_record.image_path == tiny_floodnet_root / "val" / "val-org-img" / "200.jpg"
    assert val_record.mask_path == tiny_floodnet_root / "val" / "val-label-img" / "200_lab.png"
