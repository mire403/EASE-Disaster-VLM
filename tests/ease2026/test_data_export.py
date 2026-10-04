import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from floodnet_rcmtd.data.questions import build_evidence, generate_questions
from floodnet_rcmtd.ease2026.prepare import prepare_from_manifest
from floodnet_rcmtd.ease2026.runner import verify_data_dir


def question_manifest(tmp_path, omit_family=False):
    images = []
    files = {}
    for split, values in [("train", [[3, 5], [2, 9]]), ("test", [[4, 5], [6, 9]])]:
        image = tmp_path / f"{split}.png"
        mask = tmp_path / f"{split}_mask.png"
        Image.new("RGB", (2, 2)).save(image)
        Image.fromarray(np.array(values, dtype=np.uint8)).save(mask)
        images.append({"image_id": split, "image_path": str(image), "mask_path": str(mask),
                       "group_id": split, "split": split})
        rows = []
        for question in generate_questions(split, build_evidence(np.array(values)), split=split):
            if omit_family and split == "test" and question.question_type == "presence":
                continue
            row = asdict(question)
            row["evidence"] = json.dumps(row["evidence"])
            row["answer_space"] = list(row["answer_space"])
            rows.append(row)
        path = tmp_path / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        files[split] = path.name
    pd.DataFrame(images).to_parquet(tmp_path / "images.parquet", index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": "images.parquet", "questions": files}))
    return manifest


def test_exports_six_families_with_trace_and_mask_statistics(tmp_path):
    manifest = question_manifest(tmp_path)
    output = tmp_path / "output"
    result = prepare_from_manifest(manifest, output, train_per_family=1, test_per_family=1)
    rows, _ = verify_data_dir(output)
    assert result["split_audit"]["question_counts"] == {"train": 6, "test": 6}
    assert all(row["trace_answer"] == row["gold_answer"] for row in rows["train"])
    assert all(sum(row["mask_evidence"]["area_pixels"].values()) == 4 for row in rows["test"])


def test_rejects_missing_question_family_in_a_split(tmp_path):
    manifest = question_manifest(tmp_path, omit_family=True)
    with pytest.raises(ValueError, match="famil"):
        prepare_from_manifest(manifest, tmp_path / "output", train_per_family=1, test_per_family=1)
