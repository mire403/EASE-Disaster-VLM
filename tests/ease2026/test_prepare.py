import json
import tempfile
import unittest
from pathlib import Path

from floodnet_rcmtd.ease2026.prepare import assemble_records


class PreparationTests(unittest.TestCase):
    def test_assembles_separate_records_with_mask_hash_and_trace_support(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = []
            questions = {}
            for split, suffix in [("train", "1"), ("test", "2")]:
                image = root / f"{suffix}.jpg"
                mask = root / f"{suffix}.png"
                image.write_bytes(f"image{suffix}".encode())
                mask.write_bytes(f"mask{suffix}".encode())
                images.append({"image_id": suffix, "image_path": str(image), "mask_path": str(mask), "split": split})
                questions[split] = [{"question_id": f"q{suffix}", "image_id": suffix, "split": split,
                                    "question_type": "presence", "question": "Visible?", "answer": "yes",
                                    "answer_space": json.dumps(["yes", "no"]), "evidence": "{}", "template_id": "t"}]
            rows = assemble_records(questions, images, lambda question, mask: ("yes", {"yes": 0.9, "no": 0.1}))
            self.assertEqual(rows["train"][0]["gold_answer"], "yes")
            self.assertEqual(rows["test"][0]["trace_answer"], "yes")
            self.assertNotEqual(rows["train"][0]["mask_sha256"], rows["test"][0]["mask_sha256"])

    def test_rejects_question_image_split_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, mask = root / "i.jpg", root / "m.png"
            image.write_bytes(b"image")
            mask.write_bytes(b"mask")
            with self.assertRaisesRegex(ValueError, "split"):
                assemble_records({"test": [{"question_id": "q", "image_id": "i", "split": "test",
                                             "question_type": "presence", "question": "Visible?", "answer": "yes",
                                             "answer_space": ["yes", "no"], "evidence": {}, "template_id": "t"}]},
                                 [{"image_id": "i", "image_path": str(image), "mask_path": str(mask), "split": "train"}],
                                 lambda question, mask: ("yes", {"yes": 1.0, "no": 0.0}))


if __name__ == "__main__":
    unittest.main()
