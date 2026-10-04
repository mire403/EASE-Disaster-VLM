import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from floodnet_rcmtd.ease2026.runner import build_messages, resolve_local_model_path, verify_data_dir


class RunnerTests(unittest.TestCase):
    def test_model_must_be_local_linux_directory(self):
        with self.assertRaisesRegex(ValueError, "local model"):
            resolve_local_model_path("Qwen/Qwen3.5-9B")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "config.json"):
                resolve_local_model_path(temp)
            (Path(temp) / "config.json").write_text("{}")
            self.assertEqual(resolve_local_model_path(temp), str(Path(temp).resolve()))

    def test_messages_have_image_question_candidates_and_only_answer_target(self):
        row = {"image_path": "/tmp/example.jpg", "question": "Is water present?",
               "answer_space": ["yes", "no"]}
        messages = build_messages(row, "yes")
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0], {"type": "image", "path": "/tmp/example.jpg"})
        self.assertIn("yes, no", messages[0]["content"][1]["text"])
        self.assertEqual(messages[1], {"role": "assistant", "content": "yes"})

    def test_trace_context_is_training_only(self):
        row = {"image_path": "/tmp/example.jpg", "question": "Is water present?",
               "answer_space": ["yes", "no"], "target_source": "Trace",
               "trace_steps": [{"tool": "presence", "output": True}],
               "trace_support": {"yes": 0.9, "no": 0.1}}
        trained = build_messages(row, "yes")
        evaluated = build_messages(row)
        self.assertEqual(trained[0], evaluated[0])
        self.assertIn("presence", trained[1]["content"])
        self.assertTrue(trained[1]["content"].endswith("Answer:\nyes"))
        self.assertNotIn("presence", evaluated[0]["content"][1]["text"])

    def test_rejects_modified_dataset_file_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = {"question_id": "q1", "image_id": "i1", "split": "train", "mask_sha256": "h1"}
            test = {"question_id": "q2", "image_id": "i2", "split": "test", "mask_sha256": "h2"}
            (root / "train.jsonl").write_text(json.dumps(train) + "\n")
            (root / "test.jsonl").write_text(json.dumps(test) + "\n")
            (root / "manifest.json").write_text(json.dumps({"files": {"train": {"path": "train.jsonl", "sha256": "bad"}, "test": {"path": "test.jsonl", "sha256": "bad"}}}))
            with self.assertRaisesRegex(ValueError, "sha256"):
                verify_data_dir(root)

    def test_accepts_distinct_hashed_train_and_test_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {}
            for split, suffix in [("train", "1"), ("test", "2")]:
                payload = {"question_id": f"q{suffix}", "image_id": f"i{suffix}",
                           "split": split, "mask_sha256": f"hash{suffix}"}
                path = root / f"{split}.jsonl"
                path.write_text(json.dumps(payload) + "\n")
                files[split] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (root / "manifest.json").write_text(json.dumps({"files": files}))
            rows, manifest = verify_data_dir(root)
            self.assertEqual(len(rows["train"]), 1)
            self.assertEqual(manifest["files"]["test"]["path"], "test.jsonl")


if __name__ == "__main__":
    unittest.main()
