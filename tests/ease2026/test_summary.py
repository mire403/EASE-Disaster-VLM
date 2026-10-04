import json
import tempfile
import unittest
from pathlib import Path

from floodnet_rcmtd.ease2026.summary import collect_runs, summarize_predictions


class SummaryTests(unittest.TestCase):
    def test_candidate_metrics_and_generation_without_confidence_ece(self):
        rows = [
            {"question_id": "q1", "question_type": "presence", "answer": "yes", "prediction": "yes", "confidence": 0.75},
            {"question_id": "q2", "question_type": "presence", "answer": "no", "prediction": "yes", "confidence": 0.75},
        ]
        result = summarize_predictions(rows)
        self.assertEqual(result["overall"]["accuracy"], 0.5)
        self.assertAlmostEqual(result["overall"]["macro_f1"], 1 / 3)
        self.assertAlmostEqual(result["overall"]["ece"], 0.25)
        for row in rows:
            row["confidence"] = None
        self.assertIsNone(summarize_predictions(rows)["overall"]["ece"])

    def test_collect_rejects_mismatched_question_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for cell, qid in [("PC-Ans", "q1"), ("AA-Ans", "q2")]:
                folder = root / "qwen" / "seed1" / cell / "eval_candidate"
                folder.mkdir(parents=True)
                (folder / "meta.json").write_text(json.dumps({"test_sha256": "same"}))
                (folder / "predictions.jsonl").write_text(json.dumps({"question_id": qid, "image_id": qid, "question_type": "presence", "answer": "yes", "prediction": "yes", "confidence": 0.9}) + "\n")
            with self.assertRaisesRegex(ValueError, "membership"):
                collect_runs(root, "qwen", [1], "candidate", cells=["PC-Ans", "AA-Ans"])

    def test_collect_computes_cell_means_on_same_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for cell, prediction in [("PC-Ans", "no"), ("AA-Ans", "yes")]:
                folder = root / "qwen" / "seed1" / cell / "eval_candidate"
                folder.mkdir(parents=True)
                (folder / "meta.json").write_text(json.dumps({"test_sha256": "same"}))
                (folder / "predictions.jsonl").write_text(json.dumps({
                    "question_id": "q1", "image_id": "i1", "question_type": "presence",
                    "answer": "yes", "prediction": prediction, "confidence": 0.75,
                }) + "\n")
            summary = collect_runs(root, "qwen", [1], "candidate", cells=["PC-Ans", "AA-Ans"])
            self.assertEqual(summary["aggregate"]["PC-Ans"]["accuracy"]["mean"], 0)
            self.assertEqual(summary["aggregate"]["AA-Ans"]["accuracy"]["mean"], 1)


if __name__ == "__main__":
    unittest.main()

class AlignmentGapTests(unittest.TestCase):
    def test_reports_paired_gap_from_all_four_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for cell, prediction in [("PC-Ans", "no"), ("PC-Trace", "no"),
                                     ("AA-Ans", "yes"), ("AA-Trace", "yes")]:
                folder = root / "qwen" / "seed1" / cell / "eval_candidate"
                folder.mkdir(parents=True)
                (folder / "meta.json").write_text(json.dumps({"test_sha256": "same"}))
                (folder / "predictions.jsonl").write_text(json.dumps({
                    "question_id": "q1", "image_id": "i1", "question_type": "presence",
                    "answer": "yes", "prediction": prediction, "confidence": 0.75,
                }) + "\n")
            result = collect_runs(root, "qwen", [1], "candidate")
            self.assertEqual(result["alignment_gap"]["mean"], 1.0)
            self.assertEqual(result["aggregate_by_question_type"]["AA-Ans"]["presence"]["accuracy"]["mean"], 1.0)
