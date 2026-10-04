import copy
import unittest


def prediction(qid, label, gold="yes"):
    return {"question_id": qid, "image_id": qid, "question_type": "presence",
            "answer": gold, "prediction": label, "confidence": None}


class DiagnosticTests(unittest.TestCase):
    def runs(self):
        return {
            "Zero-shot": [prediction("q1", "no"), prediction("q2", "no"), prediction("q3", "yes"), prediction("q4", "no")],
            "AA-Ans": [prediction("q1", "yes"), prediction("q2", "no"), prediction("q3", "yes"), prediction("q4", "no")],
            "AA-Trace": [prediction("q1", "yes"), prediction("q2", "yes"), prediction("q3", "no"), prediction("q4", "no")],
            "AA-Ans-second-seed": [prediction("q1", "yes"), prediction("q2", "yes"), prediction("q3", "yes"), prediction("q4", "no")],
        }

    def test_counts_keep_help_hurt_hard_and_seed_changes_distinct(self):
        from floodnet_rcmtd.ease2026.diagnostics import compare_predictions
        result = compare_predictions(self.runs())
        self.assertEqual(result["counts"], {
            "aa_ans_fixes_zero_shot": 1, "trace_helps_aa_ans": 1,
            "trace_hurts_aa_ans": 1, "remaining_hard": 1, "seed_disagreements": 1,
        })
        self.assertEqual(result["by_question_type"]["presence"], result["counts"])
        self.assertEqual(result["confusion_matrices"]["AA-Ans"]["presence"]["yes"], {"no": 2, "yes": 2})

    def test_matches_question_ids_even_when_row_orders_differ(self):
        from floodnet_rcmtd.ease2026.diagnostics import compare_predictions
        runs = self.runs()
        runs["AA-Trace"].reverse()
        self.assertEqual(compare_predictions(runs)["counts"]["trace_helps_aa_ans"], 1)

    def test_rejects_duplicate_or_changed_question_membership(self):
        from floodnet_rcmtd.ease2026.diagnostics import compare_predictions
        runs = self.runs()
        runs["AA-Trace"].append(copy.deepcopy(runs["AA-Trace"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_predictions(runs)
        runs = self.runs()
        runs["AA-Trace"][0]["image_id"] = "different"
        with self.assertRaisesRegex(ValueError, "membership"):
            compare_predictions(runs)

    def test_mask_margin_uses_total_pixels_and_support_is_checked(self):
        from floodnet_rcmtd.ease2026.diagnostics import evidence_statistics
        row = {"gold_answer": "tree", "trace_support": {"tree": 0.6, "water": 0.4},
               "mask_evidence": {"area_pixels": {"tree": 51, "water": 49}}}
        stats = evidence_statistics(row)
        self.assertAlmostEqual(stats["mask_area_margin"], 0.02)
        self.assertAlmostEqual(stats["trace_support_margin"], 0.2)
        self.assertEqual(stats["trace_gold_support"], 0.6)
        row["trace_support"]["water"] = 0.7
        with self.assertRaisesRegex(ValueError, "sum"):
            evidence_statistics(row)

    def test_area_comparison_margin_uses_the_two_road_classes(self):
        from floodnet_rcmtd.ease2026.diagnostics import evidence_statistics
        row = {"question_type": "area_comparison", "gold_answer": "road_flooded",
               "mask_evidence": {"area_pixels": {"road_flooded": 51, "road_non_flooded": 49, "grass": 900}}}
        self.assertAlmostEqual(evidence_statistics(row)["mask_area_margin"], 0.002)

    def test_analysis_writes_paired_cases_and_rejects_changed_test_hash(self):
        import csv
        import json
        import tempfile
        from pathlib import Path
        from floodnet_rcmtd.ease2026.diagnostics import analyze_runs
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.runs()
            for cell in ("Zero-shot", "PC-Ans", "PC-Trace", "AA-Ans", "AA-Trace"):
                folder = root / "qwen" / "seed1" / cell / "eval_generation"
                folder.mkdir(parents=True)
                (folder / "meta.json").write_text(json.dumps({"test_sha256": "same"}))
                (folder / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows.get(cell, rows["Zero-shot"])))
            output = root / "analysis"
            result = analyze_runs(root, "qwen", 1, "generation", output)
            self.assertEqual(result["counts"]["trace_hurts_aa_ans"], 1)
            with (output / "cases.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            (root / "qwen/seed1/AA-Trace/eval_generation/meta.json").write_text(json.dumps({"test_sha256": "changed"}))
            with self.assertRaisesRegex(ValueError, "sha256"):
                analyze_runs(root, "qwen", 1, "generation", output)
