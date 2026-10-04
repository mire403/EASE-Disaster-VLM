import unittest

from floodnet_rcmtd.ease2026.protocol import (
    balanced_sample,
    make_cells,
    make_labels,
    validate_splits,
)


def row(question_id, image_id, split, family="presence", mask_sha256=None):
    return {
        "question_id": question_id,
        "image_id": image_id,
        "split": split,
        "question_type": family,
        "mask_sha256": mask_sha256 or f"hash-{image_id}",
        "gold_answer": "yes",
        "trace_answer": "yes",
        "trace_support": {"yes": 0.9, "no": 0.1},
        "answer_space": ["yes", "no"],
    }


class SplitTests(unittest.TestCase):
    def test_rejects_image_overlap(self):
        with self.assertRaisesRegex(ValueError, "image_id"):
            validate_splits({"train": [row("q1", "im1", "train")], "test": [row("q2", "im1", "test")]})

    def test_rejects_identical_mask_across_images(self):
        with self.assertRaisesRegex(ValueError, "mask_sha256"):
            validate_splits({"train": [row("q1", "im1", "train", mask_sha256="same")], "test": [row("q2", "im2", "test", mask_sha256="same")]})

    def test_rejects_question_overlap(self):
        with self.assertRaisesRegex(ValueError, "question_id"):
            validate_splits({"train": [row("q1", "im1", "train")], "test": [row("q1", "im2", "test")]})

    def test_rejects_duplicate_question_within_split(self):
        item = row("q1", "im1", "train")
        with self.assertRaisesRegex(ValueError, "duplicate question_id"):
            validate_splits({"train": [item, dict(item)]})

    def test_rejects_sequence_group_overlap(self):
        first = row("q1", "im1", "train")
        second = row("q2", "im2", "test")
        first["group_id"] = second["group_id"] = "sequence-a"
        with self.assertRaisesRegex(ValueError, "group_id"):
            validate_splits({"train": [first], "test": [second]})

    def test_balanced_sample_is_deterministic(self):
        rows = [row(f"p{i}", f"pi{i}", "train") for i in range(5)] + [row(f"a{i}", f"ai{i}", "train", "area_comparison") for i in range(5)]
        selected = balanced_sample(rows, 2, seed=42)
        self.assertEqual(len(selected), 4)
        self.assertEqual([r["question_id"] for r in selected], [r["question_id"] for r in balanced_sample(list(reversed(rows)), 2, seed=42)])


class CellTests(unittest.TestCase):
    def test_all_cells_use_same_order_and_only_target_axis_changes_answer(self):
        rows = [row("q1", "im1", "train"), row("q2", "im2", "train")]
        cells = make_cells(rows, seed=7)
        self.assertEqual(set(cells), {"PC-Ans", "PC-Trace", "AA-Ans", "AA-Trace"})
        orders = [[r["question_id"] for r in cell] for cell in cells.values()]
        self.assertTrue(all(order == orders[0] for order in orders))
        self.assertEqual([r["target_answer"] for r in cells["PC-Ans"]], [r["target_answer"] for r in cells["AA-Ans"]])
        self.assertEqual([r["target_answer"] for r in cells["PC-Trace"]], [r["target_answer"] for r in cells["AA-Trace"]])
        self.assertEqual(cells["PC-Trace"][0]["target_source"], "Trace")

    def test_trace_target_must_match_verified_answer(self):
        bad = row("q1", "im1", "train")
        bad["trace_answer"] = "no"
        with self.assertRaisesRegex(ValueError, "trace/gold"):
            make_cells([bad], seed=7)

    def test_loss_masks_differ_only_before_assistant(self):
        ids = [5, 6, 7, 8, 0]
        prompt = [5, 6, 7]
        self.assertEqual(make_labels(ids, prompt, 0, "PC"), [5, 6, 7, 8, -100])
        self.assertEqual(make_labels(ids, prompt, 0, "AA"), [-100, -100, -100, 8, -100])

    def test_rejects_non_prefix_prompt(self):
        with self.assertRaisesRegex(ValueError, "prefix"):
            make_labels([5, 6, 7], [5, 9], None, "AA")

    def test_answer_span_excludes_evidence_and_chat_suffix(self):
        ids = [5, 6, 7, 8, 9, 10, 11, 0]
        prefix = [5, 6, 7, 8]
        self.assertEqual(make_labels(ids, prefix, 0, "AA", answer_end=6),
                         [-100, -100, -100, -100, 9, 10, -100, -100])
        self.assertEqual(make_labels(ids, prefix, 0, "PC", answer_end=6),
                         [5, 6, 7, 8, 9, 10, 11, -100])

    def test_rejects_empty_or_out_of_range_answer_span(self):
        for end in (2, 5):
            with self.assertRaisesRegex(ValueError, "answer"):
                make_labels([5, 6, 7, 8], [5, 6], None, "AA", answer_end=end)


if __name__ == "__main__":
    unittest.main()
