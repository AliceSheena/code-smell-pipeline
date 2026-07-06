import unittest

from localisation.ast_detector import detect_inplace_api_misuse
from evaluation.metrics import exact_match, line_iou
from data.prepare_dataset import verify_entry


class SmokeTests(unittest.TestCase):
    def test_ast_detector_detects_inplace_keyword(self):
        source = "df.drop(columns=['a'], inplace=True)\n"
        result = detect_inplace_api_misuse(source)
        self.assertEqual(result["smelly_lines"], [1])
        self.assertEqual(result["smelly_ranges"], [(1, 1)])

    def test_line_iou(self):
        self.assertAlmostEqual(line_iou([(1, 2)], [(2, 3)]), 1 / 3)
        self.assertEqual(line_iou([], []), 1.0)

    def test_exact_match(self):
        self.assertTrue(exact_match("foo\n", "foo"))
        self.assertFalse(exact_match("foo", "bar"))

    def test_verify_entry_valid(self):
        entry = {
            "smell_type": "In-Place APIs Misused",
            "smell_location": {"start_line": 1, "end_line": 1, "smelly_lines": [1]},
            "code_smell_code": "df.drop(columns=['a'], inplace=True)\n",
            "refactoring_code": "df = df.drop(columns=['a'])\n",
        }
        ok, reason = verify_entry(entry)
        self.assertTrue(ok)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
