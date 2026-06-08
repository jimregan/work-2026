import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ibm1_graphone_bootstrap.py"
EXAMPLE = ROOT / "examples" / "tiny_lexicon.tsv"


class BootstrapSmokeTest(unittest.TestCase):
    def test_bootstrap_outputs_seed_and_score_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            seeds = tmp_path / "seeds.tsv"
            scores = tmp_path / "scores.tsv"
            training = tmp_path / "training.tsv"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(EXAMPLE),
                    "--seeds-out",
                    str(seeds),
                    "--scores-out",
                    str(scores),
                    "--training-out",
                    str(training),
                    "--min-seed-count",
                    "1",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("Trained on", result.stdout)
            self.assertTrue(seeds.exists())
            self.assertTrue(scores.exists())
            self.assertTrue(training.exists())

            with seeds.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertTrue(
                any(row["language"] == "eng" and row["grapheme"] == "w" for row in rows)
            )
            self.assertTrue(
                any(row["language"] == "swe" and row["grapheme"] == "s" for row in rows)
            )

            with scores.open(encoding="utf-8", newline="") as handle:
                score_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertTrue(any(row["word"] == "halloweenkostym" for row in score_rows))


if __name__ == "__main__":
    unittest.main()
