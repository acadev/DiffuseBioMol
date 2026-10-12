"""Real-domain proof that remote structural similarity differs from sequence identity."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from diffusebiomol.fold_pilot import build_map, train_pilot


ROOT = Path(__file__).parents[1]
CURATED = ROOT / "examples/fold-pilot/curated.json"
MAP = ROOT / "examples/fold-pilot/similarity_map.json"


class FoldPilotTests(unittest.TestCase):
    def test_remote_same_fold_pair_and_pair_separation(self):
        mapping = build_map(CURATED)
        self.assertEqual(mapping["summary"]["same_group_pairs"], 4)
        self.assertEqual(mapping["summary"]["different_group_pairs"], 11)
        remote = next(p for p in mapping["pairs"] if {p["a"], p["b"]} == {"1UBQ", "1A5R"})
        self.assertLess(remote["sequence_identity"], 0.25)
        self.assertGreater(remote["tm_score"], 0.6)
        self.assertGreater(mapping["summary"]["min_same_group_tm"],
                           mapping["summary"]["max_different_group_tm"])
        self.assertGreater(mapping["summary"]["structure_auc"],
                           mapping["summary"]["sequence_auc"])
        saved = json.loads(MAP.read_text())
        self.assertEqual(mapping["curated_sha256"], saved["curated_sha256"])
        self.assertAlmostEqual(mapping["summary"]["structure_auc"],
                               saved["summary"]["structure_auc"])

    def test_oracle_generation_connection_runs(self):
        with TemporaryDirectory() as tmp:
            result = train_pilot(CURATED, MAP, Path(tmp) / "run", epochs=2)
            self.assertEqual(result["holdout_ids"], ["1A5R", "1PGA"])
            self.assertTrue(result["baseline"]["finite_three_step_sample"])
            self.assertTrue(result["conditioned"]["finite_three_step_sample"])
            self.assertNotEqual(result["baseline"]["mean_validation_loss"],
                                result["conditioned"]["mean_validation_loss"])
            self.assertTrue((Path(tmp) / "run/pilot.json").exists())


if __name__ == "__main__":
    unittest.main()
