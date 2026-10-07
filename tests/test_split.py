import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from diffusebiomol.split import make_split


class SplitTests(unittest.TestCase):
    def test_source_groups_and_validation_cap(self):
        entries = [dict(file=f"{i}.json", source=f"pdb{i // 2}") for i in range(12)]
        corpus = SimpleNamespace(entries=entries)
        training, validation, metadata = make_split(corpus, 17,
            validation_fraction=0.5, max_validation_sources=3)
        self.assertEqual(len(validation), 2)
        self.assertEqual(set(training) | set(validation), set(range(12)))
        self.assertFalse({entries[i]["source"] for i in training} &
                         {entries[i]["source"] for i in validation})
        self.assertEqual(metadata["method"], "source")

    def test_cluster_groups_include_transitive_multichain_records(self):
        keys = [["1ABC_1"], ["2ABC_1"], ["3ABC_1", "3ABC_2"],
                ["4ABC_1"], ["5ABC_1"], ["6ABC_1"]]
        entries = [dict(file=f"{i}.json", entity_keys=k) for i, k in enumerate(keys)]
        corpus = SimpleNamespace(entries=entries)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "clusters.txt"
            path.write_text("1ABC_1 2ABC_1\n3ABC_1 4ABC_1\n3ABC_2 5ABC_1\n6ABC_1\n")
            training, validation, metadata = make_split(corpus, 3,
                sequence_clusters=path, validation_fraction=0.4,
                max_validation_sources=3)
            self.assertEqual(set(training) | set(validation), set(range(6)))
            for group in ({0, 1}, {2, 3, 4}):
                self.assertTrue(group <= set(training) or group <= set(validation))
            self.assertEqual(metadata["cluster_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            entries[5]["entity_keys"] = ["MISSING_1"]
            with self.assertRaisesRegex(ValueError, "coverage missing"):
                make_split(corpus, 3, sequence_clusters=path)


if __name__ == "__main__":
    unittest.main()
