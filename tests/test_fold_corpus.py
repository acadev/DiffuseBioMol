import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from diffusebiomol.fold_sampling import epoch_sources
from diffusebiomol.distributed import rank_batches
from diffusebiomol.prepare_fold_corpus import prepare_fold_corpus
from diffusebiomol.split import make_split
from diffusebiomol.train import train


def pdb_record():
    lines = []
    serial = 0
    for residue in range(1, 5):
        for atom, element, dx in (("N", "N", 0), ("CA", "C", 1),
                                  ("C", "C", 2), ("O", "O", 3)):
            serial += 1
            lines.append(f"ATOM  {serial:5d} {atom:<4s} ALA A{residue:4d}    "
                         f"{residue * 4 + dx:8.3f}{0.:8.3f}{0.:8.3f}"
                         f"{1.:6.2f}{20.:6.2f}          {element:>2s}\n")
    return "".join(lines) + "TER\nEND\n"


class FoldCorpusTests(unittest.TestCase):
    def test_cath_metadata_parallel_records_and_group_split(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            domains = root / "domains"
            domains.mkdir()
            rows = []
            for i, (fold, family) in enumerate(((10, 1), (10, 1), (10, 2),
                                                 (20, 1), (20, 2), (30, 1)), 1):
                domain = f"1ab{i}A00"
                (domains / f"{domain}.pdb").write_text(pdb_record())
                rows.append(f"{domain} 1 {fold} 1 {family} 1 1 1 1 1 4 2.000\n")
            classification = root / "cath-domain-list.txt"
            classification.write_text("# CATH list\n" + "".join(rows))
            selection = root / "S40.list"
            selection.write_text("".join(row.split()[0] + "\n" for row in rows))
            output = root / "corpus"
            manifest = prepare_fold_corpus(domains, classification, output,
                selection=selection, workers=2, min_residues=3)
            self.assertEqual(len(manifest["entries"]), 6)
            self.assertEqual(manifest["preparation"]["fold_counts"],
                             {"1.10.1": 3, "1.20.1": 2, "1.30.1": 1})
            self.assertEqual(manifest["preparation"]["classification_sha256"],
                             hashlib.sha256(classification.read_bytes()).hexdigest())
            for entry in manifest["entries"]:
                self.assertEqual(entry["residues"], 4)
                self.assertTrue((output / entry["file"]).exists())
                self.assertEqual(len(json.loads((output / entry["file"]).read_text())["element"]),
                                 entry["atoms"])
            corpus = SimpleNamespace(entries=manifest["entries"])
            training, valid, meta = make_split(corpus, 17, group_by="superfamily",
                max_validation_sources=3)
            self.assertEqual(meta["method"], "superfamily")
            self.assertFalse({corpus.entries[i]["superfamily_id"] for i in training} &
                             {corpus.entries[i]["superfamily_id"] for i in valid})
            draws = epoch_sources(training, corpus, 1, 17,
                fold_sampling="uniform_fold", presentations_per_epoch=1000)
            self.assertEqual(draws, epoch_sources(training, corpus, 1, 17,
                fold_sampling="uniform_fold", presentations_per_epoch=1000))
            counts = {fold: sum(corpus.entries[i]["fold_id"] == fold for i in draws)
                      for fold in {corpus.entries[i]["fold_id"] for i in training}}
            self.assertLess(max(counts.values()) - min(counts.values()), 150)
            shards = [rank_batches(training, corpus, 2, 3, rank, 1, 17,
                fold_sampling="uniform_fold", presentations_per_epoch=11)
                for rank in range(3)]
            self.assertEqual(sum(len(batch) for rank in shards for batch in rank), 11)
            model, summary = train(output, root / "run", epochs=1, batch_size=2,
                max_residues=4, threads=1, split_group="superfamily",
                fold_sampling="uniform_fold", presentations_per_epoch=12)
            self.assertEqual(summary["presentations"], 12)
            self.assertTrue((root / "run/checkpoint.pt").is_file())

    def test_missing_coordinates_and_size_filter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            domains = root / "domains"
            domains.mkdir()
            (domains / "1abcA00.pdb").write_text(pdb_record())
            classification = root / "cath-domain-list.txt"
            classification.write_text("1abcA00 1 10 1 1 1 1 1 1 1 4 2.000\n"
                                      "2abcA00 1 10 1 2 1 1 1 1 1 4 2.000\n")
            with self.assertRaisesRegex(ValueError, "Missing 1"):
                prepare_fold_corpus(domains, classification, root / "strict", strict=True)
            with self.assertRaisesRegex(ValueError, "Corpus incomplete"):
                prepare_fold_corpus(domains, classification, root / "filtered",
                                    min_residues=5)
            manifest = json.loads((root / "filtered/manifest.json").read_text())
            self.assertEqual(manifest["preparation"]["missing_coordinates"], 1)
            self.assertEqual(len(manifest["skipped"]), 1)


if __name__ == "__main__":
    unittest.main()
