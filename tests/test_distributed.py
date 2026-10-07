"""CPU process-group regression for partitioning and exact epoch-boundary resume."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist

from diffusebiomol.data import Corpus
from diffusebiomol.distributed import rank_batches


ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "examples" / "smoke-corpus"


def equal_state(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            equal_state(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            equal_state(test, a, b)
    else:
        test.assertEqual(left, right)


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
class DistributedTrainingTests(unittest.TestCase):
    def test_no_source_repeats_or_omissions(self):
        corpus = Corpus(CORPUS)
        training = [0, 1, 2, 3, 4]
        for epoch in (1, 2):
            by_rank = [rank_batches(training, corpus, 1, 3, rank, epoch, 17)
                       for rank in range(3)]
            self.assertEqual({len(batches) for batches in by_rank}, {2})
            seen = [index for batches in by_rank for batch in batches for index in batch]
            self.assertEqual(sorted(seen), training)

    def test_three_ranks_resume_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            def launch(name, epochs, resume=False):
                # Reserve a local port, then let torchrun bind it.
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                cmd = [sys.executable, "-m", "torch.distributed.run",
                    "--nnodes=1", "--nproc-per-node=3", "--master-addr=127.0.0.1",
                    f"--master-port={port}", "-m", "diffusebiomol.train",
                    str(CORPUS), str(root / name), "--epochs", str(epochs),
                    "--max-residues", "4", "--batch-size", "1", "--threads", "1",
                    "--device", "cpu"]
                if resume:
                    cmd.append("--resume")
                result = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout)
            launch("full", 2)
            launch("resumed", 1)
            launch("resumed", 2, resume=True)
            full = torch.load(root / "full/checkpoint.pt", weights_only=True)
            resumed = torch.load(root / "resumed/checkpoint.pt", weights_only=True)
            for key in ("model", "optimizer", "rank_states", "presentations", "updates"):
                equal_state(self, full[key], resumed[key])
            self.assertEqual(full["presentations"], 10)
            self.assertEqual(full["updates"], 4)
            self.assertEqual(len(full["rank_states"]), 3)


if __name__ == "__main__":
    unittest.main()
