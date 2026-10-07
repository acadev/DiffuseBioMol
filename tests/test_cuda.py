"""Run on a CUDA worker; skipped on machines without NVIDIA GPUs."""
from pathlib import Path
import tempfile
import unittest

import torch

from diffusebiomol.data import collate
from diffusebiomol.model import FlowModel, ModelConfig
from diffusebiomol.train import train


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class CudaPilotTests(unittest.TestCase):
    def test_forward_and_gradient_match_cpu(self):
        torch.manual_seed(7)
        config = ModelConfig()
        cpu = FlowModel(config, [17, 6, 20, 66])
        gpu = FlowModel(config, [17, 6, 20, 66]).cuda()
        gpu.load_state_dict(cpu.state_dict())
        record = dict(element=[0, 1, 2, 0], modality=[0]*4,
            polymer=[0, 1, 2, 3], chain=[1]*4, residue=[0, 0, 1, 1],
            virtual=[False]*4, xyz=[[0., 0., 0.]]*4)
        batch = collate([record])
        x = torch.randn(1, 4, 3)
        time = torch.tensor([0.37])
        cpu.head.weight.data.normal_(std=0.1)
        gpu.load_state_dict(cpu.state_dict())
        expected = cpu(batch, x, time)
        actual = gpu({k: v.cuda() for k, v in batch.items()}, x.cuda(), time.cuda())
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(gpu.coord.weight.grad.cpu(), cpu.coord.weight.grad,
                                   atol=2e-4, rtol=2e-4)

    def test_training_resume_and_memory_metrics(self):
        corpus = Path(__file__).parents[1] / "examples" / "smoke-corpus"
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "gpu-run"
            train(corpus, run, device="cuda", epochs=1, max_residues=4)
            train(corpus, run, device="cuda", epochs=2, max_residues=4, resume=True)
            import json
            metrics = json.loads((run / "epoch_0002.json").read_text())
            self.assertGreater(metrics["peak_cuda_allocated_bytes"], 0)
            self.assertGreater(metrics["train_presentations_per_s"], 0)
            checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["contract"]["max_residues"], 4)


if __name__ == "__main__":
    unittest.main()
