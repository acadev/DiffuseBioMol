"""Single-device FP32 baseline. Run with python -m diffusebiomol.train --help."""
import argparse
import csv
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import Corpus, collate
from .flow import cfm_loss, prepare, sample_flow
from .model import FlowModel, ModelConfig


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def transfer(mapping, device):
    return {k: v.to(device) for k, v in mapping.items()}


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def start_wandb(project, entity, name, mode, manifest, root, run_id=None):
    """Initialize W&B only when requested; the default path has no W&B import."""
    if not project:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging requested; install it with `pip install -e '.[wandb]'`"
        ) from error
    return wandb.init(project=project, entity=entity or None, name=name or None,
        mode=mode, config=manifest, dir=str(root), id=run_id,
        resume="must" if run_id else "allow")


def train(corpus_dir, run_dir, *, epochs=3, batch_size=2, max_residues=16,
          max_atoms=None,
          seed=17, learning_rate=0.001, device="cpu", threads=2, resume=False,
          model_config=None, wandb_project=None, wandb_entity=None,
          wandb_name=None, wandb_mode="online", wandb_upload_checkpoint=False):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from .distributed import train_distributed
        return train_distributed(corpus_dir, run_dir, epochs=epochs, batch_size=batch_size,
            max_residues=max_residues, max_atoms=max_atoms, seed=seed,
            learning_rate=learning_rate, device=device, threads=threads, resume=resume,
            model_config=model_config, wandb_project=wandb_project,
            wandb_entity=wandb_entity, wandb_name=wandb_name,
            wandb_mode=wandb_mode, wandb_upload_checkpoint=wandb_upload_checkpoint)
    if min(epochs, batch_size, max_residues, threads) <= 0 or learning_rate <= 0 or (max_atoms is not None and max_atoms <= 0):
        raise ValueError("Epochs, batch size, crop limits, threads and learning rate must be positive")
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Use cpu, cuda, cuda:N or mps")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; refusing CPU fallback")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    corpus = Corpus(corpus_dir)
    if len(corpus.entries) < 3:
        raise ValueError("At least three sources are required")
    config = model_config or ModelConfig()
    contract = dict(schema=2, architecture="feature_norm_gelu_time_v1",
        objective="flow_matching", model=asdict(config),
        corpus=corpus.signature, batch_size=batch_size, max_residues=max_residues,
        max_atoms=max_atoms, seed=seed,
        learning_rate=learning_rate, device=str(dev), threads=threads)
    order = np.random.default_rng(seed).permutation(len(corpus.entries)).tolist()
    n_val = max(1, round(0.33 * len(order)))
    validation, training = order[:n_val], order[n_val:]
    root = Path(run_dir)
    if not resume and root.exists():
        raise ValueError("Choose a new run directory or pass --resume")
    if resume and not (root / "checkpoint.pt").is_file():
        raise ValueError("No checkpoint to resume")
    root.mkdir(parents=True, exist_ok=True)
    model = FlowModel(config, corpus.vocab).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start, presentations, updates = 1, 0, 0
    saved = None
    if resume:
        saved = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
        if saved["contract"] != contract:
            raise ValueError("Resume configuration or corpus differs from checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if dev.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if dev.type == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        start = saved["epoch"] + 1
        presentations, updates = saved["presentations"], saved["updates"]
        if epochs < saved["epoch"]:
            raise ValueError("Requested epoch precedes checkpoint")
    manifest = dict(contract, training=[corpus.entries[i] for i in training],
                    validation=[corpus.entries[i] for i in validation],
                    torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version())
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    tracker = start_wandb(wandb_project, wandb_entity, wandb_name, wandb_mode,
        manifest, root, saved.get("wandb_run_id") if saved else None)
    fields = ["epoch", "batch", "presentations", "loss", "crop_residues", "real_atoms",
              "padded_atoms", "padded_pair_elements", "load_prepare_s", "transfer_s",
              "forward_s", "backward_update_s"]
    profile = root / "steps.csv"
    # Remove uncommitted-epoch rows after a crash before restarting that epoch.
    if resume and profile.exists():
        with profile.open() as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < start]
        with profile.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    for epoch in range(start, epochs + 1):
        begun = time.perf_counter()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        model.train()
        sorted_indices = sorted(training, key=lambda i: min(corpus.entries[i]["atoms"],
            max_atoms if max_atoms is not None else corpus.entries[i]["atoms"]))
        batches = [sorted_indices[i:i+batch_size] for i in range(0, len(sorted_indices), batch_size)]
        rng.shuffle(batches)
        epoch_atoms = 0
        epoch_pair_elements = 0
        for number, indices in enumerate(batches, 1):
            t0 = time.perf_counter()
            records = [corpus.load(i, max_residues, rng, max_atoms) for i in indices]
            host = collate(records)
            example = prepare(host, rng)
            real_atoms = int(example["mask"].sum())
            crop_residues = sum(len(set(zip(r["chain"], r["residue"]))) for r in records)
            padded_atoms = int(host["valid"].numel())
            pair_elements = int(host["relpos"].numel())
            epoch_atoms += real_atoms
            epoch_pair_elements += pair_elements
            load_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            batch, example = transfer(host, dev), transfer(example, dev)
            synchronize(dev)
            transfer_s = time.perf_counter() - t0
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            loss = cfm_loss(model, batch, example)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            synchronize(dev)
            forward_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            loss.backward()
            # Detect invalid gradients before altering model/optimizer state.
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
            synchronize(dev)
            backward_s = time.perf_counter() - t0
            presentations += len(indices)
            updates += 1
            if tracker is not None:
                tracker.log({"training/cfm_loss": loss.item(),
                    "training/presentations": presentations,
                    "training/crop_residues": crop_residues,
                    "training/real_atoms": real_atoms,
                    "training/padded_atoms": padded_atoms,
                    "training/padded_pair_elements": pair_elements,
                    "timing/load_prepare_s": load_s, "timing/transfer_s": transfer_s,
                    "timing/forward_s": forward_s,
                    "timing/backward_update_s": backward_s}, step=updates)
            new = not profile.exists()
            with profile.open("a", newline="") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(fields)
                writer.writerow([epoch, number, presentations, loss.item(),
                    crop_residues, real_atoms, padded_atoms, pair_elements,
                    load_s, transfer_s, forward_s, backward_s])
        train_seconds = time.perf_counter() - begun
        # Persist training progress before evaluation. Validation has separate RNG.
        atomic_save(dict(contract=contract, model=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch, presentations=presentations, updates=updates,
            wandb_run_id=tracker.id if tracker is not None else None,
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if dev.type == "cuda" else [],
            mps_rng=torch.mps.get_rng_state() if dev.type == "mps" else torch.empty(0, dtype=torch.uint8)),
            root / "checkpoint.pt")
        metrics = evaluate(model, corpus, validation, max_residues, max_atoms, batch_size, seed)
        metrics.update(epoch=epoch, presentations=presentations, updates=updates,
            train_seconds=train_seconds,
            train_presentations_per_s=len(training) / train_seconds,
            train_real_atoms_per_s=epoch_atoms / train_seconds,
            train_padded_pair_elements_per_s=epoch_pair_elements / train_seconds)
        if dev.type == "cuda":
            metrics.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(dev),
                           peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(dev))
        (root / f"epoch_{epoch:04d}.json").write_text(json.dumps(metrics, indent=2))
        if tracker is not None:
            tracker.log({"epoch": epoch,
                "validation/cfm_loss": metrics["validation_cfm_loss"],
                "validation/finite_sample": int(metrics["finite_sample"]),
                "timing/epoch_training_s": train_seconds,
                "timing/evaluation_s": metrics["evaluation_seconds"],
                "throughput/presentations_per_s": metrics["train_presentations_per_s"],
                "throughput/real_atoms_per_s": metrics["train_real_atoms_per_s"],
                **({"memory/peak_cuda_allocated_bytes": metrics["peak_cuda_allocated_bytes"],
                    "memory/peak_cuda_reserved_bytes": metrics["peak_cuda_reserved_bytes"]}
                   if dev.type == "cuda" else {})}, step=updates)
            tracker.save(str(root / "manifest.json"), base_path=str(root))
            tracker.save(str(root / f"epoch_{epoch:04d}.json"), base_path=str(root))
            if wandb_upload_checkpoint:
                tracker.save(str(root / "checkpoint.pt"), base_path=str(root))
        print(json.dumps(metrics), flush=True)
    # Also supports retrying final evaluation if a previous evaluation crashed.
    final = evaluate(model, corpus, validation, max_residues, max_atoms, batch_size, seed)
    final.update(presentations=presentations, updates=updates, objective="flow_matching",
                 device=str(dev), protein_quality_established=False)
    (root / "summary.json").write_text(json.dumps(final, indent=2))
    if tracker is not None:
        tracker.summary.update(final)
        tracker.save(str(root / "steps.csv"), base_path=str(root))
        tracker.save(str(root / "summary.json"), base_path=str(root))
        tracker.finish()
    return model, final


@torch.no_grad()
def evaluate(model, corpus, indices, max_residues, max_atoms, batch_size, seed):
    model.eval()
    rng = np.random.default_rng(seed + 10000)
    dev = next(model.parameters()).device
    total, count = 0., 0
    begun = time.perf_counter()
    for i in range(0, len(indices), batch_size):
        host = collate([corpus.load(j, max_residues, rng, max_atoms)
                        for j in indices[i:i+batch_size]])
        example = prepare(host, rng)
        atoms = int(example["mask"].sum())
        loss = cfm_loss(model, transfer(host, dev), transfer(example, dev)).item()
        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        total += loss * atoms
        count += atoms
    sample = sample_flow(model, host, np.random.default_rng(seed + 20000), steps=3)
    if not torch.isfinite(sample).all():
        raise FloatingPointError("Non-finite sampled coordinates")
    return dict(validation_cfm_loss=total/count, finite_sample=True,
                evaluation_seconds=time.perf_counter()-begun)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir")
    parser.add_argument("run_dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-residues", type=int, default=16,
                        help="Maximum number of consecutive complete residues per crop")
    parser.add_argument("--max-atoms", type=int,
                        help="Optional hard atom safety limit; never splits a residue")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-config", help="JSON file containing ModelConfig fields")
    parser.add_argument("--wandb-project", help="Enable W&B logging for this project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-upload-checkpoint", action="store_true")
    options = vars(parser.parse_args())
    if options["model_config"]:
        options["model_config"] = ModelConfig(**json.loads(Path(options["model_config"]).read_text()))
    train(**options)


if __name__ == "__main__":
    main()
