"""Synchronous data-parallel training launched with torchrun."""
import csv
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .data import Corpus, collate
from .flow import cfm_loss, prepare
from .model import FlowModel, ModelConfig
from .train import atomic_save, evaluate, start_wandb, synchronize, transfer


def rank_batches(training, corpus, batch_size, world_size, rank, epoch, seed):
    """Partition a seeded source order without repeats or dropped sources."""
    order = sorted(training, key=lambda i: corpus.entries[i]["atoms"])
    batches = [order[i:i + batch_size * world_size]
               for i in range(0, len(order), batch_size * world_size)]
    np.random.default_rng(seed + epoch).shuffle(batches)
    return [batch[rank * batch_size:(rank + 1) * batch_size] for batch in batches]


def rank_state(rng, device):
    return dict(numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rank_state(state, rng, device):
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)


def train_distributed(corpus_dir, run_dir, *, epochs, batch_size, max_residues,
        max_atoms, seed, learning_rate, device, threads, resume, model_config,
        wandb_project, wandb_entity, wandb_name, wandb_mode, wandb_upload_checkpoint):
    if min(epochs, batch_size, max_residues, threads) <= 0 or learning_rate <= 0 or (max_atoms is not None and max_atoms <= 0):
        raise ValueError("Epochs, batch size, crop limits, threads and learning rate must be positive")
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    requested = torch.device(device)
    if requested.type == "cuda":
        if requested.index is not None:
            raise ValueError("Use --device cuda with torchrun; LOCAL_RANK selects the GPU")
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError("Each local rank requires its own visible CUDA device")
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
        backend = "nccl"
    elif requested.type == "cpu":
        dev, backend = torch.device("cpu"), "gloo"
    else:
        raise ValueError("Distributed training supports CPU or CUDA")
    torch.set_num_threads(threads)
    dist.init_process_group(backend=backend, init_method="env://")
    try:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed + rank * 1000003)
        corpus = Corpus(corpus_dir)
        if len(corpus.entries) < 3:
            raise ValueError("At least three sources are required")
        config = model_config or ModelConfig()
        contract = dict(schema=3, architecture="feature_norm_gelu_time_v1",
            objective="flow_matching", model=asdict(config), corpus=corpus.signature,
            batch_size=batch_size, max_residues=max_residues, max_atoms=max_atoms,
            seed=seed, learning_rate=learning_rate, device=requested.type,
            threads=threads, world_size=world_size)
        order = np.random.default_rng(seed).permutation(len(corpus.entries)).tolist()
        n_val = max(1, round(0.33 * len(order)))
        validation, training = order[:n_val], order[n_val:]
        root = Path(run_dir)
        exists = torch.tensor([int(root.exists()), int((root / "checkpoint.pt").is_file())]
            if rank == 0 else [0, 0], dtype=torch.long, device=dev)
        dist.broadcast(exists, src=0)
        if not resume and exists[0].item():
            raise ValueError("Choose a new run directory or pass --resume")
        if resume and not exists[1].item():
            raise ValueError("No checkpoint to resume")
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        model = FlowModel(config, corpus.vocab).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        start, presentations, updates = 1, 0, 0
        saved = None
        if resume:
            saved = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
            if saved["contract"] != contract:
                raise ValueError("Resume configuration, world size, or corpus differs from checkpoint")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start = saved["epoch"] + 1
            presentations, updates = saved["presentations"], saved["updates"]
            if epochs < saved["epoch"]:
                raise ValueError("Requested epoch precedes checkpoint")
        wrapped = DistributedDataParallel(model,
            device_ids=[local_rank] if dev.type == "cuda" else None)
        if saved is not None:
            restore_rank_state(saved["rank_states"][rank], rng, dev)
        manifest = dict(contract, training=[corpus.entries[i] for i in training],
            validation=[corpus.entries[i] for i in validation],
            torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version())
        tracker = None
        fields = ["epoch", "batch", "presentations", "loss", "crop_residues",
            "real_atoms", "padded_atoms", "padded_pair_elements", "load_prepare_s",
            "transfer_s", "forward_s", "backward_update_s"]
        profile = root / "steps.csv"
        if rank == 0:
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
            tracker = start_wandb(wandb_project, wandb_entity, wandb_name,
                wandb_mode, manifest, root, saved.get("wandb_run_id") if saved else None)
            if resume and profile.exists():
                with profile.open() as f:
                    rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < start]
                with profile.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
        dist.barrier()
        for epoch in range(start, epochs + 1):
            begun = time.perf_counter()
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            wrapped.train()
            batches = rank_batches(training, corpus, batch_size, world_size, rank, epoch, seed)
            epoch_atoms = epoch_pairs = epoch_presentations = 0
            for number, indices in enumerate(batches, 1):
                t0 = time.perf_counter()
                # Empty final shard still participates in DDP's forward/backward.
                records = [corpus.load(i, max_residues, rng, max_atoms)
                           for i in (indices or [training[0]])]
                host = collate(records)
                example = prepare(host, rng)
                local_atoms = int(example["mask"].sum()) if indices else 0
                crop_residues = sum(len(set(zip(r["chain"], r["residue"])))
                                    for r in records) if indices else 0
                padded_atoms = int(host["valid"].numel()) if indices else 0
                pair_elements = int(host["relpos"].numel()) if indices else 0
                load_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                batch, example = transfer(host, dev), transfer(example, dev)
                synchronize(dev)
                transfer_s = time.perf_counter() - t0
                counts = torch.tensor([local_atoms, len(indices), crop_residues,
                    padded_atoms, pair_elements], dtype=torch.long, device=dev)
                dist.all_reduce(counts)
                total_atoms = int(counts[0].item())
                if total_atoms == 0:
                    raise ValueError("A global batch has no observed atoms")
                optimizer.zero_grad(set_to_none=True)
                t0 = time.perf_counter()
                local_loss = cfm_loss(wrapped, batch, example)
                if not torch.isfinite(local_loss):
                    raise FloatingPointError("Non-finite training loss")
                synchronize(dev)
                forward_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                (local_loss * (local_atoms * world_size / total_atoms)).backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError("Non-finite gradient")
                optimizer.step()
                synchronize(dev)
                backward_s = time.perf_counter() - t0
                loss_sum = local_loss.detach() * local_atoms
                dist.all_reduce(loss_sum)
                global_loss = float((loss_sum / total_atoms).item())
                times = torch.tensor([load_s, transfer_s, forward_s, backward_s], device=dev)
                dist.all_reduce(times, op=dist.ReduceOp.MAX)
                presentations += int(counts[1].item())
                updates += 1
                epoch_presentations += int(counts[1].item())
                epoch_atoms += total_atoms
                epoch_pairs += int(counts[4].item())
                if rank == 0:
                    row = [epoch, number, presentations, global_loss,
                        int(counts[2].item()), total_atoms, int(counts[3].item()),
                        int(counts[4].item()), *times.tolist()]
                    with profile.open("a", newline="") as f:
                        writer = csv.writer(f)
                        if f.tell() == 0:
                            writer.writerow(fields)
                        writer.writerow(row)
                    if tracker is not None:
                        tracker.log({"training/cfm_loss": global_loss,
                            "training/presentations": presentations,
                            "training/crop_residues": int(counts[2].item()),
                            "training/real_atoms": total_atoms,
                            "training/padded_atoms": int(counts[3].item()),
                            "training/padded_pair_elements": int(counts[4].item()),
                            "timing/load_prepare_s": float(times[0]),
                            "timing/transfer_s": float(times[1]),
                            "timing/forward_s": float(times[2]),
                            "timing/backward_update_s": float(times[3])}, step=updates)
            dist.barrier()
            train_seconds = time.perf_counter() - begun
            elapsed = torch.tensor(train_seconds, device=dev)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            train_seconds = float(elapsed.item())
            states = [None] * world_size
            dist.all_gather_object(states, rank_state(rng, dev), weights_only=True)
            peak = torch.tensor([torch.cuda.max_memory_allocated(dev),
                torch.cuda.max_memory_reserved(dev)] if dev.type == "cuda" else [0, 0],
                dtype=torch.long, device=dev)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                atomic_save(dict(contract=contract, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), rank_states=states, epoch=epoch,
                    presentations=presentations, updates=updates,
                    wandb_run_id=tracker.id if tracker is not None else None),
                    root / "checkpoint.pt")
                metrics = evaluate(model, corpus, validation, max_residues,
                    max_atoms, batch_size, seed)
                metrics.update(epoch=epoch, presentations=presentations,
                    updates=updates, train_seconds=train_seconds,
                    train_presentations_per_s=epoch_presentations / train_seconds,
                    train_real_atoms_per_s=epoch_atoms / train_seconds,
                    train_padded_pair_elements_per_s=epoch_pairs / train_seconds,
                    world_size=world_size)
                if dev.type == "cuda":
                    metrics.update(peak_cuda_allocated_bytes=int(peak[0]),
                        peak_cuda_reserved_bytes=int(peak[1]))
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
                           if dev.type == "cuda" else {})},
                        step=updates)
                    tracker.save(str(root / "manifest.json"), base_path=str(root))
                    tracker.save(str(root / f"epoch_{epoch:04d}.json"), base_path=str(root))
                    if wandb_upload_checkpoint:
                        tracker.save(str(root / "checkpoint.pt"), base_path=str(root))
                print(json.dumps(metrics), flush=True)
            dist.barrier()
        if rank == 0:
            final = evaluate(model, corpus, validation, max_residues,
                max_atoms, batch_size, seed)
            final.update(presentations=presentations, updates=updates,
                objective="flow_matching", device=requested.type,
                world_size=world_size, protein_quality_established=False)
            (root / "summary.json").write_text(json.dumps(final, indent=2))
            if tracker is not None:
                tracker.summary.update(final)
                tracker.save(str(profile), base_path=str(root))
                tracker.save(str(root / "summary.json"), base_path=str(root))
                tracker.finish()
        dist.barrier()
        return (model, final) if rank == 0 else (None, None)
    finally:
        dist.destroy_process_group()
