# Training and evaluation

Install from the repository root with `python -m pip install -e .`.
`python -m diffusebiomol.train --help` lists options; `diffusebiomol-train` is an
equivalent installed command.

The default model uses one Pairformer block, one DiT block, single width 16, pair
width 8 and four heads. `configs/small.json` selects a larger two-block model.
Run directories must be new unless `--resume` is specified.

## Run and resume

```sh
python -m diffusebiomol.train examples/smoke-corpus runs/experiment \
  --epochs 10 --batch-size 2 --max-atoms 128 --seed 17 --device cpu
python -m diffusebiomol.train examples/smoke-corpus runs/experiment \
  --epochs 20 --batch-size 2 --max-atoms 128 --seed 17 --device cpu --resume
```

Keep the corpus, model configuration, batch size, crop budget, seed, learning
rate, device and thread count unchanged on resume. Target epoch count can grow.
Checkpoints use an architecture identifier and corpus-manifest hash to reject
incompatible resumes. Exact CPU resume was tested; cross-device resume is not
supported. After interruption, the last incomplete epoch is replayed and its
uncommitted timing rows are removed.

Use `--device cuda`, `cuda:1` or `mps` for an available accelerator. The default
is CPU. CUDA/MPS hardware tests are still outstanding. Install the appropriate
PyTorch build for the training host. This baseline does not implement DDP, AMP,
gradient accumulation or asynchronous prefetch.

## Weights & Biases

W&B is an optional extra; normal training has no W&B import or tracking overhead.

```sh
python -m pip install -e '.[wandb]'
export WANDB_API_KEY='...'
python -m diffusebiomol.train examples/smoke-corpus runs/tracked \
  --epochs 10 --wandb-project DiffuseBioMol --wandb-entity YOUR_TEAM \
  --wandb-name baseline
```

The integration logs training CFM loss, cumulative presentations, real/padded
atoms, stage timings, validation CFM loss, finite-sample status and epoch times.
It saves the manifest, epoch metrics, steps CSV and summary. Checkpoint upload is
opt-in with `--wandb-upload-checkpoint` because model files can be large. The W&B
run ID is stored in the checkpoint and reused by online resumes. W&B deliberately
ignores resume semantics in offline mode and starts another local run.
`--wandb-mode offline` records locally without network access; `disabled`
exercises the SDK's disabled mode. Credentials remain in W&B's normal
environment/configuration path.

## Artifacts

| File | Contents |
|---|---|
| `manifest.json` | Source split, model/configuration, corpus hash and software versions |
| `steps.csv` | Loss, cumulative crop presentations, real/padded atom counts, phase timings |
| `checkpoint.pt` | Model, optimizer, RNG states, completed epoch, update/presentation counters |
| `epoch_*.json` | Fixed-validation CFM loss, training/evaluation durations, finite-sample check |
| `summary.json` | Final evaluation and cumulative counters |

Checkpoint writes are atomic and occur before evaluation. Validation randomness
is independent of training randomness, with fixed crops/prior/time draws across
epochs. Validation loss is normalized over observed coordinate components;
virtual and batch padding positions are excluded. The sampler's three steps only
check finite outputs, not scientific quality. Validation runs on the selected
model device and is batched.

Per-step timings distinguish host loading/preparation, transfer, forward and
backward/optimizer work. Accelerator boundaries are synchronized for diagnostic
accuracy; finite-gradient checks may synchronize too. Per-epoch training times
exclude imports, initialization, checkpoint serialization and evaluation. They
are not an overlapped throughput benchmark or a billion-presentation forecast.

## Interpreting results

A decreasing fixed-example loss demonstrates learnability; it does not establish
generalization. The six-source example is not sequence-clustered. Before scientific
comparisons, build a clustered holdout and evaluate geometry, long-range contacts
and external refolding/designability. Count repeated crops as presentations, not
new independent structures.
