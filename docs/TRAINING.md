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
  --epochs 10 --batch-size 2 --max-residues 16 --max-atoms 256 --seed 17 --device cpu
python -m diffusebiomol.train examples/smoke-corpus runs/experiment \
  --epochs 20 --batch-size 2 --max-residues 16 --max-atoms 256 --seed 17 --device cpu --resume
```

Keep the corpus, model configuration, batch size, residue/atom limits, seed, learning
rate, device and thread count unchanged on resume. Target epoch count can grow.
Checkpoints use an architecture identifier, corpus-manifest hash, and crop policy
to reject incompatible resumes. Runs created before residue-counted cropping
need a new directory; their old checkpoint contract cannot be resumed. Exact CPU resume was tested; cross-device resume is not
supported. After interruption, the last incomplete epoch is replayed and its
uncommitted timing rows are removed.

Use `--device cuda`, `cuda:1` or `mps` for an available accelerator. The default
is CPU. CUDA/MPS hardware tests are still outstanding. Install the appropriate
PyTorch build for the training host. AMP, gradient accumulation and asynchronous prefetch are not implemented.

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
| `steps.csv` | Loss, cumulative presentations, crop residues, real/padded atoms, padded pair elements, phase timings |
| `checkpoint.pt` | Model, optimizer, RNG states, completed epoch, update/presentation counters |
| `epoch_*.json` | Fixed-validation CFM loss, training/evaluation durations, finite-sample check |
| `summary.json` | Final evaluation and cumulative counters |

Checkpoint writes are atomic and occur before evaluation. Validation randomness
is independent of training randomness, with fixed crops/prior/time draws across
epochs. Validation loss is normalized over observed coordinate components;
virtual and batch padding positions are excluded. The sampler's three steps only
check finite outputs, not scientific quality. Validation runs on the selected
model device and is batched.

Epoch metrics include training presentations/second, observed atoms/second,
padded pair elements/second, and peak allocated/reserved CUDA bytes. GPU metrics
require a CUDA run and exclude evaluation memory; evaluation has a separate time.

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

## One GPU pilot

On a cluster GPU node, install a CUDA-enabled PyTorch build compatible with the
cluster driver, then run the CUDA-only checks and a bounded pilot. For example:

```sh
python -m unittest discover -s tests -p test_cuda.py -v
diffusebiomol-train /path/to/prepared-corpus runs/gpu-pilot \
  --device cuda --model-config configs/gpu_pilot.json \
  --max-residues 16 --max-atoms 384 --batch-size 2 --epochs 3
```

Use new run directories. Inspect `epoch_*.json` for loss, speed and peak GPU
memory, and `steps.csv` for padding and stage costs. Repeat at 32 and 64 residues
only after the preceding run fits and finishes. The optional atom limit is a
resource guard; report both limits for comparisons. These commands use one GPU.
`configs/gpu_pilot.json` is a roughly two-million-parameter measurement model;
it has not been validated on a GPU yet. Start with the smaller
`configs/small.json` if the pilot model cannot complete the correctness check.
Distributed training is available through `torchrun`; see below.

## Distributed training

Use `torchrun` with one process per GPU. On one node with four visible GPUs:

```sh
torchrun --standalone --nnodes=1 --nproc-per-node=4 -m diffusebiomol.train \
  /path/to/prepared-corpus runs/ddp-pilot --device cuda \
  --model-config configs/gpu_pilot.json --max-residues 16 \
  --max-atoms 384 --batch-size 2 --epochs 3
```

`--batch-size` is **per rank**, so the maximum global batch here is eight
sources. `torchrun` sets `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`; each local rank
uses its corresponding visible CUDA device. The model uses NCCL on CUDA and Gloo
on CPU. A CPU verification run can use `--device cpu` and `--nproc-per-node=3`.

Every source in the training split is assigned once per epoch. The final global
batch can be smaller; ranks without a real source perform a zero-weight forward
pass to participate in gradient synchronization. Losses are weighted by the
global number of observed atoms, and all ranks use the same optimizer update.
`presentations` counts actual sources, not synchronization placeholders.

Rank zero alone writes checkpoints, metrics and W&B logs. Checkpoints contain
model and optimizer state plus independent crop/CPU/CUDA RNG state for every
rank. An interrupted epoch is replayed. Resume requires the same corpus, model,
world size, per-rank batch size, crop limits, seed, device type and thread count;
only the target epoch count may grow. All nodes must see the same input corpus
and run directory through shared storage. Single-process and DDP checkpoints
have different contracts and cannot be interchanged.

For multiple nodes, launch one `torchrun` agent per node using the same
`--nnodes`, `--nproc-per-node`, `--rdzv-id`, and `--rdzv-endpoint` values; set
`--node-rank` appropriately. The scheduler must provide network reachability
between nodes and a shared run directory. See the
[PyTorch torchrun documentation](https://docs.pytorch.org/docs/2.14/elastic/run.html)
for rendezvous options.

Epoch metrics report aggregate presentations and observed atoms per second,
and the maximum peak CUDA memory across ranks. Stage timings in `steps.csv`
are the slowest rank at each step. Evaluation and sampling run on rank zero
after all ranks finish training. This design was checked with three CPU ranks,
including an uneven final batch and exact epoch-boundary resume. Multi-GPU
NCCL performance and restart behavior still require a cluster run.
