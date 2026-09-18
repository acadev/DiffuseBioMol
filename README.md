# DiffuseBioMol

A PyTorch all-atom protein **flow-matching training baseline**, developed alongside
[DiffuseBioMol.jl](https://github.com/acadev/DiffuseBioMol.jl). It combines a
Pairformer-lite encoder, DiT decoder, polymer prior and Euler sampler.

This repository is independently installable. Six small real-structure token
records are included so the training example and tests work without Julia or
network downloads after installation. Julia is only needed to export additional
PDB/mmCIF corpora using the reference tokenizer.

## Quick start

Python 3.12 is the tested environment. From a fresh clone:

```sh
git clone https://github.com/acadev/DiffuseBioMol.git
cd DiffuseBioMol
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m diffusebiomol.train examples/smoke-corpus runs/smoke --epochs 3
python -m diffusebiomol.train examples/smoke-corpus runs/smoke --epochs 5 --resume
```

W&B remains optional and is never imported unless enabled:

```sh
python -m pip install -e '.[wandb]'
export WANDB_API_KEY='...'
python -m diffusebiomol.train examples/smoke-corpus runs/wandb-smoke \
  --epochs 3 --wandb-project DiffuseBioMol --wandb-name smoke
```

Use `--wandb-mode offline` for a network-free tracked run and
`--wandb-upload-checkpoint` to upload the latest checkpoint. Resume reuses the
stored W&B run ID in online mode. W&B ignores resume semantics in offline mode
and starts another local run. API keys are read by W&B and are never written to
manifests.

Use a **new output directory** unless resuming. The example trains four sources
and holds out two, using residue-complete crops capped at 128 atom tokens. It is a
pipeline check, not evidence of useful protein design or full-protein generation.

The trainer writes a checkpoint, source/configuration manifest, per-step timing
CSV, epoch validation metrics and a final summary. Resume restores the model,
optimizer, RNG states and presentation/update counts at epoch boundaries.

## Validate the implementation

```sh
python -m unittest discover -s tests -v
DBM_CORPUS=examples/smoke-corpus python -m unittest discover -s tests -v
python -m diffusebiomol.parity tests/fixtures/julia_reference.json
```

Tests cover real-crop loss reduction, masking, padding/output/gradient invariance,
residue-complete crops, corpus integrity, configuration checks and exact CPU
checkpoint resume. The bundled numerical fixture compares identical Julia and
PyTorch weights/inputs, forward outputs and selected gradients; maximum absolute
error was below **4e-7**. [Measured results](docs/BASELINE_RESULTS.md).

## Larger training experiments

```sh
python -m diffusebiomol.train /path/to/exported-corpus runs/larger \
  --model-config configs/small.json --max-atoms 512 --batch-size 2 \
  --epochs 10 --device cuda
```

CPU has been tested. CUDA and Apple MPS device selection are implemented but not
yet validated on accelerator hardware; unavailable requested devices fail instead
of falling back. The dense pair representation remains quadratic: increase crop
and batch sizes only after measuring memory and throughput.

- [Training, outputs, resume and timing](docs/TRAINING.md)
- [Preparing data and corpus schema](docs/DATA.md)
- [Architecture, scope and scaling roadmap](docs/ROADMAP.md)
- [Julia export and parity regeneration](tools/julia/README.md)
- [Example data provenance](examples/README.md)

## Status and provenance

Implemented: single-device FP32, unconditional linear-path flow matching, random
rotation/centering, source-disjoint validation and finite sampling checks. The
backbone accepts four conditioning features, but this runner does not yet implement
motif clamping, CFG, geometry guidance or verifier training.

**Diffusion is a planned separate objective/schedule/sampler**, not a name for the
current flow objective. Distributed training, mixed precision, full-structure
generation and billion-presentation capacity remain future milestones.

The implementation derives from this project's Julia reference, inspired by
published biomolecular modeling ideas. It is not an official RFdiffusion3 or
NeuralPLexer implementation and does not ship pretrained scientific weights.

Code is [MIT licensed](LICENSE). Example PDB-derived data provenance and the
archive's CC0 policy are described in [examples/README.md](examples/README.md).
