# Scope and scaling roadmap

The scale target is **one billion crop/sample presentations**, potentially
revisiting sources. The eventual scientific target includes full protein
structures, which is separate from learning local crop geometry.

## Current baseline

- Pairformer-lite encoder, dense pair biases and DiT velocity decoder.
- Linear-path conditional flow-matching mathematics; unconditional training CLI.
- Polymer random-walk prior, Haar rotations and centroid removal.
- Lazy source reads, residue-complete sequence crops and padded batches.
- Source-disjoint validation, atomic checkpointing, exact CPU resume and timing.
- Julia/PyTorch numerical reference checks and real-crop learning tests.

The model does not have built-in rotational equivariance. Coordinate augmentation
does not provide an exact equivariance guarantee. Full atom-pair features remain
quadratic in crop length, even with fused attention.

## Next gates

1. Reproduce correctness and sustained throughput on the target GPU, separating
   warm compute from loading, compilation/startup, validation and checkpointing.
2. Scale to representative source counts and crop lengths; profile memory,
   padding, gradient checks and input wait. Add compact shards and bounded prefetch.
3. Establish clustered held-out quality and external structural validation.
4. Add and independently test diffusion schedules, prediction targets, objectives
   and samplers while retaining the working flow-matching path.
5. Port motif clamping, CFG, geometry guidance and trained/calibrated verification.
6. Add mixed precision, multi-GPU data parallelism and distributed restart tests.
7. Demonstrate full-structure generation with global context and long-range
   constraints; local crops cannot simply be pasted together to establish topology.
8. Add accelerated sampling and generate/verify/curate/retrain only after quality,
   provenance, replay and failure-detection gates are reliable.

One billion presentations in fourteen days requires 827/s sustained aggregate
throughput, or 1,033/s while training if only 80% of wall time is available.
These are capacity requirements, not performance claims. Measure the actual crop
distribution and hardware before estimating how many devices or days are needed.
