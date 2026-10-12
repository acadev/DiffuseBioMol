# Curated domain similarity pilot

This is a deliberately small **proof of measurement**, not a representative
fold benchmark. Six public PDB coordinate files are bundled as compressed mmCIF,
with original entry links, chain/range selections, fold groups and byte-level
checksums in [curated.json](curated.json). They were downloaded from RCSB PDB on
2026-10-09. PDB archive data are [CC0](https://www.rcsb.org/pages/policies);
cite the original structure publications linked from each entry page in
scientific use.

The three groups are ubiquitin-like beta-grasp (1UBQ, SUMO-1 1A5R, SUMO-3
1U4A), protein G B1 (2GB1, 1PGA), and crambin (1CRN) as an outgroup. SUMO-1
uses modeled residues 21–97 to omit its flexible N-terminal tail. This range is
a pilot selection, not an imported CATH domain boundary. The [SUMO-1 PDB record](https://www.rcsb.org/structure/1A5R)
describes its shared ubiquitin fold despite 18% overall sequence identity.
Every structure uses its first model and chain A. The 1U4A entry is an NMR
ensemble; the parser uses its first conformer.

From the repository root with the Python package installed:

```sh
python -m diffusebiomol.fold_pilot map \
  examples/fold-pilot/curated.json examples/fold-pilot/similarity_map.json
python -m diffusebiomol.fold_pilot train \
  examples/fold-pilot/curated.json examples/fold-pilot/similarity_map.json \
  runs/fold-pilot --epochs 30
```

The map aligns sequences with BLOSUM62 and compares structures with Biotite's
3Di structural-homolog alignment. TM-scores use the longer domain for
normalization, which penalizes partial matches. The checked-in
[similarity_map.json](similarity_map.json) has four same-group and eleven
cross-group pairs. Their sequence-identity AUROC is **0.591**; structural
TM-score AUROC is **1.000**. The least similar same-group pair has TM-score
**0.636**, above the most similar cross-group pair at **0.451**. This establishes
the intended signal on these *selected* examples only; it does not estimate
performance on uncurated PDB domains.

The training command holds out 1A5R and 1PGA, then compares the original flow
model with a version given four context values: TM-scores to the three landmark
structures (1UBQ, 2GB1, 1CRN) plus a zero padding channel. It uses the
existing four-feature conditioning layer, 16-residue crops, the same seed and
the same training schedule for both variants. The holdout's true structure is
used to calculate its context vector: this is **oracle fold conditioning**, not
a sequence-only predictor. Results from a 30-epoch CPU run are preserved in
[generation_pilot.json](generation_pilot.json). Both variants produced finite
three-step samples; mean held-out CFM loss was **247.58** without context and
**247.93** with context. There is no observed improvement on this pilot.

The next scientific step is a larger, domain-level training set with independent
fold and sequence holdouts, then a sequence encoder trained to predict fold
context without access to the target structure. Full-domain generation also
needs a more memory-efficient residue-level representation before all-atom
refinement; the present dense atom-pair tensor scales quadratically.
