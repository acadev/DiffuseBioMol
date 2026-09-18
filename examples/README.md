# Example corpus provenance

The smoke corpus contains largest-chain atom-token exports of six public PDB
entries, fetched as mmCIF from RCSB PDB on 2026-09-17:

| Entry | Source and original citation information |
|---|---|
| 1CRN | https://www.rcsb.org/structure/1CRN |
| 1L2Y | https://www.rcsb.org/structure/1L2Y |
| 1UBQ | https://www.rcsb.org/structure/1UBQ |
| 1VII | https://www.rcsb.org/structure/1VII |
| 2GB1 | https://www.rcsb.org/structure/2GB1 |
| 5PTI | https://www.rcsb.org/structure/5PTI |

The records were produced with the DiffuseBioMol.jl parser/tokenizer, retain its
virtual atom slots, and are not complete original mmCIF files. The manifest
preserves source URLs, vocabulary metadata and per-record checksums. Personal
filesystem paths, caches and training checkpoints are not included.

PDB archive data are available under CC0 1.0, as described by the
[RCSB PDB data usage policy](https://www.rcsb.org/pages/policies). Cite the original
structure publications, linked from the entry pages, for scientific use.
RCSB PDB reference: Berman et al., *The Protein Data Bank* (2000),
[Nucleic Acids Research 28:235–242](https://doi.org/10.1093/nar/28.1.235).

This small corpus is for software checks only. It is not a clustered benchmark,
representative training dataset, or evidence of model quality.
