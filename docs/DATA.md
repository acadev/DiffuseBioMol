# Data preparation and schema

## PDB-scale fold-diverse protein corpus

Use the [CATH classification release](https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/)
with a matching [nonredundant domain-coordinate set](https://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/).
Download `cath-domain-list.txt`, one `cath-dataset-nonredundant-S20` or `S40`
`.list`, and its matching `.pdb.tgz` from the same CATH release. Pin the release
URL and retain these input files. Extract the archive on cluster scratch storage;
the input directory must contain the **CATH-chopped domain PDBs**, not whole-entry
PDB structures. CATH's archive members are extensionless files such as
`dompdb/12asA00`; the preparer recognizes these names and reads them as PDB.
For S20:

```sh
mkdir -p /scratch/cath-s20-domains
tar -xzf /scratch/cath-dataset-nonredundant-S20.pdb.tgz -C /scratch/cath-s20-domains
diffusebiomol-prepare-folds /scratch/cath-s20-domains \
  /scratch/cath-domain-list.txt /scratch/dbm-cath-s20 \
  --selection /scratch/cath-dataset-nonredundant-S20.list \
  --workers 16 --strict
```

`--strict` fails for missing coordinates or parse/filter skips, with a manifest
for diagnostics in the latter case. Omit it to build an explicitly incomplete
corpus and inspect `missing_coordinates` and `skipped`. The worker queue is
bounded at twice the worker count. Classification and selection are read as
linear files; no pairwise structure comparisons are performed. Benchmark a
small extracted subset before choosing worker count and storage layout. The
current schema writes one JSON file per domain, so filesystem metadata traffic
and AtomWorks parsing should be measured before processing the full collection.
If you use `--min-residues` or `--max-residues` to filter domains, omit
`--strict`: genuine S20 domains fall outside a 40–512-residue range, so the
filtered corpus is intentionally incomplete. Use a new output directory for
each preparation attempt.

Each domain entry includes `domain_id`, CATH topology (`fold_id`, C.A.T),
homologous superfamily (`superfamily_id`, C.A.T.H), S35 family, resolution,
source path, observed residue count, and record checksum. The manifest records
classification/selection SHA-256 values and a fold histogram. The S20/S40
selection is already nonredundant at the stated sequence-identity threshold;
it does not by itself make a homology-disjoint validation split.

Train with a superfamily holdout and uniform fold draws:

```sh
diffusebiomol-train /scratch/dbm-cath-s20 runs/cath-s20 \
  --split-group superfamily --fold-sampling uniform_fold \
  --presentations-per-epoch 20000 --max-residues 256 \
  --max-atoms 8192 --batch-size 1 --device cuda --epochs 10
```

For multiple GPUs, replace `diffusebiomol-train` with `torchrun ... -m
diffusebiomol.train` as in [distributed training](TRAINING.md). Uniform fold
sampling draws a fold, then a domain in that fold, both uniformly **with
replacement**; presentations may repeat within an epoch. Validation is the
fixed held-out superfamily set and is not oversampled. `--split-group fold`
tests unseen CATH topologies instead, but this is a harder and different
generalization question. Fold labels guide sampling only: the current model
does not yet consume a fold or sequence-similarity conditioning signal.

`examples/smoke-corpus` is ready to use. Prepare additional local structures
with the native AtomWorks parser and tokenizer:

```sh
diffusebiomol-prepare /path/to/coordinate-files /path/to/new-corpus --workers 8
diffusebiomol-train /path/to/new-corpus runs/new-experiment --epochs 10
```

Use a new output directory. The default creates one record per canonical polymer
chain, so multi-chain entries no longer silently lose all but their largest chain.
Use `--chain largest` for the old policy, `--chain all` to keep the full asymmetric
unit in one record, or `--chain A` for one parser chain ID. Multiple parser workers
process different files concurrently and the manifest remains in source/chain order.
Benchmark a representative sample before choosing a worker count; each worker
needs memory for an entire parsed structure. PDB/mmCIF and gzip inputs are supported.
See [parser policies](WORKFLOW.md)
for chain selection, missing atoms, alternate conformers, and residue numbering.
The checked-in vocabulary fixes categorical IDs, including virtual atom slots.

## Schema version 1

`manifest.json` records `schema_version=1`, `index_base=0`, vocabulary sizes,
the polymer vocabulary, entries and skipped-source diagnostics. Each entry has
`file`, `source`, `label`, `chain`, `entity_keys`, `atoms` and the SHA-256 of its
record bytes. `entity_keys` are PDB ID plus mmCIF polymer entity ID (for example,
`1ABC_1`). Nonstandard filenames or sources without entity annotations have no
keys and cannot use the strict RCSB cluster split.

Each record is JSON with equal-length arrays:

- `element`, `modality`, `polymer`: zero-based categorical indices.
- `chain`, `residue`: original chain/residue identifiers encoded as integers.
- `virtual`: missing-coordinate flags.
- `xyz`: atom-major `[N,3]` coordinates; virtual placeholders are neutralized in loss preparation.

Records are checked against their checksums when loaded. Records are split before
cropping. Crops contain up to `--max-residues` consecutive complete residues. The optional
`--max-atoms` limit is a memory guard: it may end a crop early but never cuts a
residue. Pair-position indices are computed after cropping. A crop with no
observed atoms or no residue fitting the atom limit raises an error. Spatial crops
are not implemented.

Without a cluster file, all records from the same source file stay in the same
split. For protein corpora, pass `--sequence-clusters` to training with the
[RCSB clusters-by-entity file](https://www.rcsb.org/docs/programmatic-access/file-download-services#sequence-clusters-data)
(for example, the 30% identity file). All entities in one record and all records
sharing a cluster are kept in the same split, even transitively. Missing entity
keys or cluster coverage cause an error rather than a leaky split. RCSB's protein
clusters do not cover nucleic-acid-only records; filter those out or prepare a
separate clustering scheme before making a scientific holdout for them.

Validation is deterministic, defaults to roughly 10% of records, and is capped
at 1,024 records evaluated on rank zero per epoch. Change this with
`--validation-fraction` and `--max-validation-sources`. The split method, cluster
file SHA-256, and limits are part of the checkpoint contract; a changed cluster
file requires a new run.

Do not merge independently prepared corpora without checking vocabulary and
parser metadata. Remap indices if necessary, or prepare a unified corpus. Source filenames do not detect biological
duplicates; representative selection and other corpus-quality filters remain
preparation tasks.

The manifest is loaded in memory and one JSON record is opened per source use.
This bounded-batch design is useful for a baseline but is not the final sharded
storage format for billion-presentation training.
