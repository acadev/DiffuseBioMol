# Data preparation and schema

`examples/smoke-corpus` is ready to use. Prepare additional local structures
with the native AtomWorks parser and tokenizer:

```sh
diffusebiomol-prepare /path/to/coordinate-files /path/to/new-corpus
diffusebiomol-train /path/to/new-corpus runs/new-experiment --epochs 10
```

Use a new output directory. The default selects the largest canonical polymer
chain. PDB/mmCIF and gzip inputs are supported. See [parser policies](WORKFLOW.md)
for chain selection, missing atoms, alternate conformers, and residue numbering.
The checked-in vocabulary fixes categorical IDs, including virtual atom slots.

## Schema version 1

`manifest.json` records `schema_version=1`, `index_base=0`, vocabulary sizes,
the polymer vocabulary, entries and skipped-source diagnostics. Each entry has
`file`, `source`, `label`, `atoms` and the SHA-256 of its record bytes.

Each record is JSON with equal-length arrays:

- `element`, `modality`, `polymer`: zero-based categorical indices.
- `chain`, `residue`: original chain/residue identifiers encoded as integers.
- `virtual`: missing-coordinate flags.
- `xyz`: atom-major `[N,3]` coordinates; virtual placeholders are neutralized in loss preparation.

Records are checked against their checksums when loaded. Sources are split before
cropping. Crops contain up to `--max-residues` consecutive complete residues. The optional
`--max-atoms` limit is a memory guard: it may end a crop early but never cuts a
residue. Pair-position indices are computed after cropping. A crop with no
observed atoms or no residue fitting the atom limit raises an error. This baseline does not include spatial crops or
sequence-based clustering.

Do not merge independently prepared corpora without checking vocabulary and
parser metadata. Remap indices if necessary, or prepare a unified corpus. Source filenames do not detect biological
duplicates; deduplication and sequence clustering remain preparation tasks.

The manifest is loaded in memory and one JSON record is opened per source use.
This bounded-batch design is useful for a baseline but is not the final sharded
storage format for billion-presentation training.
