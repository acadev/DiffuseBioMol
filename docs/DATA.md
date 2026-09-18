# Data preparation and schema

`examples/smoke-corpus` is ready to use. For more structures, use the existing
Julia parser/tokenizer to export a local PDB/mmCIF directory:

```sh
# JULIA_PROJECT must be a prepared DiffuseBioMol.jl checkout.
JULIA_PROJECT=/path/to/DiffuseBioMol.jl
julia --project="$JULIA_PROJECT" tools/julia/export_python_corpus.jl \
  /path/to/coordinate-files /path/to/new-export
python -m diffusebiomol.train /path/to/new-export runs/new-experiment --epochs 10
```

The export directory must be new. The exporter selects the largest chain, retains
the reference atom vocabulary and virtual atoms, reports skipped sources and
stores only linear-size arrays. Python requires no Julia process during training.

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
cropping. Crops include complete consecutive residues; pair-position indices are
computed after cropping. A crop with no observed atoms or no residue fitting the
budget raises an error. This baseline does not include spatial crops or
sequence-based clustering.

Do not merge independently exported corpora by concatenating manifests: Julia
vocabulary enumeration may differ between exports. Remap indices using explicit
vocabularies or export a unified corpus. Source filenames do not detect biological
duplicates; deduplication and sequence clustering remain preparation tasks.

The manifest is loaded in memory and one JSON record is opened per source use.
This bounded-batch design is useful for a baseline but is not the final sharded
storage format for billion-presentation training.
