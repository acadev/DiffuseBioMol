# Julia reference utilities

These utilities require the environment and package from
[DiffuseBioMol.jl](https://github.com/acadev/DiffuseBioMol.jl). They do not turn this
Python repository into a Julia project.

```sh
JULIA_PROJECT=/path/to/DiffuseBioMol.jl
julia --project="$JULIA_PROJECT" tools/julia/export_python_corpus.jl \
  /path/to/pdb-or-cif-files /path/to/new-export
julia --project="$JULIA_PROJECT" tools/julia/export_python_parity.jl runs/julia_reference.json
python -m diffusebiomol.parity runs/julia_reference.json
```

Parity requires feature-only LayerNorm (`dims=1`) in the Julia network. This bug
was corrected in the Julia reference while developing the Python version; older
checkouts may still use `dims=:` and should be updated before regenerating parity
fixtures.

`tests/fixtures/julia_reference.json` is a self-contained numerical fixture from
the corrected Julia model, with deterministic random weights/inputs, nonzero
conditioning/output head, forward results and selected gradients. Python parity
tests need only this fixture. It contains no trained scientific checkpoint or
private data. Use Julia 1.12.6 and the reference project's dependency environment
for reproduction; Float32 comparison allows small backend numerical differences.
