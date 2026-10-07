"""Build a training corpus directly from local PDB/mmCIF files using AtomWorks."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

from .tokenizer import SLOTS, VOCABULARY, VOCAB_SIZES, parse_structure, tokenize

EXTENSIONS = (".pdb", ".ent", ".cif", ".mmcif")


def _pdb_id(path):
    name = path.name.lower().removesuffix(".gz")
    name = re.sub(r"\.(pdb|ent|cif|mmcif)$", "", name)
    if name.startswith("pdb_") and len(name) == 12:
        name = name[-4:]
    return name.upper() if re.fullmatch(r"[0-9][a-z0-9]{3}", name) else None


def _prepare_one(path, chain):
    """Parse once per source and return records in stable chain order."""
    try:
        atoms = parse_structure(path)
        counts = {}
        seen = set()
        for i in range(len(atoms)):
            if str(atoms.res_name[i]) not in SLOTS:
                continue
            key = (str(atoms.chain_id[i]), int(atoms.res_id[i]),
                   str(atoms.ins_code[i]), str(atoms.res_name[i]), bool(atoms.hetero[i]))
            if key not in seen:
                counts[key[0]] = counts.get(key[0], 0) + 1
                seen.add(key)
        if chain == "each":
            chains = sorted(counts)
        elif chain == "largest":
            chains = [min(counts, key=lambda c: (-counts[c], c))] if counts else []
        else:
            chains = [chain]
        if not chains:
            raise ValueError("No canonical polymer chain")
        entry_id = _pdb_id(path)
        records, failures = [], []
        categories = set(atoms.get_annotation_categories())
        for selected in chains:
            try:
                record = tokenize(atoms, chain=selected)
                payload = json.dumps(record, allow_nan=False, separators=(",", ":")).encode()
                entity_keys = []
                if entry_id and "label_entity_id" in categories and "is_polymer" in categories:
                    entity_keys = sorted({f"{entry_id}_{atoms.label_entity_id[i]}"
                        for i in range(len(atoms)) if bool(atoms.is_polymer[i])
                        and (selected == "all" or str(atoms.chain_id[i]) == selected)})
                records.append((selected, payload, len(record["element"]), entity_keys))
            except Exception as exc:
                failures.append(dict(chain=selected, error=f"{type(exc).__name__}: {exc}"))
        return records, failures
    except Exception as exc:
        return [], [dict(chain=chain, error=f"{type(exc).__name__}: {exc}")]


def _bounded_results(executor, files, chain, workers):
    pending = deque()
    iterator = iter(files)
    for _ in range(min(len(files), workers * 2)):
        pending.append(executor.submit(_prepare_one, next(iterator), chain))
    for path in files:
        result = pending.popleft().result()
        yield path, result
        next_path = next(iterator, None)
        if next_path is not None:
            pending.append(executor.submit(_prepare_one, next_path, chain))


def prepare_corpus(source, output, *, chain="each", workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    source, output = Path(source), Path(output)
    files = [source] if source.is_file() else sorted(
        p for p in source.rglob("*") if p.is_file()
        and p.name.lower().removesuffix(".gz").endswith(EXTENSIONS))
    if not files:
        raise ValueError(f"No PDB/mmCIF files found in {source}")
    # Fail before creating output if the required parser is not installed.
    parser_version = version("atomworks")
    output.mkdir(parents=True, exist_ok=False)
    entries, skipped = [], []
    if workers == 1:
        results = ((path, _prepare_one(path, chain)) for path in files)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = _bounded_results(executor, files, chain, workers)
    try:
        for i, (path, (records, failures)) in enumerate(results, 1):
            for failure in failures:
                skipped.append(dict(source=str(path.resolve()), **failure))
            for j, (selected, payload, atoms, entity_keys) in enumerate(records, 1):
                name = f"source_{i:08d}_{j:03d}.json"
                (output / name).write_bytes(payload)
                entries.append(dict(file=name, source=str(path.resolve()), label=path.name,
                    chain=selected, entity_keys=entity_keys, atoms=atoms,
                    sha256=hashlib.sha256(payload).hexdigest()))
    finally:
        if workers > 1:
            executor.shutdown(wait=True, cancel_futures=True)
    manifest = dict(schema_version=1, index_base=0, vocab_sizes=VOCAB_SIZES,
        tokenizer_version=VOCABULARY["version"],
        polymer_vocabulary=VOCABULARY["polymer_vocabulary"],
        parser=dict(name="atomworks", version=parser_version, preset="minimal",
                    model=1, assembly="asymmetric_unit", altloc="first",
                    hydrogens="remove", missing_atoms="virtual", chain=chain),
        preparation=dict(workers=workers),
        entries=entries, skipped=skipped)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not entries:
        raise ValueError(f"No usable structures; see {output / 'manifest.json'}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Local structure file or recursively scanned directory")
    parser.add_argument("output", help="New corpus directory")
    parser.add_argument("--chain", default="each", help="each (default), largest, all, or exact chain ID")
    parser.add_argument("--workers", type=int, default=1, help="Parallel parser processes")
    args = parser.parse_args()
    manifest = prepare_corpus(args.source, args.output, chain=args.chain, workers=args.workers)
    print(f"Prepared {len(manifest['entries'])} sources; {len(manifest['skipped'])} skipped. See {args.output}/manifest.json")


if __name__ == "__main__":
    main()
