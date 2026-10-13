"""Prepare a fold-annotated corpus from extracted CATH domain PDB files."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from .prepare_corpus import _prepare_one
from .tokenizer import VOCABULARY, VOCAB_SIZES


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _domain_id(name):
    stem = Path(name).stem
    if len(stem) != 7 or not stem[0].isdigit():
        return None
    return stem[:4].lower() + stem[4:]


def _selection(path):
    if path is None:
        return None
    ids = set()
    with Path(path).open() as source:
        for line in source:
            if line.strip() and not line.startswith("#"):
                domain = line.split()[0]
                if _domain_id(domain) != domain:
                    raise ValueError(f"Invalid CATH selection domain: {domain}")
                ids.add(domain)
    if not ids:
        raise ValueError("CATH selection is empty")
    return ids


def _classifications(path, selected):
    labels = {}
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 12:
                raise ValueError(f"Invalid CATH list row {line_number}: expected 12 columns")
            domain = fields[0]
            if selected is not None and domain not in selected:
                continue
            if domain in labels:
                raise ValueError(f"Duplicate CATH domain {domain}")
            try:
                hierarchy = [int(value) for value in fields[1:10]]
                length = int(fields[10])
                resolution = float(fields[11])
            except ValueError as error:
                raise ValueError(f"Invalid CATH classification row {line_number}") from error
            if any(value < 0 for value in hierarchy) or length <= 0:
                raise ValueError(f"Invalid CATH classification row {line_number}")
            fold = ".".join(str(value) for value in hierarchy[:3])
            superfamily = ".".join(str(value) for value in hierarchy[:4])
            labels[domain] = dict(fold_id=fold, superfamily_id=superfamily,
                cath_s35=f"{superfamily}.{hierarchy[4]}",
                cath_length=length, resolution=resolution)
    if not labels:
        raise ValueError("No classified CATH domains match the selection")
    return labels


def _results(jobs, workers):
    if workers == 1:
        for domain, path, label in jobs:
            yield domain, path, label, _prepare_one(path, "largest", "pdb")
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        iterator = iter(jobs)
        for _ in range(min(len(jobs), 2 * workers)):
            domain, path, label = next(iterator)
            pending.append((domain, path, label, executor.submit(_prepare_one, path, "largest", "pdb")))
        while pending:
            domain, path, label, future = pending.popleft()
            yield domain, path, label, future.result()
            next_job = next(iterator, None)
            if next_job is not None:
                next_domain, next_path, next_label = next_job
                pending.append((next_domain, next_path, next_label,
                    executor.submit(_prepare_one, next_path, "largest", "pdb")))


def prepare_fold_corpus(domains, classification, output, *, selection=None,
                        workers=1, min_residues=1, max_residues=None,
                        strict=False):
    """Build schema-1 records with CATH fold and homology metadata.

    Coordinates must be the CATH pre-extracted domain PDB files, not whole PDB
    entries. This avoids silently mixing adjacent domains into a fold label.
    """
    if workers < 1 or min_residues < 1 or (max_residues is not None and max_residues < min_residues):
        raise ValueError("Invalid workers or residue limits")
    domains, classification, output = map(Path, (domains, classification, output))
    selected = _selection(selection)
    labels = _classifications(classification, selected)
    unclassified = sorted(selected - labels.keys()) if selected is not None else []
    if unclassified:
        raise ValueError(f"Selection has {len(unclassified)} domains absent from the CATH classification; first: {unclassified[0]}")
    files = {}
    for path in domains.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ("", ".pdb", ".ent"):
            continue
        domain = _domain_id(path.name)
        if domain not in labels:
            continue
        if domain in files:
            raise ValueError(f"Duplicate coordinate file for {domain}: {files[domain]} and {path}")
        files[domain] = path
    missing = sorted(labels.keys() - files.keys())
    if not files:
        raise ValueError(f"No selected CATH coordinates found under {domains}. "
            "The CATH archive extracts extensionless files such as dompdb/12asA00; "
            "check the archive extraction and input directory.")
    if strict and missing:
        raise ValueError(f"Missing {len(missing)} selected CATH coordinates; first: {missing[0]}")
    jobs = [(domain, files[domain], labels[domain]) for domain in sorted(files)]
    parser_version = version("atomworks")
    output.mkdir(parents=True, exist_ok=False)
    entries, skipped = [], []
    for domain, path, label, (records, failures) in _results(jobs, workers):
        for failure in failures:
            skipped.append(dict(domain_id=domain, source=str(path.resolve()), **failure))
        if len(records) != 1:
            if records:
                skipped.append(dict(domain_id=domain, source=str(path.resolve()),
                                    error=f"Expected one domain record, got {len(records)}"))
            continue
        chain, payload, atoms, _ = records[0]
        record = json.loads(payload)
        residues = len(set(zip(record["chain"], record["residue"])))
        if residues < min_residues or (max_residues is not None and residues > max_residues):
            skipped.append(dict(domain_id=domain, source=str(path.resolve()),
                                error=f"Residue count {residues} outside limits"))
            continue
        filename = f"source_{len(entries) + 1:08d}.json"
        (output / filename).write_bytes(payload)
        entries.append(dict(file=filename, source=str(path.resolve()), label=domain,
            domain_id=domain, chain=chain, entity_keys=[], atoms=atoms,
            residues=residues, **label, sha256=hashlib.sha256(payload).hexdigest()))
    fold_counts = dict(sorted(Counter(entry["fold_id"] for entry in entries).items()))
    manifest = dict(schema_version=1, index_base=0, vocab_sizes=VOCAB_SIZES,
        tokenizer_version=VOCABULARY["version"],
        polymer_vocabulary=VOCABULARY["polymer_vocabulary"],
        parser=dict(name="atomworks", version=parser_version, preset="minimal",
            model=1, assembly="asymmetric_unit", altloc="first",
            hydrogens="remove", missing_atoms="virtual", chain="largest"),
        preparation=dict(method="cath_domains", workers=workers,
            classification=str(classification.resolve()),
            classification_sha256=_sha256(classification),
            selection=str(Path(selection).resolve()) if selection else None,
            selection_sha256=_sha256(selection) if selection else None,
            min_residues=min_residues, max_residues=max_residues,
            selected_domains=len(labels), missing_coordinates=len(missing),
            fold_counts=fold_counts), entries=entries, skipped=skipped)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not entries or (strict and skipped):
        raise ValueError(f"Corpus incomplete; see {output / 'manifest.json'}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domains", help="Directory containing extracted CATH domain PDBs")
    parser.add_argument("classification", help="CATH cath-domain-list.txt")
    parser.add_argument("output", help="New corpus directory")
    parser.add_argument("--selection", help="Optional CATH S20/S40 domain .list file")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-residues", type=int, default=1)
    parser.add_argument("--max-residues", type=int)
    parser.add_argument("--strict", action="store_true", help="Fail for missing or skipped domains")
    args = parser.parse_args()
    manifest = prepare_fold_corpus(args.domains, args.classification, args.output,
        selection=args.selection, workers=args.workers,
        min_residues=args.min_residues, max_residues=args.max_residues, strict=args.strict)
    print(f"Prepared {len(manifest['entries'])} domains in {len(manifest['preparation']['fold_counts'])} folds; "
          f"{len(manifest['skipped'])} skipped, {manifest['preparation']['missing_coordinates']} missing")


if __name__ == "__main__":
    main()
