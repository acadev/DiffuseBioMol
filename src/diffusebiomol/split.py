"""Deterministic source or sequence-cluster-disjoint train/validation splits."""
import hashlib
from pathlib import Path

import numpy as np


def make_split(corpus, seed, *, sequence_clusters=None, validation_fraction=0.1,
               max_validation_sources=1024, group_by="source"):
    if not 0 < validation_fraction < 1 or max_validation_sources < 1:
        raise ValueError("Validation fraction must be in (0, 1) and cap must be positive")
    if group_by not in ("source", "superfamily", "fold"):
        raise ValueError("group_by must be source, superfamily, or fold")
    if sequence_clusters is not None and group_by != "source":
        raise ValueError("Choose either sequence clusters or a CATH grouping")
    entries = corpus.entries
    parent = list(range(len(entries)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    if sequence_clusters is None:
        first = {}
        for i, entry in enumerate(entries):
            if group_by == "source":
                key = entry.get("source", entry["file"])
            else:
                field = "superfamily_id" if group_by == "superfamily" else "fold_id"
                if field not in entry:
                    raise ValueError(f"Corpus entry {i} lacks {field}")
                key = entry[field]
            if key in first:
                union(i, first[key])
            else:
                first[key] = i
        metadata = dict(method=group_by, cluster_sha256=None)
    else:
        raw = Path(sequence_clusters).read_bytes()
        cluster_by_entity = {}
        for line_number, line in enumerate(raw.decode().splitlines(), 1):
            for key in line.split():
                key = key.upper()
                if key in cluster_by_entity:
                    raise ValueError(f"Entity {key} occurs in multiple cluster rows (line {line_number})")
                cluster_by_entity[key] = line_number
        first = {}
        missing = set()
        for i, entry in enumerate(entries):
            keys = entry.get("entity_keys", [])
            if not keys:
                missing.add(entry.get("label", entry["file"]))
                continue
            for key in keys:
                cluster = cluster_by_entity.get(key.upper())
                if cluster is None:
                    missing.add(key)
                    continue
                if cluster in first:
                    union(i, first[cluster])
                else:
                    first[cluster] = i
        if missing:
            examples = ", ".join(sorted(missing)[:5])
            raise ValueError(f"Sequence cluster coverage missing for {len(missing)} entities/records: {examples}")
        metadata = dict(method="sequence_clusters",
                        cluster_sha256=hashlib.sha256(raw).hexdigest())

    groups = {}
    for i in range(len(entries)):
        groups.setdefault(find(i), []).append(i)
    components = list(groups.values())
    if len(components) < 2:
        raise ValueError("Need at least two independent source/sequence groups for validation")
    order = np.random.default_rng(seed).permutation(len(components))
    target = max(1, min(max_validation_sources, round(validation_fraction * len(entries))))
    validation = []
    for j in order:
        group = components[int(j)]
        if len(validation) + len(group) <= max_validation_sources \
                and len(validation) + len(group) < len(entries):
            validation.extend(group)
            if len(validation) >= target:
                break
    if not validation:
        raise ValueError("No independent validation group fits the validation cap")
    held_out = set(validation)
    training = [i for i in range(len(entries)) if i not in held_out]
    metadata.update(validation_fraction=validation_fraction,
                    max_validation_sources=max_validation_sources,
                    validation_sources=len(validation), training_sources=len(training),
                    independent_groups=len(components))
    return training, sorted(validation), metadata
