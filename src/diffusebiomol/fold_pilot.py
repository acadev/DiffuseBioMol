"""Small domain-level similarity proof and oracle-conditioned generation diagnostic."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

from biotite.sequence.align import SubstitutionMatrix, align_optimal
from biotite.structure import superimpose_structural_homologs, tm_score, to_sequence
import numpy as np
import torch

from .data import Corpus, collate
from .flow import cfm_loss, prepare, sample_flow
from .model import FlowModel, ModelConfig
from .tokenizer import VOCAB_SIZES, parse_structure, tokenize


def load_domains(curated_path):
    curated_path = Path(curated_path)
    raw = curated_path.read_bytes()
    curated = json.loads(raw)
    if curated["schema_version"] != 1:
        raise ValueError("Unsupported curated dataset schema")
    ids = [spec["id"] for spec in curated["domains"]]
    if len(ids) != len(set(ids)) or not set(curated["landmarks"]) <= set(ids):
        raise ValueError("Domain IDs must be unique and landmarks must be present")
    domains = {}
    for spec in curated["domains"]:
        path = curated_path.parent / spec["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Curated source checksum mismatch: {path}")
        atoms = parse_structure(path)
        selected = (atoms.chain_id == spec["chain"]) & atoms.is_polymer & \
            (atoms.res_id >= spec["start"]) & (atoms.res_id <= spec["end"])
        atoms = atoms[selected]
        if len(atoms) == 0:
            raise ValueError(f"Empty domain selection: {spec['id']}")
        sequences, _ = to_sequence(atoms)
        if len(sequences) != 1:
            raise ValueError(f"Expected one peptide chain: {spec['id']}")
        domains[spec["id"]] = dict(spec=spec, atoms=atoms, sequence=sequences[0],
                                   record=tokenize(atoms, chain="all"))
    return curated, hashlib.sha256(raw).hexdigest(), domains


def sequence_identity(a, b):
    trace = align_optimal(a, b, SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=(-10, -1), terminal_penalty=False, max_number=1)[0].trace
    aligned = trace[np.all(trace >= 0, axis=1)]
    identity = sum(str(a)[i] == str(b)[j] for i, j in aligned) / len(aligned)
    return identity, len(aligned) / max(len(a), len(b))


def structure_similarity(a, b):
    fitted, _, fixed_indices, mobile_indices = superimpose_structural_homologs(
        a, b, max_iterations=10, reference_length="longer")
    return float(tm_score(a, fitted, fixed_indices, mobile_indices,
                          reference_length="longer")), len(fixed_indices)


def pair_auc(pairs, score_key):
    positive = [p[score_key] for p in pairs if p["same_group"]]
    negative = [p[score_key] for p in pairs if not p["same_group"]]
    if not positive or not negative:
        raise ValueError("Need both same-group and different-group pairs")
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / \
        (len(positive) * len(negative))


def build_map(curated_path):
    curated, curated_sha256, domains = load_domains(curated_path)
    ids = [spec["id"] for spec in curated["domains"]]
    pairs, score = [], {}
    for a, b in combinations(ids, 2):
        left, right = domains[a], domains[b]
        identity, coverage = sequence_identity(left["sequence"], right["sequence"])
        tm, anchors = structure_similarity(left["atoms"], right["atoms"])
        pairs.append(dict(a=a, b=b, same_group=left["spec"]["group"] == right["spec"]["group"],
                          sequence_identity=identity, sequence_coverage=coverage,
                          tm_score=tm, structural_anchors=anchors))
        score[a, b] = score[b, a] = tm
    for id in ids:
        score[id, id] = 1.0
    contexts = {id: [score[id, landmark] for landmark in curated["landmarks"]] + [0.0]
                for id in ids}
    return dict(schema_version=1, curated_sha256=curated_sha256,
        methods=dict(sequence="BLOSUM62 global alignment; identity over aligned residue pairs",
                     structure="Biotite 3Di structural-homolog alignment; TM-score normalized by longer domain"),
        domains=[dict(id=id, group=domains[id]["spec"]["group"],
                      sequence=str(domains[id]["sequence"]),
                      residues=len(domains[id]["sequence"]),
                      atoms=len(domains[id]["record"]["element"])) for id in ids],
        pairs=pairs, landmarks=curated["landmarks"], contexts=contexts,
        summary=dict(same_group_pairs=sum(p["same_group"] for p in pairs),
                     different_group_pairs=sum(not p["same_group"] for p in pairs),
                     sequence_auc=pair_auc(pairs, "sequence_identity"),
                     structure_auc=pair_auc(pairs, "tm_score"),
                     min_same_group_tm=min(p["tm_score"] for p in pairs if p["same_group"]),
                     max_different_group_tm=max(p["tm_score"] for p in pairs if not p["same_group"])))


def _pilot_corpus(curated, domains, root):
    root.mkdir()
    entries = []
    for spec in curated["domains"]:
        id = spec["id"]
        record = domains[id]["record"]
        payload = json.dumps(record, separators=(",", ":"), allow_nan=False).encode()
        name = f"{id}.json"
        (root / name).write_bytes(payload)
        entries.append(dict(file=name, label=id, group=spec["group"],
            atoms=len(record["element"]), sha256=hashlib.sha256(payload).hexdigest()))
    (root / "manifest.json").write_text(json.dumps(dict(schema_version=1,
        index_base=0, vocab_sizes=VOCAB_SIZES, entries=entries), indent=2))
    return Corpus(root)


def _condition(mapping, id, batch):
    return torch.tensor(mapping["contexts"][id], dtype=torch.float32)[None, None, :].expand(
        len(batch["element"]), batch["element"].shape[1], -1)


def _run_generator(corpus, ids, holdout, mapping, *, conditioned, epochs, seed,
                   max_residues, max_atoms):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = FlowModel(ModelConfig(), corpus.vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(seed)
    index = {entry["label"]: i for i, entry in enumerate(corpus.entries)}
    last_loss = None
    for _ in range(epochs):
        for j in rng.permutation(len(ids)):
            id = ids[int(j)]
            host = collate([corpus.load(index[id], max_residues, rng, max_atoms)])
            example = prepare(host, rng)
            cond = _condition(mapping, id, host) if conditioned else None
            optimizer.zero_grad(set_to_none=True)
            loss = cfm_loss(model, host, example, cond=cond)
            loss.backward()
            optimizer.step()
            last_loss = loss.detach().item()
    model.eval()
    validation = {}
    with torch.no_grad():
        for id in holdout:
            evaluation_rng = np.random.default_rng(seed + index[id] + 10000)
            host = collate([corpus.load(index[id], max_residues, evaluation_rng, max_atoms)])
            example = prepare(host, evaluation_rng)
            cond = _condition(mapping, id, host) if conditioned else None
            validation[id] = float(cfm_loss(model, host, example, cond=cond))
        id = holdout[0]
        sample_rng = np.random.default_rng(seed + 20000)
        host = collate([corpus.load(index[id], max_residues, sample_rng, max_atoms)])
        cond = _condition(mapping, id, host) if conditioned else None
        finite_sample = bool(torch.isfinite(sample_flow(model, host, sample_rng,
            steps=3, cond=cond)).all())
    return dict(last_training_loss=last_loss, validation_losses=validation,
                mean_validation_loss=sum(validation.values()) / len(validation),
                finite_three_step_sample=finite_sample)


def train_pilot(curated_path, map_path, run_dir, *, epochs=30, seed=17,
                max_residues=16, max_atoms=256):
    if min(epochs, max_residues, max_atoms) <= 0:
        raise ValueError("Epochs and crop limits must be positive")
    curated, sha256, domains = load_domains(curated_path)
    mapping = json.loads(Path(map_path).read_text())
    if mapping["schema_version"] != 1 or mapping["curated_sha256"] != sha256:
        raise ValueError("Similarity map does not match the curated dataset")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=False)
    corpus = _pilot_corpus(curated, domains, root / "corpus")
    holdout = curated["generation_holdout"]
    train_ids = [spec["id"] for spec in curated["domains"] if spec["id"] not in holdout]
    result = dict(schema_version=1, training_ids=train_ids, holdout_ids=holdout,
        epochs=epochs, max_residues=max_residues, max_atoms=max_atoms, seed=seed,
        context="oracle TM-score similarities to structural landmarks; held-out coordinates used only to construct context and evaluation target",
        baseline=_run_generator(corpus, train_ids, holdout, mapping,
            conditioned=False, epochs=epochs, seed=seed,
            max_residues=max_residues, max_atoms=max_atoms),
        conditioned=_run_generator(corpus, train_ids, holdout, mapping,
            conditioned=True, epochs=epochs, seed=seed,
            max_residues=max_residues, max_atoms=max_atoms))
    (root / "pilot.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mapping = commands.add_parser("map", help="Build a sequence/structure similarity map")
    mapping.add_argument("curated")
    mapping.add_argument("output")
    pilot = commands.add_parser("train", help="Compare baseline and oracle-conditioned tiny generators")
    pilot.add_argument("curated")
    pilot.add_argument("map")
    pilot.add_argument("run_dir")
    pilot.add_argument("--epochs", type=int, default=30)
    pilot.add_argument("--seed", type=int, default=17)
    pilot.add_argument("--max-residues", type=int, default=16)
    pilot.add_argument("--max-atoms", type=int, default=256)
    args = parser.parse_args()
    if args.command == "map":
        result = build_map(args.curated)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"]))
    else:
        result = train_pilot(args.curated, args.map, args.run_dir,
            epochs=args.epochs, seed=args.seed, max_residues=args.max_residues,
            max_atoms=args.max_atoms)
        print(json.dumps({k: result[k] for k in ("baseline", "conditioned")}))


if __name__ == "__main__":
    main()
