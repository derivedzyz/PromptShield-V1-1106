"""Selection and source accounting for continuation from a released teacher."""
from collections import Counter
import hashlib
import json
import math

import torch


FLOORS = (.82, .86, .90, .94, .97)
MAX_TV = .12


def agreement(first, second):
    label = first.argmax(1)
    score = torch.minimum(first.max(1).values, second.max(1).values)
    distance = (first - second).abs().sum(1) / 2
    stable = (label == second.argmax(1)) & (distance <= MAX_TV)
    return label, score, distance, stable


def fit_policy(first, second, truth):
    label, score, _, stable = agreement(first, second)
    policy = []
    for c in range(3):
        rule = {"threshold": None, "support": 0, "correct": 0}
        for floor in FLOORS:
            kept = stable & (label == c) & (score >= floor)
            support = int(kept.sum())
            correct = int((truth[kept] == c).sum())
            if support < 5 or correct / support < .95:
                continue
            # Few calibration examples give little evidence for relaxing the floor.
            if support < 12 and (floor < .94 or correct < support):
                continue
            rule = {"threshold": floor, "support": support, "correct": correct}
            break
        policy.append(rule)
    return policy


def select(rows, first, second, policy, rng, cap):
    label, score, distance, stable = agreement(first, second)
    choices = []
    stats = Counter()
    for c, rule in enumerate(policy):
        cohort = stable & (label == c)
        if rule["threshold"] is None or not cohort.any():
            continue
        confidence = score[cohort]
        mean, spread = float(confidence.mean()), max(.035, float(confidence.std(unbiased=False)))
        for i in (cohort & (score >= rule["threshold"])).nonzero().flatten().tolist():
            shortfall = min(0., float(score[i]) - mean)
            quality = math.exp(-.5 * (shortfall / spread) ** 2) * (1 - float(distance[i]))
            choices.append({**rows[i], "class": c, "soft": ((first[i]+second[i])/2).tolist(),
                            "view_scores": [float(first[i,c]), float(second[i,c])],
                            "weight": quality, "view_tv": float(distance[i])})
            stats[c] += 1
    rng.shuffle(choices)
    selected, seen, counts = [], set(), Counter()
    for row in choices:
        group, c = row["group"], row["class"]
        if group not in seen and counts[c] < cap:
            selected.append(row)
            counts[c] += 1
            seen.add(group)
    return selected, dict(stats)


def inherited_groups(teacher, info, prior_trial):
    ledger = teacher / "source-groups.json"
    if "source_groups_sha256" in info:
        data = ledger.read_bytes()
        if hashlib.sha256(data).hexdigest() != info["source_groups_sha256"]:
            raise ValueError("Source group ledger fingerprint mismatch")
        saved = json.loads(data)
        if saved["adapter_sha256"] != info["sha256"]:
            raise ValueError("Source group ledger belongs to another teacher")
        return set(saved["groups"])
    if prior_trial is None:
        if "student" in info.get("role", "").lower():
            raise ValueError("Recursive teacher requires a source ledger or its accepted trial")
        return set()
    trial = json.loads((prior_trial / "trial.json").read_bytes())
    if not trial["complete"] or trial["active_sha256"] != info["sha256"]:
        raise ValueError("Prior trial does not identify this accepted teacher")
    active = trial["start_sha256"]
    groups = set()
    for record in trial["rounds"]:
        if record["teacher_sha256"] != active:
            raise ValueError("Broken teacher chain")
        if not record["promoted"]:
            break
        folder = prior_trial / f"round-{record['round']}"
        weights = folder / "student/adapter/adapter_model.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != record["student_sha256"]:
            raise ValueError("Prior student fingerprint mismatch")
        selected = json.loads((folder / "pseudo-labels.json").read_bytes())
        batch = {r["source_group"] for r in selected}
        if len(selected) != record["selected"] or len(batch) != len(selected):
            raise ValueError("Incomplete prior source group ledger")
        groups.update(batch)
        active = record["student_sha256"]
    if active != info["sha256"]:
        raise ValueError("Prior trial ended at a different teacher")
    return groups
