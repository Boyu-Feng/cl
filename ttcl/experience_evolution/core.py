from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

def workspace_root(anchor=None):
    """Locate the checkout even when this module lives in a frozen snapshot.

    Keep this helper self-contained: all historical snapshot builders already
    copy experience_evolution, but do not necessarily copy top-level modules.
    """
    override = os.environ.get("TTCL_WORKSPACE")
    if override:
        return Path(override).expanduser().resolve()
    source = Path(anchor or __file__).resolve()
    start = source if source.is_dir() else source.parent
    for candidate in (start, *start.parents):
        if (candidate / "ttcl").is_dir() and (candidate / "current_work").is_dir():
            return candidate
    raise RuntimeError(
        "Cannot locate the project checkout; set TTCL_WORKSPACE to its root."
    )


def python_executable(workspace=None):
    """Use an explicit interpreter, the project runtime, or this interpreter."""
    root = Path(workspace) if workspace is not None else workspace_root()
    override = os.environ.get("TTCL_PYTHON")
    if override:
        expanded = os.path.expanduser(override)
        if os.path.sep not in expanded:
            executable = shutil.which(expanded)
            if executable is None:
                raise FileNotFoundError(f"TTCL_PYTHON executable not found: {override}")
            return Path(executable).absolute()
        executable = Path(expanded)
        if not executable.is_absolute():
            executable = root / executable
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"TTCL_PYTHON is not executable: {executable}")
        # Do not resolve virtualenv symlinks: that can select the system Python.
        return executable.absolute()
    runtime = root / "ttcl/.runtime/alf_delta_env/bin/python"
    if runtime.is_file() and os.access(runtime, os.X_OK):
        return runtime
    return Path(sys.executable).absolute()


ROOT = workspace_root()
MODEL = ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"
FAMILIES = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
            "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]
WRITER_SYSTEM = """You learn reusable experience from completed interactions. Update the previous experience document using ONLY the completed trajectory and its observed outcome. Preserve useful earlier knowledge, revise contradicted beliefs, merge redundant entries, and distinguish supported observations from tentative hypotheses. State when a lesson applies; do not generalize object identifiers or locations across different tasks. Focus on actionable procedures and failure recovery that can help a DIFFERENT future task. No future task or answer is available. Return ONLY the complete updated experience document, in at most 200 words. You may keep the old document unchanged if nothing is learned. Do not solve or describe an imagined next task."""


def digest(value):
    data = value if isinstance(value, bytes) else str(value).encode()
    return hashlib.sha256(data).hexdigest()


def seed(*parts):
    return int(digest("|".join(map(str, parts)))[:8], 16) % (2**31 - 1)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def writer_messages(memory, episode):
    # Explicit allowlist: evaluation metadata, future task, and baseline rollouts
    # cannot enter the writer payload.
    public = {k: episode[k] for k in ["initial_observation", "trajectory", "reward", "steps"]}
    return [{"role": "system", "content": WRITER_SYSTEM},
            {"role": "user", "content": json.dumps({"previous_experience": memory,
             "completed_interaction": public}, ensure_ascii=False)}]


def reward_signal(with_reward, without_reward, mode):
    if mode not in {"delta", "absolute"}:
        raise ValueError(mode)
    delta = float(with_reward) - float(without_reward)
    return delta if mode == "delta" else float(with_reward)


def prepare(data, out):
    data, out = Path(data), Path(out)
    if (out / "plan.json").exists():
        raise FileExistsError("Refusing to overwrite a frozen experiment")
    pools = {split: {f: [] for f in FAMILIES} for split in ["train", "valid_unseen"]}
    hashes = {}
    for split in pools:
        for game in sorted((data / "json_2.1.1" / split).glob("*/*/game.tw-pddl")):
            relative = game.relative_to(data).as_posix()
            if "movable" in relative or "Sliced" in relative:
                continue
            family = next((f for f in FAMILIES if game.parent.parent.name.startswith(f + "-")), None)
            if not family or not read(game).get("solvable", False):
                continue
            pools[split][family].append(relative)
    rng = random.Random(722)
    calibration, training, evaluation = [], [], []
    for family in FAMILIES:
        for split, groups, count in [("train", training, 8), ("valid_unseen", evaluation, 2)]:
            values = pools[split][family].copy()
            rng.shuffle(values)
            if split == "train":
                calibration.append({"family": family, "game": values.pop()})
            if len(values) < count * 3:
                raise ValueError(f"Too few {split}/{family}: {len(values)}")
            for i in range(count):
                chosen = []
                for _ in range(3):
                    # Different task instances and prefer different target objects.
                    old_objects = {v.split("/")[-3].split("-")[1] for v in chosen}
                    idx = next((j for j, v in enumerate(values)
                                if v.split("/")[-3].split("-")[1] not in old_objects), 0)
                    chosen.append(values.pop(idx))
                groups.append({"id": f"{split}:{family}:{i}", "family": family, "games": chosen})
    rng.shuffle(training)
    # Every training task instance appears once; validation is a separate official split.
    all_train = [g for s in training for g in s["games"]]
    all_eval = [g for s in evaluation for g in s["games"]]
    assert len(set(all_train)) == len(all_train)
    assert not set(all_train) & set(all_eval)
    for game in all_train + all_eval + [c["game"] for c in calibration]:
        hashes[game] = digest((data / game).read_bytes())
    plan = {"created_at": time.time(), "dataset": "ALFWorld 0.4.2 / json_2.1.1",
            "data_root": str(data.resolve()), "model": str(MODEL), "seed": 722,
            "training": training, "evaluation": evaluation, "calibration": calibration,
            "pool_counts": {s: {f: len(v) for f, v in p.items()} for s, p in pools.items()},
            "batch_sequences": 4, "train_updates": 12, "sequence_length": 3,
            "actor_url": "http://127.0.0.1:18197", "actor_temperature": 0.7,
            "actor_max_tokens": 64, "max_steps": 30, "writer_max_tokens": 384,
            "writer_temperature": 1.0, "writer_context_limit": 12288,
            "rank": 8, "learning_rate": 1e-5, "kl_beta": 0.01,
            "train_seed": 722, "eval_seeds": [505, 606],
            "arms": ["none", "untrained", "delta", "absolute", "delta_reset"],
            "actor_frozen": True, "experience_only_cross_task_channel": True,
            "reward_baseline": "same next task with EMPTY experience",
            "advantage_centering": False, "checkpoint_selection": "fixed final",
            "pilot": True}
    save(out / "data_hashes.json", hashes)
    save(out / "plan.json", plan)
    save(out / "status.json", {"phase": "prepared"})
    return plan
