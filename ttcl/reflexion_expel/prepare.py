"""Freeze implementation, upstream fragments and declared evaluation protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from ttcl.experience_v2.common import WORKSPACE, BENCH, MODEL, save, sha_file

PROTOCOL = """# Reflexion and ExpeL on CLBench

This is a source-grounded CLBench port, not a reproduction of the papers' scores.
Official repositories and commits are in provenance.json. Frozen fragments live
in upstream/. The legacy OpenAI/LangChain launchers are not used. Both actor and
critic run the local frozen Qwen3-4B-Instruct-2507, with no LoRA or parameter update.
The generic CLBench JSON actor replaces domain-specific ReAct action parsers and
actor demonstrations. Official ALFWorld reflection demonstrations are retained.

## Fixed split and evaluation

For each task, the first 20% of canonical instances are reserved for ExpeL
experience gathering. The remaining 80% are the common held-out evaluation.
Indices are fixed before execution: BSM 18/72, Poker 24/96, Database 4/16,
Cohort 4/16. Environment seed 42; model repeats 303/404, with a separate
experience bank for each repeat. Public initial query hashes and index separation
are audited. Related studies/database schema may be shared between instances,
as intended by CLBench; this is not a disjoint-world generalization claim.
The prefix split may encounter later distribution shifts and has small training
sets for Database/Cohort. No outcome-driven split or checkpoint selection.

Methods: none (one attempt), retry_none (up to three independent attempts),
reflexion (up to three attempts with failure reflection), expel (one attempt
using frozen learned rules plus up to two semantically retrieved successful
training trajectories). Reflexion memory resets between held-out instances;
it is same-task adaptation, not a cross-task reflection-bank variant.
Primary per-method result is first official success or last allowed attempt,
never the maximum scalar reward. Report first-attempt reward, official success,
final reward, attempt count, model-format failures, paired differences, repeat
results and costs. retry_none controls for the extra environment retry budget,
but has no critic calls, so it is not equal total model compute.
Shared identical first attempts are reused with explicit provenance and counted
logically for each policy; physical compute is reported separately.
No aggregate across task reward scales. No claim of statistical significance
from intermediate means. Unscored failures reduce coverage, not silent zeros.

## Reflexion

Use the official alfworld_runs/generate_reflections.py prompt builder and examples
unchanged. Supply complete recorded actions/public observations plus completed
attempt reward and official success. Retain at most three previous reflection
plans. Retry a freshly reset identical canonical instance only after failure.
The maximum here is three total attempts (a declared budget, not a reproduction
of every upstream experiment). Failed format/turn-cap attempts are disclosed.

## ExpeL

Gather each training instance with the same three-attempt Reflexion loop.
Extract insights by contrasting same-instance successful and failed attempts,
then from batches of at most eight successes, split at 48000 trajectory tokens
to fit complete evidence within the context limit. As in upstream create_rules,
instances without any successful attempt supply reflections during gathering,
but do not create contrastive or all-success rule updates.
Use official human critique templates, parse_rules, update_rules and importance
counters. The household-specific critique system instruction is replaced with
a generic CLBench instruction; human operation templates are unchanged.
Target 10 rules; at >=10 request removals and discourage additions; at >=15
use upstream stronger removal. This is the original soft limit, not a hard cap.
The only parser guard rejects EDIT indices below one, avoiding accidental Python
negative indexing; all guards/empty updates and length stops are logged.

Retrieval uses the official all-mpnet-base-v2 model on CPU, max length 384,
attention-mask mean pooling and L2 normalization, with exact cosine ranking.
This is equivalent ranking to L2 nearest neighbours on normalized embeddings.
Retrieve by initial task query, up to two different successful training tasks.
Whole trajectories only, with a declared total context retrieval cap of 12000
actor tokens. Oversized examples are skipped and logged, not silently truncated.
All rules are retained. Model context overflow causes an explicit failure.
No CLBench domain-specific handcrafted demonstrations. Frozen rules and retrieval
bank receive no held-out trajectory or feedback updates. No weight training.

## Feedback and budgets

After each completed attempt the official scalar reward and success flag are
provided for reflection and source experience extraction. No hidden ground truth,
future task outcome, evaluator internals or baseline trajectory enters prompts.
This is a declared reward-visible extension, different from the previous v2
public-observations-only transfer run. Do not directly interpret a score difference
from that experiment as a method-only gain. LoCoMo is outside this experiment.

Actor sampling: temperature .7, top_p .9, max 4096 tokens/action, 64 turns,
two packaging retries, same canonical instance/attempt seeds across controls.
Critic: greedy, max 1024 tokens, no best-of-N selection or quality retries.
Context length 65536. Full raw messages, outputs, terminal rewards, costs,
retrieval provenance, rule operations and retry trajectories are saved.
Sales/Codebase remain excluded because Docker socket access is unavailable;
they are not assigned zero scores. Runtime and relevant benchmark source hashes
are frozen. Use REPORT.md/summary.json/status.json for progress and results.
"""


def upstream_metadata(repo, workspace):
    """Read the original source revision after vendoring without nested Git."""
    manifest = workspace / "config/upstreams.json"
    if manifest.is_file():
        repositories = json.loads(manifest.read_text())["repositories"]
        entry = repositories.get(repo.name)
        if entry and entry.get("url") and entry.get("commit"):
            return {"url": entry["url"], "commit": entry["commit"]}
    # Git otherwise searches parent directories and could report this project's
    # revision instead of the upstream. Require metadata at the upstream root.
    if not (repo / ".git").exists():
        raise FileNotFoundError(
            f"Missing original upstream revision for {repo.name}; "
            f"restore its entry in {manifest}"
        )
    git_root = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if Path(git_root).resolve() != repo.resolve():
        raise ValueError(f"Git metadata does not belong to upstream {repo}")
    return {
        "url": subprocess.check_output(
            ["git", "-C", str(repo), "remote", "get-url", "origin"], text=True
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def prepare(root, smoke=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        raise FileExistsError('Do not replace a frozen experiment')
    totals = {'blind_spectrum_monitoring': 90, 'exploitable_poker': 120,
              'database_exploration': 20, 'cohort_studies': 20}
    tasks = {task: {'total': count, 'train': list(range(count//5)),
                   'test': list(range(count//5, count))} for task, count in totals.items()}
    if smoke:
        tasks = {task: {'total': totals[task], 'train': [0, 1], 'test': [2, 3]}
                 for task in ['blind_spectrum_monitoring', 'exploitable_poker']}
    plan = {'tasks': tasks, 'repeats': [303] if smoke else [303, 404],
        'url': 'http://127.0.0.1:18227', 'model': str(MODEL), 'concurrency': 2,
        'embedding_model': str(WORKSPACE/'ttcl/models/all-mpnet-base-v2'),
        'smoke_only': smoke, 'feedback': 'terminal scalar and official success after attempt',
        'arms': ['none', 'retry_none', 'reflexion', 'expel'], 'max_attempts': 3}
    save(root/'plan.json', plan)
    (root/'PROTOCOL.md').write_text(PROTOCOL + ('\nSMOKE ONLY: 2 source + 2 evaluation instances/task; not main results.\n' if smoke else ''))
    source = root/'source/ttcl'
    source.mkdir(parents=True, exist_ok=False)
    (source/'__init__.py').write_text('')
    for name in ['reflexion_expel', 'experience_v2', 'experience_evolution',
                 'common', 'structured_memory', 'llm_memory']:
        shutil.copytree(WORKSPACE/'ttcl'/name, source/name,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    upstream = {
        'reflexion': {'repo': WORKSPACE/'current_work/reflexion', 'files': {
            'alfworld_runs/generate_reflections.py': 'generate_reflections.py',
            'alfworld_runs/reflexion_few_shot_examples.txt': 'reflexion_few_shot_examples.txt'}},
        'expel': {'repo': WORKSPACE/'current_work/ExpeL', 'files': {
            'agent/expel.py': 'expel.py', 'prompts/templates/human.py': 'human.py'}}}
    provenance = {}
    for name, info in upstream.items():
        out = root/'upstream'/name
        out.mkdir(parents=True)
        repo = info['repo']
        metadata = upstream_metadata(repo, WORKSPACE)
        for src, target in info['files'].items():
            shutil.copy2(repo/src, out/target)
        for p in repo.glob('LICENSE*'):
            if p.is_file():
                shutil.copy2(p, out/p.name)
        provenance[name] = {**metadata, 'source_files': info['files']}
    save(root/'provenance.json', provenance)
    files = [p for d in [root/'source', root/'upstream'] for p in d.rglob('*') if p.is_file()]
    files += list((BENCH/'src').rglob('*.py'))
    files += [root/n for n in ['plan.json', 'PROTOCOL.md', 'provenance.json']]
    files += [p for p in Path(plan['embedding_model']).glob('*') if p.is_file()]
    save(root/'input_hashes.json', {str(p): sha_file(p) for p in files})
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    print(prepare(args.root, args.smoke))
