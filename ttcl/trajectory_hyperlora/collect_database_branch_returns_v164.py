"""UNREVIEWED DRAFT: database branch collector; do not train from its output.

Instance binding and instance-vs-run termination require review.
Collect same-question database action branches with a fixed continuation.

The first action is branched from one freshly reset question.  Both branches
then replay the same previously recorded continuation, so the labels are local
counterfactual returns rather than guessed action-quality tags.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "current_work/continual-learning-bench"
sys.path.insert(0, str(BENCH))
from src.interface import Response
from src.tasks.database_exploration.task import DatabaseExploration, DatabaseAction

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def new_task(seed, index):
    # The reviewed source episodes were generated with the official 30-question
    # schedule.  Keep that schedule so the public query and instance binding
    # remain byte-identical when resetting one indexed question.
    t = DatabaseExploration(num_instances=30, seed=seed, max_queries_per_question=15)
    t.build_canonical_run_state()
    # Do not use reset_baseline_instance here: it intentionally selects a
    # one-question run and changes the public ``Question i/30`` binding to
    # ``Question 1/1``.  Keep the canonical schedule and activate index i.
    t._current_question_idx = index
    t._awaiting_action = True
    return t, t.build_current_query()

def response(action):
    return Response(action=DatabaseAction(action=action["action"], content=action.get("content", "")))

def replay(seed, index, actions):
    t, q0 = new_task(seed, index)
    result = None
    for action in actions:
        result = t.step(response(action))
        if result.done:
            break
    outcome = result.instance_outcome if result is not None else None
    if outcome is None:
        # Force termination with the fixed continuation when the action list
        # ended before the environment emitted an outcome.
        while result is not None and not result.done:
            result = t.step(response({"action": "ANSWER", "content": ""}))
        outcome = result.instance_outcome if result is not None else None
    return q0, result, outcome

def main(args):
    if args.output.exists(): raise FileExistsError(args.output)
    rows=[]
    for idx in range(args.num_questions):
        src = ROOT / args.source_root / f"episode_{idx+1:03d}" / "trajectory.json"
        traj = json.loads(src.read_text())
        actions = [s["action"] for s in traj["steps"]]
        if not actions: continue
        positive = actions
        negative = [{"action":"ANSWER","content":""}, *actions[1:]]
        branches=[]; prefix_query=None
        for name, branch in (("recorded_first", positive), ("empty_answer_first", negative)):
            q0, result, outcome = replay(args.seed, idx, branch)
            prefix_query = q0.prompt
            branches.append({"action": branch[0], "label": name,
                "reward": outcome.reward if outcome else None,
                "success": outcome.success if outcome else None,
                "terminal_feedback": result.observation.content if result else ""})
        prefix={"question_index":idx,"query":prefix_query,"recorded_first":positive[0],
                "fixed_continuation":positive[1:],"source_episode_sha256":digest(traj)}
        rows.append({"prefix":prefix,"prefix_sha256":digest(prefix),"branches":branches,
                     "state_id":f"database:seed{args.seed}:question{idx}"})
    out={"protocol":"database_same_state_branch_returns_v164","task":"database_exploration",
         "seed":args.seed,"rows":rows,
         "limitations":"The recorded continuation is replayed after either first action; this is a local counterfactual return and not an optimal SQL policy."}
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"rows":len(rows),"branches":sum(len(x["branches"]) for x in rows)}))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--source-root",default="results/trajectory_hyperlora/clbench_independent_base_v22_database_exploration_full_20261009_episodes"); p.add_argument("--seed",type=int,default=42); p.add_argument("--num-questions",type=int,default=12); main(p.parse_args())
