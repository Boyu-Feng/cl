"""Mechanistic ALF train replay with matched native actor/writer outputs pinned.

This is a post-hoc diagnostic, not an independent benchmark evaluation. It
reuses recorded output only when the entire prompt and seed match. Unmatched
actor and writer calls still use the live frozen actor service.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .audit_paired_marginal_q import audit as audit_paired
from .collect_credit_train import TTCL_ROOT
from .memory import Embedder, digest
from .paired_marginal_q import PairedQMemory, alf_cell_paired_q


FILES = ('memrl_comparison/memory.py', 'memrl_comparison/paired_marginal_q.py',
         'memrl_comparison/diagnose_paired_q_writer_replay.py')


class WriterReplayClient(Client):
    def __init__(self, plan: dict, repeat: int, writer_log: Path,
                 native_dir: Path, paired_dir: Path, indices: list[int]):
        super().__init__(plan, repeat)
        self.writer_log = writer_log
        self.lookup = {}
        for line in writer_log.read_text().splitlines():
            row = json.loads(line)
            key = (digest(row['messages']), row['actual_generation_seed'])
            if key in self.lookup and self.lookup[key]['raw_response'] != row['raw_response']:
                raise ValueError('Native writer has ambiguous same-prompt output')
            self.lookup[key] = row
        self.hits = []
        self.fallbacks = []
        self.actor_lookup = {}
        self.actor_collisions = []
        for index in indices:
            native_episode = native_dir / f'episode_{index:03d}'
            paired_episode = paired_dir / f'episode_{index:03d}'
            for attempt in range(1, plan['alf']['max_attempts'] + 1):
                path = native_episode / f'attempt_{attempt}' / 'episode.json'
                if path.exists():
                    self._add_actor_episode(path, 'native')
            counterfactual = paired_episode / 'counterfactual_1' / 'episode.json'
            if counterfactual.exists():
                self._add_actor_episode(counterfactual, 'paired_drop')

    def _add_actor_episode(self, path: Path, origin: str) -> None:
        episode = read(path)
        for generation in episode['generations']:
            key = (generation['prompt_sha256'], generation['seed'] % 2**32)
            prior = self.actor_lookup.get(key)
            if prior is not None:
                if prior['text'] != generation['text']:
                    self.actor_collisions.append(dict(prompt_digest=key[0],
                                                      seed=key[1],
                                                      prior_origin=prior['origin'],
                                                      other_origin=origin))
                continue  # Native source has priority for identical prompts.
            self.actor_lookup[key] = dict(origin=origin, **generation)

    def complete(self, messages, random_seed, *, tokens=4096,
                 temperature=.7, top_p=.9):
        if (tokens == self.plan['alf']['actor_max_tokens'] and
                temperature == self.plan['alf']['actor_temperature']):
            key = (digest(messages), random_seed % 2**32)
            generation = self.actor_lookup.get(key)
            if generation is not None:
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                if (hashlib.sha256(rendered.encode()).hexdigest() !=
                        generation['rendered_prompt_sha256']):
                    raise ValueError('Recorded actor prompt does not match rendered replay')
                self.hits.append(dict(kind='actor', prompt_digest=key[0],
                                      seed=key[1], origin=generation['origin']))
                return dict(raw_response=generation['text'],
                            input_tokens=generation['usage']['prompt_tokens'],
                            output_tokens=generation['usage']['completion_tokens'],
                            finish_reason=generation['finish_reason'],
                            actual_generation_seed=key[1],
                            rendered_prompt_sha256=
                            generation['rendered_prompt_sha256'],
                            context_limit=self.plan['context'],
                            seconds=generation['seconds'])
            self.fallbacks.append(dict(kind='actor', prompt_digest=key[0],
                                       seed=key[1]))
        # Native scripts and failure reflections may use different writer
        # temperatures. Identity is the full messages plus generation seed;
        # the writer token budget distinguishes these calls from actor calls.
        if tokens == self.plan['writer_max_tokens']:
            key = (digest(messages), random_seed % 2**32)
            row = self.lookup.get(key)
            if row is not None:
                self.hits.append(dict(kind='writer', prompt_digest=key[0], seed=key[1],
                                      response_sha256=sha_text(row['raw_response'])))
                return {k:copy.deepcopy(v) for k,v in row.items()
                        if k != 'messages'}
            self.fallbacks.append(dict(kind='writer', prompt_digest=key[0], seed=key[1]))
        return super().complete(messages, random_seed, tokens=tokens,
                                temperature=temperature, top_p=top_p)


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def run(source: Path, paired: Path, family: str, output: Path, url: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source_report, paired_report = audit_source(source), audit_paired(paired)
    if (not source_report['complete'] or not paired_report['complete'] or
            source_report['missing'] or paired_report['missing']):
        raise ValueError('Source or paired-Q chain is incomplete')
    source_design = read(source / 'design.json')
    paired_design = read(paired / 'design.json')
    if (family not in source_design['families'] or
            paired_design['selected_games'] != source_design['selected_games'] or
            paired_design['repeat'] != source_design['repeat']):
        raise ValueError('Family or paired-Q lineage differs')
    plan = read(source / 'plan.json')
    plan['url'] = url
    plan['alf']['actor_url'] = url
    repeat = source_design['repeat']
    boot = source_design['bootstrap'][family]
    native_dir = source / 'runs' / 'alfworld' / family / str(repeat) / 'memrl'
    root = native_dir / f"episode_{boot['index']:03d}" / 'memory_after.json'
    writer_log = native_dir / 'memory' / 'writer.jsonl'
    if (sha(root) != boot['sha256'] or not writer_log.exists()):
        raise ValueError('Native root or writer log changed')
    output.mkdir(parents=True)
    design = dict(schema='paired_q_writer_replay_diagnostic_v1', family=family,
                  source=str(source), paired=str(paired), repeat=repeat,
                  source_design_sha256=sha(source / 'design.json'),
                  paired_design_sha256=sha(paired / 'design.json'),
                  root_sha256=sha(root), writer_log_sha256=sha(writer_log),
                  selected_games=source_design['selected_games'][family],
                  url=url, source_sha256={name:sha(TTCL_ROOT / name)
                                          for name in FILES},
                  note='Post-hoc diagnostic: exact matched native actor and writer outputs replayed; unmatched calls live')
    save(output / 'design.json', design)
    for name in FILES:
        target = output / 'source' / 'ttcl' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TTCL_ROOT / name, target)
    paired_dir = paired / 'runs' / 'alfworld' / family / str(repeat) / 'paired_q'
    indices = [boot['index'] + offset
               for offset in range(1, len(design['selected_games']) + 1)]
    replay = WriterReplayClient(plan, repeat, writer_log, native_dir,
                                paired_dir, indices)
    memory = PairedQMemory(plan, replay, output / 'memory',
                           plan['calibration']['alfworld'],
                           embedder=Embedder(plan['embedding']))
    memory.restore(root)
    rows = []
    for offset, item in enumerate(design['selected_games'], 1):
        index = boot['index'] + offset
        target = output / f'episode_{index:03d}'
        memory.snapshot(output / f'episode_{index:03d}_memory_before.json')
        row = alf_cell_paired_q(plan, replay, memory, item, repeat, target)
        if row['status'] != 'complete':
            save(output / 'failure.json', dict(index=index, row=row))
            raise RuntimeError(f'Diagnostic cell failed: {index}')
        native = read(native_dir / f'episode_{index:03d}' / 'row.json')
        rows.append(dict(index=index, game_sha256=item['sha256'],
                         native_first=native['first_attempt'],
                         replay_first=row['first_attempt'],
                         native_within_three=native['within_three'],
                         replay_within_three=row['within_three'],
                         replay_memory_after_sha256=row['memory_after_sha256'],
                         native_memory_after_sha256=sha(
                             native_dir / f'episode_{index:03d}' /
                             'memory_after.json'),
                         matched_native_bank=row['memory_after_sha256'] ==
                         sha(native_dir / f'episode_{index:03d}' /
                             'memory_after.json')))
        save(output / 'progress.json', dict(completed=offset,
                                            expected=len(design['selected_games']),
                                            rows=rows,
                                            writer_replay_hits=replay.hits,
                                            writer_fallbacks=replay.fallbacks,
                                            actor_collisions=replay.actor_collisions))
        print(json.dumps(rows[-1]), flush=True)
        if offset == 1:
            native_episode = read(native_dir / f'episode_{index:03d}' /
                                  'attempt_1' / 'episode.json')
            replay_episode = read(target / 'attempt_1' / 'episode.json')
            native_actions = [step['action'] for step in
                              native_episode['trajectory']]
            replay_actions = [step['action'] for step in
                              replay_episode['trajectory']]
            same_prompt = (native_episode['generations'][0]
                           ['rendered_prompt_sha256'] ==
                           replay_episode['generations'][0]
                           ['rendered_prompt_sha256'])
            same_seed = native_episode['seed'] == replay_episode['seed']
            if not (same_prompt and same_seed and
                    native_actions == replay_actions):
                first_difference = next(
                    (step for step,(left,right) in enumerate(
                        zip(native_actions, replay_actions), 1)
                     if left != right), None)
                stopped = dict(status='invalid_control',
                               reason='First actor trajectory diverged before the writer could be replayed',
                               index=index, same_prompt=same_prompt,
                               same_seed=same_seed,
                               first_action_difference=first_difference,
                               native_actions=len(native_actions),
                               replay_actions=len(replay_actions))
                save(output / 'stopped.json', stopped)
                return dict(design_sha256=sha(output / 'design.json'),
                            completed=offset, rows=rows,
                            writer_replay_hits=replay.hits,
                            writer_fallbacks=replay.fallbacks,
                            actor_collisions=replay.actor_collisions,
                            **stopped)
    report = dict(status='complete',
                  design_sha256=sha(output / 'design.json'),
                  completed=len(rows), rows=rows,
                  writer_replay_hits=replay.hits,
                  writer_fallbacks=replay.fallbacks,
                  actor_collisions=replay.actor_collisions)
    save(output / 'report.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    report = run(args.source.resolve(), args.paired.resolve(), args.family,
                 args.output.resolve(), args.url)
    print(json.dumps({key:value for key,value in report.items()
                      if key not in {'rows','writer_replay_hits','writer_fallbacks'}},
                     indent=2))


if __name__ == '__main__':
    main()
