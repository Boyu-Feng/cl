"""Pilot evidence-bound claim judgments for exact-text credit sign reversals.

The ten selected inputs were chosen after outcome inspection. This pilot can
diagnose feature quality but cannot train or validate a deployable selector.
The actor sees only public task/initial observation and three memory texts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .credit_probe import memory_arms


PROMPT = """Read this ALFWorld task and three retrieved experience texts. Judge
the TARGET experience's most concrete instruction or prerequisite. Use only
the supplied public information. Do not predict reward or infer hidden object
locations. An earlier failure is not proof that this game is impossible.

Current public task: {task}
Initial public observation:\n{observation}

TARGET experience:\n{target}

OTHER experience 1:\n{other1}

OTHER experience 2:\n{other2}

Return exactly one JSON object with keys:
"claim" (one short imperative or prerequisite from TARGET, or empty),
"claim_quote" (an exact nonempty substring of TARGET supporting claim, or
empty when claim is empty),
"initial_support" (one of "supported", "contradicted", "unknown", "no_claim"),
"observation_quote" (an exact substring of the initial observation when
supported or contradicted; otherwise empty),
"peer_relation" (one of "conflicts", "complements", "unrelated", "unknown"),
"peer_quote" (an exact substring of one OTHER experience if conflicts or
complements; otherwise empty).

Use "unknown" when the initial observation does not establish whether the
prerequisite holds. Do not cite the task goal as proof that an object was
found, carried, or acted on. Never invent an observation or quote.
"""
SUPPORT = {'supported', 'contradicted', 'unknown', 'no_claim'}
RELATION = {'conflicts', 'complements', 'unrelated', 'unknown'}
KEYS = {'claim', 'claim_quote', 'initial_support', 'observation_quote',
        'peer_relation', 'peer_quote'}
REVIEWS = {
    '20261003_alf_three_memory_coalitions_holdout6_v1':
        ('group5', 'alf_set_credit_reviewed_20261003.json'),
    '20261003_alf_three_memory_coalitions_group6_v1':
        ('group6', 'alf_set_credit_group6_reviewed_20261003.json'),
    '20261003_alf_three_memory_coalitions_group7_v1':
        ('group7', 'alf_set_credit_group7_reviewed_20261003.json'),
    '20261004_alf_three_memory_coalitions_group8_v1':
        ('group8', 'alf_set_credit_group8_reviewed_20261004.json'),
    '20261004_alf_three_memory_coalitions_enriched_v1':
        ('enriched', 'alf_set_credit_enriched_reviewed_20261004.json'),
}


def _parse(raw: str, target: str, peers: list[str],
           observation: str) -> dict:
    value = raw.strip()
    if value.startswith('```'):
        lines = value.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == '```':
            value = '\n'.join(lines[1:-1])
    parsed = json.loads(value)
    if (not isinstance(parsed, dict) or set(parsed) != KEYS or
            any(not isinstance(v, str) for v in parsed.values()) or
            parsed['initial_support'] not in SUPPORT or
            parsed['peer_relation'] not in RELATION or
            (parsed['claim_quote'] and parsed['claim_quote'] not in target) or
            (parsed['observation_quote'] and
             parsed['observation_quote'] not in observation) or
            (parsed['peer_quote'] and
             not any(parsed['peer_quote'] in peer for peer in peers)) or
            (parsed['initial_support'] in ('supported', 'contradicted') and
             not parsed['observation_quote']) or
            (parsed['peer_relation'] in ('conflicts', 'complements') and
             not parsed['peer_quote']) or
            (parsed['claim'] and not parsed['claim_quote'])):
        raise ValueError('Invalid or ungrounded claim annotation')
    return parsed


def prepare(sign_report: Path, output: Path, url: str) -> tuple[dict, list[dict]]:
    report = read(sign_report)
    if report['schema'] != 'alf_credit_sign_transfer_audit_v1':
        raise ValueError('Wrong sign-transfer source')
    selected = [row for item in report['repeated_memories']
                if item['opposite_signs'] for row in item['instances']]
    if len(selected) != 10:
        raise ValueError('Unexpected diagnostic case count')
    bindings = []
    review_root = Path(__file__).parents[2] / 'data' / 'annotations'
    for row in selected:
        source = Path(row['output'])
        origin = Path(read(source / 'design.json')['origin'])
        spec, arms = memory_arms(origin, row['case'])
        if (spec['source_input_sha256'] != row['input_sha256'] or
                spec['memory_text_sha256'][row['memory_id']] !=
                next(item['memory_text_sha256'] for item in
                     report['repeated_memories'] if row in item['instances'])):
            raise ValueError('Diagnostic text or input binding changed')
        kind, review_name = REVIEWS[source.name]
        review_path = review_root / review_name
        review = read(review_path)
        matching = [item for item in review['targets']
                    if item['origin'] == kind and item['case'] == row['case'] and
                    item['input_sha256'] == row['input_sha256'] and
                    item['public_task'] == row['public_task'] and item['reviewed']]
        if len(matching) != 1:
            raise ValueError('Missing content-bound reviewed public target')
        initial_path = origin / 'runs' / row['case'] / 'attempt_1' / 'episode.json'
        observation = read(initial_path)['initial_observation']
        target = arms[f"only_{row['memory_index']}"]
        peers = [arms[f'only_{i}'] for i in range(3)
                 if i != row['memory_index']]
        prompt = PROMPT.format(task=row['public_task'],
                               observation=observation, target=target,
                               other1=peers[0], other2=peers[1])
        bindings.append(dict(source_output=str(source),
                             source_design_sha256=sha(source / 'design.json'),
                             review_sha256=sha(review_path),
                             origin=str(origin), case=row['case'],
                             input_sha256=row['input_sha256'],
                             snapshot_sha256=spec['snapshot_sha256'],
                             retrieval_sha256=spec['retrieval_sha256'],
                             memory_id=row['memory_id'],
                             memory_index=row['memory_index'],
                             memory_text_sha256=spec['memory_text_sha256'][row['memory_id']],
                             initial_observation_sha256=hashlib.sha256(
                                 observation.encode()).hexdigest(),
                             prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                             prompt=prompt, target=target,
                             peers=peers, observation=observation,
                             diagnostic_sign=row['robust_shapley_sign']))
    manifest = dict(schema='alf_claim_applicability_pilot_v1',
                    sign_report_sha256=sha(sign_report),
                    annotator_script_sha256=sha(Path(__file__)),
                    prompt_template_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                    url=url, count=len(bindings),
                    items=[{key:value for key,value in item.items()
                            if key not in ('prompt', 'target', 'peers',
                                           'observation', 'diagnostic_sign')}
                           for item in bindings],
                    caveat='Posthoc selected sign-flip inputs; judgments are actor-generated diagnostic features, not reviewed reward labels')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'design.json'
    if path.exists():
        if read(path) != manifest:
            raise ValueError('Frozen claim pilot design changed')
    else:
        save(path, manifest)
    return manifest, bindings


def run(sign_report: Path, output: Path, url: str, prepare_only: bool) -> dict:
    design, bindings = prepare(sign_report, output, url)
    if prepare_only:
        return dict(design_sha256=sha(output / 'design.json'), expected=len(bindings))
    plan = read(Path(bindings[0]['origin']) / 'plan.json')
    plan['url'] = url
    client = Client(plan, 0)
    parsed = 0
    for index, item in enumerate(bindings):
        path = output / f'item_{index:02d}.json'
        if path.exists():
            record = read(path)
        else:
            key = hashlib.sha256((item['input_sha256'] + '/' +
                                  item['memory_text_sha256']).encode()).hexdigest()
            response = client.complete([{'role':'user','content':item['prompt']}],
                                       int(key[:8], 16), tokens=384,
                                       temperature=0., top_p=1.)
            record = dict(index=index, design_sha256=sha(output / 'design.json'),
                          prompt_sha256=item['prompt_sha256'],
                          input_sha256=item['input_sha256'],
                          memory_text_sha256=item['memory_text_sha256'],
                          raw_response=response['raw_response'],
                          rendered_prompt_sha256=response['rendered_prompt_sha256'],
                          input_tokens=response['input_tokens'],
                          output_tokens=response['output_tokens'])
            try:
                record['features'] = _parse(response['raw_response'],
                                            item['target'], item['peers'],
                                            item['observation'])
            except (ValueError, json.JSONDecodeError) as exc:
                record['parse_error'] = str(exc)
            save(path, record)
        if (record['design_sha256'] != sha(output / 'design.json') or
                record['prompt_sha256'] != item['prompt_sha256'] or
                record['input_sha256'] != item['input_sha256'] or
                record['memory_text_sha256'] != item['memory_text_sha256']):
            raise ValueError('Cached claim annotation binding changed')
        parsed += int('features' in record)
        print(json.dumps(dict(index=index, features=record.get('features'),
                              parse_error=record.get('parse_error'))), flush=True)
    return dict(design_sha256=sha(output / 'design.json'),
                expected=design['count'], parsed=parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sign-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.sign_report.resolve(), args.output.resolve(),
                         args.url, args.prepare_only)))


if __name__ == '__main__':
    main()
