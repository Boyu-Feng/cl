"""Independent public-event curricula for memory writing and delayed utility training."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from ttcl.memory_writer.core import apply_update
from ttcl.memory_writer.prepare_data import make_row, put, write_rows
from ttcl.memory_writer.train import read_rows, save_json


DOMAINS = {'bridge': ['orchard', 'parcel_depot', 'repair_shop', 'book_archive'],
           'preference': ['orchard', 'parcel_depot', 'repair_shop', 'book_archive'],
           'dev': ['aquarium', 'theater'], 'test': ['observatory', 'textile_studio']}


def episode(index, split):
    rng = random.Random(71000 + index + 100000 * list(DOMAINS).index(split))
    scope = f'{rng.choice(DOMAINS[split])}-{split}-{index}'
    a, b = f'item-{rng.randrange(1000, 4999)}', f'item-{rng.randrange(5000, 9999)}'
    first, second, third = [f'zone-{rng.randrange(10000, 99999)}' for _ in range(3)]
    n, delta = rng.randrange(10, 70), rng.randrange(2, 9)
    command = f'python check_{rng.randrange(1000, 9999)}.py --local'
    memory, rows = [], []
    for turn in range(8):
        event = f'event-{turn}'
        if turn == 0:
            text = (f'Registry {scope}: distinct objects {a} and {b} are both observed. '
                    f'The location of {a} is {first}; the location of {b} is {second}. '
                    'These locations are confirmed, not guesses.')
            ops = [put(scope, a, 'location', first, event), put(scope, b, 'location', second, event)]
        elif turn == 1:
            text = f'Inventory audit at {scope}: {a} has count {n}. All prior location records still apply.'
            ops = [put(scope, a, 'count', n, event)]
        elif turn == 2:
            text = (f'In registry {scope}, {a}-short is an alias for {a}; retain canonical name {a}. '
                    f'An authoritative correction says its location is {third}, replacing {first}. '
                    f'{b} is a different object and has not moved.')
            ops = [put(scope, a, 'location', third, event)]
        elif turn == 3:
            text = (f'A transaction at {scope} adds {delta} to the count of {a}; no removals. '
                    f'The previous audit confirmed {n}. Other attributes remain unchanged.')
            ops = [put(scope, a, 'count', n + delta, event)]
        elif turn == 4:
            text = (f'At {scope}, somebody guesses {b} is now at zone-UNKNOWN. No observation supports '
                    'this and the last confirmed location remains valid. A score from another task '
                    'does not confirm the guess. No verified record changed.')
            ops = []
        elif turn == 5:
            text = (f'Tool feedback in {scope}: checking {b} with command "{command}" returned PASS. '
                    'Store this successful command as validation_command for reuse. A different '
                    'attempt "python broken.py" returned FAIL; do not record it as successful.')
            ops = [put(scope, b, 'validation_command', command, event)]
        elif turn == 6:
            text = (f'Two independent gauges on {b} in {scope} show level_A=17 cm and level_B=17 inch. '
                    'These readings use different units and must remain separate attributes.')
            ops = [put(scope, b, 'level_A', '17 cm', event), put(scope, b, 'level_B', '17 inch', event)]
        else:
            text = (f'Official withdrawal at {scope}: the location record of {a} is no longer valid. '
                    'Remove that attribute; no replacement is known. The other records remain valid.')
            ops = [dict(op='delete', scope=scope, entity=a, attribute='location', evidence=[event])]
        # Both forms are public interaction representations, including an assistant guess
        # and feedback; targets and later questions are never placed inside events.
        if (index + turn) % 2:
            events = [dict(id=event, role='completed_interaction', text=json.dumps({
                'task': f'Inspect records in {scope}.', 'response': 'I have not independently verified any change.',
                'feedback': text}))]
        else:
            events = [dict(id=event, role='observation', text=text)]
        row = make_row(f'utility/{split}/{index}/{turn}', f'operation_{turn}', memory, events, ops,
                       split=split, services=[scope], episode=f'{split}-{index}')
        # Exactly two matched follow-up questions. Writer never sees these fields.
        changed_attribute = ('location', 'count', 'location', 'count', 'location',
                             'validation_command', 'level_B', 'location')[turn]
        queried_entity = b if turn in (4, 5, 6) else a
        queries = [(queried_entity, changed_attribute), (b, 'location')]
        expected = {(r['entity'], r['attribute']): r for r in row['expected_after']}
        row['probes'] = [dict(scope=scope, entity=entity, attribute=attribute,
                              expected={k: expected[(entity, attribute)][k] for k in ('value', 'status')}
                              if (entity, attribute) in expected else {'value': [], 'status': 'unknown'})
                         for entity, attribute in queries]
        rows.append(row)
        memory = row['expected_after']
    return {'id': f'utility/{split}/{index}', 'rows': rows}


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    bridge = [r for i in range(64) for r in episode(i, 'bridge')['rows']]
    # Maintain an explicit replay control against forgetting, equal to half the budget.
    replay = random.Random(733).sample(read_rows(source / 'train_mixed.jsonl'), 512)
    train = bridge + replay
    random.Random(42).shuffle(train)
    pref = [r for i in range(16) for r in episode(i, 'preference')['rows']]
    dev = [r for i in range(4) for r in episode(i, 'dev')['rows']]
    test_streams = [episode(i, 'test') for i in range(8)]
    test = [r for s in test_streams[:4] for r in s['rows']]
    files = {'train_bridge': train, 'preference_train': pref, 'dev_utility': dev,
             'test_utility': test, 'test_utility_streams': test_streams}
    manifest = {'clbench_used_for_training': False, 'generator_seed': 71000,
                'replay_source': str(source), 'sgd_derived_license': 'CC-BY-SA-4.0',
                'independent_domains': DOMAINS, 'datasets': {},
                'limitations': 'Synthetic templates; held-out domains and IDs, not unrestricted semantic generalization.'}
    for name, rows in files.items():
        path = output / (name + '.jsonl')
        write_rows(path, rows)
        manifest['datasets'][name] = {'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    # Assert no complete public event repeats across independent curriculum splits.
    seen = {}
    for split, rows in [('bridge', bridge), ('preference', pref), ('dev', dev), ('test', test)]:
        fingerprints = {hashlib.sha256(json.dumps(r['events'], sort_keys=True).encode()).hexdigest() for r in rows}
        assert not any(fingerprints & old for old in seen.values())
        seen[split] = fingerprints
        for row in rows:
            assert apply_update(row['memory_before'], row['target'], {e['id'] for e in row['events']}) == row['expected_after']
    manifest['public_events_disjoint'] = True
    save_json(output / 'manifest.json', manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    prepare(args.source, args.output)


if __name__ == '__main__':
    main()
