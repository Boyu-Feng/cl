"""Convert official SGD states to supervised deltas; add independent synthetic tasks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from ttcl.memory_writer.core import apply_update, key


def write_rows(path, rows):
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def put(scope, entity, attribute, value, event, status='observed'):
    return dict(op='put', scope=scope, entity=entity, attribute=attribute,
                value=value if isinstance(value, list) else [str(value)], status=status, evidence=[event])


def make_row(identity, family, memory, events, ops, schema=None, **metadata):
    target = {'operations': ops}
    expected = apply_update(memory, target, {e['id'] for e in events})
    return dict(id=identity, family=family, memory_before=memory, events=events, schema=schema,
                target=target, expected_after=expected, **metadata)


def sgd_rows(source, split):
    schemas = {s['service_name']: s for s in json.loads((source / split / 'schema.json').read_text())}
    rows, dialogues = [], []
    for path in sorted((source / split).glob('dialogues_*.json')):
        for dialogue in json.loads(path.read_text()):
            memory, history, sequence = [], [], []
            for turn_index, turn in enumerate(dialogue['turns']):
                event_id = f'e{turn_index:03}'
                history.append(dict(id=event_id, role=turn['speaker'].lower(), text=turn['utterance']))
                if turn['speaker'] != 'USER':
                    continue
                ops, active_schemas = [], []
                existing = {key(r): r for r in memory}
                for frame in turn['frames']:
                    service = frame['service']
                    if 'state' not in frame:
                        continue
                    public_schema = schemas[service]
                    active_schemas.append({'service': service, 'description': public_schema['description'],
                                           'slots': [{k: slot[k] for k in ('name', 'description', 'possible_values')}
                                                     for slot in public_schema['slots']]})
                    state = frame['state']['slot_values']
                    for attribute, values in sorted(state.items()):
                        k = service, 'user', attribute
                        values = sorted(set(values))
                        if k not in existing or existing[k]['value'] != values:
                            ops.append(put(service, 'user', attribute, values, event_id))
                    for k in sorted(existing):
                        if k[0] == service and k[1] == 'user' and k[2] not in state:
                            ops.append(dict(op='delete', scope=k[0], entity=k[1], attribute=k[2], evidence=[event_id]))
                if not active_schemas:
                    continue
                row = make_row(f"sgd/{split}/{dialogue['dialogue_id']}/{turn_index}", 'sgd', memory,
                               history[-4:], ops, active_schemas, split=split,
                               dialogue_id=f"{split}:{dialogue['dialogue_id']}",
                               source_dialogue_id=dialogue['dialogue_id'],
                               services=[s['service'] for s in active_schemas])
                rows.append(row)
                sequence.append(row)
                memory = row['expected_after']
            if sequence:
                dialogues.append(sequence)
    return rows, dialogues


def synthetic_row(index, split, rng):
    skill = ['alias', 'distinct', 'retention', 'correction', 'uncertain', 'retraction', 'procedure', 'numeric'][index % 8]
    domains = {'train': ['warehouse', 'library', 'package_build', 'garden'],
               'dev': ['festival', 'weather_station'], 'test': ['museum', 'space_habitat']}
    scope = rng.choice(domains[split]) + '-' + str(rng.randrange(10000, 99999))
    entity = rng.choice(['Lumen', 'Orion', 'Vela', 'Nova']) + '-' + str(rng.randrange(1000, 9999))
    attribute = rng.choice(['owner', 'destination', 'label'])
    old, new = rng.sample(['Cedar', 'Birch', 'Maple', 'Iris', 'Oak', 'Fern'], 2)
    initial = [put(scope, entity, attribute, old, 'e000')]
    for j in range(rng.randrange(2, 7)):
        initial.append(put(scope, f'unrelated-{j}', 'reference', str(rng.randrange(100, 999)), 'e000'))
    memory = apply_update([], {'operations': initial}, {'e000'})
    event = 'e100' if skill == 'retention' else 'e010'
    if skill == 'alias':
        text = (f"The registry confirms '{entity}-short' is an alias for '{entity}' in {scope}. "
                f"Its {attribute} has been changed to {new}. Keep using the canonical entity name.")
        ops = [put(scope, entity, attribute, new, event)]
    elif skill == 'distinct':
        text = (f"In {scope}, '{entity}-B' is a separate entity from '{entity}', despite similar names. "
                f"The verified {attribute} of '{entity}-B' is {new}.")
        ops = [put(scope, entity + '-B', attribute, new, event)]
    elif skill == 'retention':
        text = (f"A later verified log for {scope} adds entity '{entity}-later' with {attribute} {new}. "
                "Nothing has retracted the earlier records. Do not discard unmentioned records.")
        ops = [put(scope, entity + '-later', attribute, new, event)]
    elif skill == 'correction':
        text = (f"Correction from the authoritative record: the previous {attribute} {old} for {entity} "
                f"in {scope} was mistaken. The verified value is {new}.")
        ops = [put(scope, entity, attribute, new, event)]
    elif skill == 'uncertain':
        text = (f"An assistant speculates that {entity} in {scope} has {attribute} {new}, but no source "
                f"confirms it. Existing verified {attribute} {old} remains valid. A score of 0.8 "
                "for a different task is not evidence for this speculation.")
        ops = []
    elif skill == 'retraction':
        text = (f"The administrator explicitly withdraws the {attribute} record for {entity} in {scope}. "
                "Remove this active attribute; no replacement is known. Other records remain valid.")
        ops = [dict(op='delete', scope=scope, entity=entity, attribute=attribute, evidence=[event])]
    elif skill == 'procedure':
        command = f'python verify_{rng.randrange(10, 99)}.py --offline'
        text = (f"In {scope}, tool execution confirmed the command '{command}' successfully validates {entity}. "
                "Save this verified command under attribute validation_command for future reuse.")
        ops = [put(scope, entity, 'validation_command', command, event)]
    else:
        n, increment = rng.randrange(2, 25), rng.randrange(1, 9)
        memory = apply_update(memory, {'operations': [put(scope, entity, 'count', n, 'e000')]}, {'e000'})
        text = (f"The current verified count for {entity} in {scope} was {n}. A new transaction adds "
                f"{increment} units without any removals. Update its count; preserve all other facts.")
        ops = [put(scope, entity, 'count', n + increment, event)]
    if split != 'train':
        text = 'Evidence bulletin. ' + text.replace('verified', 'confirmed').replace('Keep using', 'Retain')
    return make_row(f'synthetic/{split}/{index}', skill, memory,
                    [dict(id=event, role='observation', text=text)], ops, split=split, services=[scope.split('-')[0]])


def balanced(rows, count, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['services'][0], bool(row['target']['operations']))].append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count and any(groups.values()):
        for k in sorted(groups):
            if groups[k] and len(selected) < count:
                selected.append(groups[k].pop())
    rng.shuffle(selected)
    return selected


def retention_stream(index):
    scope, memory, rows = f'archive-test-{index}', [], []
    for turn in range(12):
        event = f'e{turn:03}'
        entity = f'record-{turn}'
        value = str(400 + index * 20 + turn)
        if turn == 11:
            entity, value = 'record-0', str(900 + index)
            text = f'Correction: in {scope}, record-0 reference is now {value}. Other stored records remain valid.'
        else:
            text = f'The archive {scope} confirms {entity} has reference {value}. Preserve prior records.'
        row = make_row(f'synthetic/stream/{index}/{turn}', 'long_gap_stream', memory,
                       [dict(id=event, role='observation', text=text)],
                       [put(scope, entity, 'reference', value, event)], split='test', services=[scope])
        rows.append(row)
        memory = row['expected_after']
    return dict(id=f'retention-{index}', rows=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-size', type=int, default=2048)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    splits, sequences = {}, {}
    for split in ('train', 'dev', 'test'):
        splits[split], sequences[split] = sgd_rows(args.source, split)
    train = balanced(splits['train'], args.train_size, 42)
    if len(train) != args.train_size:
        raise ValueError('Insufficient SGD training rows')
    synth = {split: [synthetic_row(i, split, random.Random(offset + i)) for i in range(count)]
             for split, offset, count in [('train', 100000, args.train_size), ('dev', 200000, 64), ('test', 300000, 128)]}
    datasets = {'train_sgd': train,
                'train_mixed': train[:args.train_size // 2] + synth['train'][:args.train_size // 2],
                'dev_sgd': balanced(splits['dev'], 64, 43), 'test_sgd': balanced(splits['test'], 128, 44),
                'dev_synthetic': synth['dev'], 'test_synthetic': synth['test']}
    manifest = {'source_manifest': json.loads((args.source / 'source_manifest.json').read_text()),
                'sgd_derived_license': 'CC-BY-SA-4.0', 'clbench_used_for_training': False,
                'annotations_visible_to_writer': False, 'datasets': {}}
    for name, rows in datasets.items():
        path = args.output / (name + '.jsonl')
        write_rows(path, rows)
        manifest['datasets'][name] = {'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'families': dict(Counter(r['family'] for r in rows)),
                                     'services': sorted({s for r in rows for s in r['services']}),
                                     'noops': sum(not r['target']['operations'] for r in rows)}
    random.Random(43).shuffle(sequences['test'])
    write_rows(args.output / 'test_streams.jsonl', [dict(id=f'stream-{i}', rows=seq[:12])
                                                  for i, seq in enumerate(sequences['test'][:4])]
               + [retention_stream(i) for i in range(4)])
    train_dialogues = {r['dialogue_id'] for r in train}
    assert not train_dialogues & {r['dialogue_id'] for r in splits['dev'] + splits['test']}
    dialogue_hashes = {}
    for split in ('train', 'dev', 'test'):
        dialogue_hashes[split] = set()
        for path in (args.source / split).glob('dialogues_*.json'):
            for dialogue in json.loads(path.read_text()):
                text = json.dumps([(t['speaker'], t['utterance']) for t in dialogue['turns']])
                dialogue_hashes[split].add(hashlib.sha256(text.encode()).hexdigest())
    assert not dialogue_hashes['train'] & (dialogue_hashes['dev'] | dialogue_hashes['test'])
    manifest['train_eval_dialogue_content_disjoint'] = True
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: len(v) for k, v in datasets.items()}, indent=2))


if __name__ == '__main__':
    main()
