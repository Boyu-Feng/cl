"""Model-generated updates; Python only validates and executes generic writes."""
import copy
import json

PROMPT = """You maintain evidence-based structured memory for a sequence of interactions.
Given existing memory, optional field descriptions, and new public events, output ONLY
JSON: {"operations": [...]}. Each operation has op ("put" or "delete"), scope,
entity, attribute, evidence (a list of event IDs). A put also has value (a list of
strings) and status ("observed" or "hypothesis"). Use [] for no changes.
Use exact entity names, values and units from evidence; field descriptions may define
canonical field names and categorical values. Retain unmentioned old facts without
rewriting them. Put replaces only the same scope/entity/attribute. Delete marks that
key inactive only if explicit evidence retracts it. Correct superseded values. Keep
different contexts and non-equivalent concepts separate. Do not treat an assistant's
guess or a scalar reward as confirmation. Uncertain claims use hypothesis status.
For dialogue-state schemas, use the supplied service as scope and "user" as entity;
store filled slot values, not requests for information. Do not guess missing values.
For general observations, infer useful entities/attributes from the evidence, including
concrete numeric records and their conditions. Citations must be visible event IDs.
"""


def key(row):
    return row['scope'], row['entity'], row['attribute']


def messages(memory, events, schema=None):
    return [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': json.dumps(
        {'memory': memory, 'field_descriptions': schema, 'events': events}, ensure_ascii=False, sort_keys=True)}]


def parse_update(raw):
    raw = raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = '\n'.join(raw.splitlines()[1:-1])
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {'operations'} or not isinstance(value['operations'], list):
        raise ValueError('Expected exactly one operations list')
    return value


def apply_update(memory, update, event_ids):
    """Atomic application; invalid proposals leave the original memory untouched."""
    result = {key(row): copy.deepcopy(row) for row in memory}
    touched = set()
    if len(update['operations']) > 64:
        raise ValueError('Too many operations')
    for op in update['operations']:
        if not isinstance(op, dict) or op.get('op') not in ('put', 'delete'):
            raise ValueError('Unknown memory operation')
        fields = {'op', 'scope', 'entity', 'attribute', 'evidence'}
        if op['op'] == 'put':
            fields |= {'value', 'status'}
        if set(op) != fields:
            raise ValueError('Incorrect operation fields')
        if any(not isinstance(op[f], str) or not op[f].strip() for f in ('scope', 'entity', 'attribute')):
            raise ValueError('Memory keys must be nonempty strings')
        k = key(op)
        if k in touched:
            raise ValueError('Repeated writes to one key')
        touched.add(k)
        refs = op['evidence']
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in event_ids for r in refs):
            raise ValueError('Evidence must cite visible events')
        if op['op'] == 'delete':
            if k not in result:
                raise ValueError('Cannot delete an unknown key')
            del result[k]  # Original and proposal remain in the append-only audit log.
        else:
            if (not isinstance(op['value'], list) or not op['value']
                    or any(not isinstance(v, str) or not v.strip() for v in op['value'])
                    or op['status'] not in ('observed', 'hypothesis')):
                raise ValueError('Invalid value/status')
            result[k] = {field: copy.deepcopy(op[field]) for field in sorted(fields - {'op'})}
    return [result[k] for k in sorted(result)]


def state_values(memory):
    return {key(r): (tuple(sorted(r['value'])), r['status']) for r in memory}


def state_score(actual, expected):
    a, b = state_values(actual), state_values(expected)
    correct = sum(k in b and b[k] == v for k, v in a.items())
    precision = correct / len(a) if a else float(not b)
    recall = correct / len(b) if b else float(not a)
    return {'exact_state': a == b, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
