"""Load audited prompt/algorithm fragments from pinned official repositories.

AST selection avoids importing the upstream legacy OpenAI/LangChain runtime.
The selected function bodies and prompt strings are executed unchanged.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import re
from typing import List, Tuple


def fragments(path, names, namespace=None):
    namespace = dict(namespace or {})
    nodes = []
    found = set()
    for node in ast.parse(Path(path).read_text()).body:
        name = node.name if isinstance(node, ast.FunctionDef) else None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            nodes.append(node)
            found.add(name)
    if found != set(names):
        raise ValueError(f'Missing upstream fragments: {set(names) - found}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class Upstream:
    def __init__(self, root):
        root = Path(root)
        self.reflexion = fragments(root / 'reflexion/generate_reflections.py',
            {'_get_scenario', '_generate_reflection_query'},
            {'List': List, 'FEW_SHOT_EXAMPLES':
             (root / 'reflexion/reflexion_few_shot_examples.txt').read_text()})
        self.expel = fragments(root / 'expel/expel.py',
            {'parse_rules', 'retrieve_rule_index', 'is_existing_rule', 'update_rules'},
            {'re': re, 'List': List, 'Tuple': Tuple})
        self.prompts = fragments(root / 'expel/human.py',
            {'FORMAT_RULES_OPERATION_TEMPLATE', 'CRITIQUE_SUMMARY_SUFFIX',
             'human_critique_existing_rules_all_success_template',
             'human_critique_existing_rules_template'})

    def reflection(self, trajectory, memory):
        return self.reflexion['_generate_reflection_query'](
            'Here is the task:\n' + trajectory, memory[-3:])

    def critique(self, rules, successful, failed=None, task=''):
        name = ('human_critique_existing_rules_template' if failed is not None
                else 'human_critique_existing_rules_all_success_template')
        prompt = self.prompts[name].format(instruction='', success_history=successful,
            fail_history=failed, task=task,
            existing_rules='\n'.join(f'{i}. {r[0]}' for i, r in enumerate(rules, 1)))
        prompt += self.prompts['CRITIQUE_SUMMARY_SUFFIX'][
            'full' if len(rules) >= 10 else 'not_full']
        return prompt

    def update(self, rules, output):
        operations = self.expel['parse_rules'](output)
        # Reject invalid 0/negative EDIT indices before calling upstream code.
        # Such indices otherwise exploit Python negative indexing accidentally.
        rejected = [op for op in operations if op[0].startswith('EDIT ')
                    and int(op[0].split()[1]) < 1]
        operations = [op for op in operations if op not in rejected]
        result = self.expel['update_rules'](copy.deepcopy(rules),
            copy.deepcopy(operations), list_full=len(rules) >= 15)
        return result, operations, rejected


class Retriever:
    """Official all-mpnet-base-v2: 384 tokens, mean pooling, L2 normalization."""
    def __init__(self, model_path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().cpu()
        self.cache = {}

    def encode(self, text):
        if text not in self.cache:
            batch = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=384)
            with self.torch.no_grad():
                states = self.model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1)
                pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1)
                self.cache[text] = self.torch.nn.functional.normalize(pooled, dim=-1)[0]
        return self.cache[text]

    def rank(self, query, documents):
        query_vector = self.encode(query)
        scored = [(i, float(query_vector @ self.encode(d['query'])))
                  for i, d in enumerate(documents)]
        return sorted(scored, key=lambda item: (-item[1], item[0]))
