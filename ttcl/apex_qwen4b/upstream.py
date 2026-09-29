"""Execute selected official functions unchanged, outside the distributed launcher."""
from __future__ import annotations
import ast
from collections import defaultdict
import importlib.util
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np
from .common import workspace


def source_root():
    return Path(__file__).resolve().parent/'official' if (Path(__file__).resolve().parent/'official').exists() else workspace()/'current_work/APEx'


def module(relative, name):
    spec=importlib.util.spec_from_file_location(name,source_root()/relative)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def functions(relative,names,namespace=None,constants=()):
    path=source_root()/relative;tree=ast.parse(path.read_text())
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    found=set()
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list=[];nodes.append(node);found.add(node.name)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
    if found!=set(names):raise ValueError(f'Missing official functions: {set(names)-found}')
    ns=dict(json=json,re=re,logger=logging.getLogger('apex.official'))
    ns.update(namespace or {})
    code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec')
    exec(code,ns)
    return SimpleNamespace(**{name:ns[name] for name in names})


def grpo():
    import torch
    def masked_mean(x,mask,axis=None):
        return (x*mask).sum(dim=axis)/mask.sum(dim=axis).clamp_min(1)
    return functions('Executor-Train/Train/verl/trainer/ppo/core_algos.py',
        ['compute_grpo_outcome_advantage','compute_policy_loss','agg_loss'],
        dict(torch=torch,np=np,defaultdict=defaultdict,verl_F=SimpleNamespace(masked_mean=masked_mean)))


def executor_reward():
    return functions('Executor-Train/Train/local_search/mmsearch.py',['compute_score']).compute_score


def writer_reward(client,prompts):
    return functions('Writer-Train/Train/writer_skill/writer_dataset.py',
        ['_parse_skill_json','_score_format','_score_skill_quality','_score_llm_quality',
         '_score_evolution_rationality','compute_score'],
        dict(client=client,model_name='qwen',format_memories=prompts.format_memories,
             SKILL_JUDGE_PROMPT=prompts.SKILL_JUDGE_PROMPT),
        ['VALID_OPERATIONS','REQUIRED_SKILL_FIELDS'])
