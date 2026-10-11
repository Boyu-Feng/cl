"""One bounded correction from an observed prefix, without resetting the env.

Hooks are scoped to this process and restored on exit. The frozen actor source
file is unchanged. No future feedback or episode outcome enters the trigger.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import torch

from ttcl.trajectory_hyperlora import alfworld_zero_shot_probe as probe
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52b import bounded_source_fields
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import bounded_residual, effective_norm
from ttcl.trajectory_hyperlora.prepare_alfworld_off_goal_feedback_v57 import off_goal


def trigger_reason(initial, records):
    if not records:
        return None
    latest=records[-1]
    if off_goal(initial,latest['action'],latest['feedback']) is not None:
        return 'public_goal_object_mismatch'
    if len(records)>=12:
        counts=Counter((r['observation'],r['action']) for r in records[-8:])
        if sum(n-1 for n in counts.values())>=2:
            return 'repeated_observation_action_pairs'
    return None


def corrected_episode(agent, tokenizer, game, success_fields, *, device,
                      mode='away', seed=42, max_steps=30):
    if mode not in ('none','away','random'):
        raise ValueError(mode)
    if not agent.task_conditioned or agent.task_context_scope!='current':
        raise ValueError('Current-observation task conditioning required')
    original_source=agent.set_source
    original_generate=probe.generate
    records=[]; residual=None; last_target=None; previous_command=None
    previous_feedback=None; current_available=None; event=None

    def apply_residual():
        if residual is not None:
            for layer,value in zip(agent.adapters,residual,strict=True):
                layer.b=layer.b+value.to(layer.b)

    def set_source(fields,*args,**kwargs):
        nonlocal last_target
        result=original_source(fields,*args,**kwargs)
        if fields is not None:
            last_target=kwargs.get('target_fields')
            apply_residual()
        return result

    def generate(*args,**kwargs):
        nonlocal previous_command
        response=original_generate(*args,**kwargs)
        previous_command=probe.clean_command(response,current_available)
        return response

    def observe(initial,feedback,available):
        nonlocal previous_feedback,current_available,residual,event
        if previous_command is not None:
            records.append({'observation':previous_feedback,'action':previous_command,'feedback':feedback})
        previous_feedback=feedback; current_available=available
        reason=trigger_reason(initial,records)
        if mode!='none' and event is None and reason is not None:
            if last_target is None: raise ValueError('Missing current target context')
            with torch.no_grad():
                good=[layer.b.detach().clone() for layer in agent.adapters]
                prefix_fields=bounded_source_fields(agent,tokenizer,records,device)
                original_source(prefix_fields,target_fields=last_target)
                prefix=[layer.b.detach().clone() for layer in agent.adapters]
                toward,metrics=bounded_residual(agent,good,prefix,max_relative=.25)
                away=[-x for x in toward]
                if mode=='away':
                    residual=away
                else:
                    task_seed=int(hashlib.sha256((str(game)+str(seed)).encode()).hexdigest()[:16],16)
                    generator=torch.Generator(device='cpu').manual_seed(task_seed)
                    random_factors=[torch.randn(x.shape,generator=generator,dtype=torch.float32) for x in away]
                    norm=effective_norm(agent,random_factors)
                    if norm<=0: raise ValueError('Zero random correction norm')
                    scale=metrics['residual_effective_norm']/norm
                    residual=[x*scale for x in random_factors]
                applied=effective_norm(agent,residual)
                if abs(applied-metrics['residual_effective_norm'])>max(1e-7,1e-5*applied):
                    raise ValueError('Correction norm mismatch')
                event={'before_turn':len(records),'reason':reason,'prefix_records':list(records),
                    'prefix_sha256':digest(records),'mode':mode,'metrics':metrics,
                    'applied_effective_norm':applied,
                    'residual_sha256':digest([x.cpu().tolist() for x in residual])}
                original_source(success_fields,target_fields=last_target)
                apply_residual()
        return ''

    agent.set_source=set_source
    probe.generate=generate
    try:
        episode=probe.run_episode(agent,tokenizer,game,success_fields,adapter=True,
            device=device,max_steps=max_steps,max_new_tokens=64,constrain_actions=True,
            actor_history_turns=2,loop_guard_max=2,evidence_provider=observe)
        episode['within_attempt_correction']=event
        return episode
    finally:
        probe.generate=original_generate
        agent.set_source=original_source
        original_source(None)
