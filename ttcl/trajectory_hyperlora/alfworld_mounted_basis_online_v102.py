"""ALFWorld reward-gated mounted-B coefficient pilot (v102)."""
from __future__ import annotations
import argparse, hashlib, json, re
from contextlib import nullcontext
from pathlib import Path
import torch
from ttcl.experience_evolution.environment import ACTOR_SYSTEM
from .alfworld_zero_shot_probe import load_agent, run_episode, score_commands
from .alfworld_online_from_empty import file_hash
from .contextual_alf_source import contextual_text_fields, task_context_text, contextual_source_fields
from .alfworld_online_from_empty import records_from_episode
from .basis_state_controller_v101 import (BasisStateController, PersistentBasisMemory,
                                           FeatureBasisRouter, SuccessPrototypeMemory)
from .basis_state_controller_v101 import AnchoredDeltaController
from .basis_state_controller_v101 import SignedBasisController
from .joint_domains_v93 import ROOT, filehash

def structured_task_key(task_id: str, device: str, dim: int = 128) -> torch.Tensor:
    """Deterministic sparse feature key from public task metadata."""
    vec = torch.zeros(dim, device=device)
    tokens = [x.lower() for x in re.split(r'[/_.-]+', task_id) if x]
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], 'little') % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[index] += sign
    if vec.norm() == 0:
        vec[0] = 1.0
    return torch.nn.functional.normalize(vec, dim=0)

def loss_for_action(agent, tok, observation, commands, command, device, bb):
    for layer, b in zip(agent.adapters, bb, strict=True): layer.b = b * 1.0
    messages=[{'role':'system','content':ACTOR_SYSTEM},
              {'role':'user','content':str(observation)+'\nAvailable commands:\n'+'\n'.join(commands)},
              {'role':'assistant','content':command}]
    text=tok.apply_chat_template(messages[:2],tokenize=False,add_generation_prompt=True)
    prefix=tok(text,add_special_tokens=False).input_ids
    suffix=tok(command+tok.eos_token,add_special_tokens=False).input_ids
    ids=torch.tensor([prefix+suffix],device=device)
    labels=torch.tensor([[-100]*len(prefix)+suffix],device=device)
    return agent.model(input_ids=ids,labels=labels,use_cache=False).loss

def loss_for_trajectory(agent, tok, episode, device, bb, discount=0.9,
                        credit_steps: int = 4):
    """Credit verified actions with their actual preceding observations.

    ``credit_steps=0`` uses complete-trajectory action cross-entropy; the historical default keeps the last four action window.
    """
    observation = episode['initial_observation']
    losses=[]
    weights=[]
    start = 0 if credit_steps <= 0 else max(0, len(episode['trajectory']) - credit_steps)
    # Full-trajectory hindsight can retain many transformer activations. Move
    # saved tensors to host memory so the low-dimensional state update does not
    # fail merely because a successful episode is long. The historical tail
    # path remains unchanged and stays on device.
    graph_context = (torch.autograd.graph.save_on_cpu(pin_memory=False)
                     if credit_steps <= 0 else nullcontext())
    with graph_context:
        for index, step in enumerate(episode['trajectory']):
            commands = step.get('available_commands', [])
            if index >= start and commands:
                weight = discount ** (index - start)
                weights.append(weight)
                losses.append(weight * loss_for_action(
                    agent, tok, observation, commands, step['command'], device, bb))
            observation = step['observation']
    if not losses: raise ValueError('successful episode has no action records')
    return torch.stack(losses).sum() / sum(weights)


def loss_for_action_kl(agent, tok, observation, commands, command, device,
                       student_bb, teacher_bb, temperature=1.0):
    """Distill teacher action-token probabilities into a mounted adapter."""
    if temperature <= 0:
        raise ValueError('KL temperature must be positive')
    messages = [{'role': 'system', 'content': ACTOR_SYSTEM},
                {'role': 'user', 'content': str(observation) +
                 '\nAvailable commands:\n' + '\n'.join(commands)}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    prefix = tok(text, add_special_tokens=False).input_ids
    suffix = tok(command + tok.eos_token, add_special_tokens=False).input_ids
    ids = torch.tensor([prefix + suffix], device=device)
    start = len(prefix) - 1
    stop = start + len(suffix)
    for layer, b in zip(agent.adapters, teacher_bb, strict=True):
        layer.b = b * 1.0
    with torch.no_grad():
        teacher = agent.model(input_ids=ids, use_cache=False).logits[:, start:stop]
    for layer, b in zip(agent.adapters, student_bb, strict=True):
        layer.b = b * 1.0
    student = agent.model(input_ids=ids, use_cache=False).logits[:, start:stop]
    teacher_prob = torch.softmax(teacher / temperature, dim=-1)
    student_logprob = torch.log_softmax(student / temperature, dim=-1)
    return torch.nn.functional.kl_div(student_logprob, teacher_prob,
                                      reduction='batchmean') * (temperature ** 2)


def loss_for_trajectory_kl(agent, tok, episode, device, student_bb,
                           teacher_bb, discount=0.9, credit_steps=4,
                           temperature=1.0):
    """KL credit over a verified trajectory, paired with preceding observations."""
    observation = episode['initial_observation']
    losses, weights = [], []
    start = 0 if credit_steps <= 0 else max(0, len(episode['trajectory']) - credit_steps)
    for index, step in enumerate(episode['trajectory']):
        commands = step.get('available_commands', [])
        if index >= start and commands:
            weight = discount ** (index - start)
            weights.append(weight)
            losses.append(weight * loss_for_action_kl(
                agent, tok, observation, commands, step['command'], device,
                student_bb, teacher_bb, temperature=temperature))
        observation = step['observation']
    if not losses:
        raise ValueError('successful episode has no action records')
    return torch.stack(losses).sum() / sum(weights)

def failure_residual(agent, tok, episode, device, bb):
    """Return the observed-action loss and its ascent direction in mounted B."""
    vars_ = [x.detach().clone().requires_grad_(True) for x in bb]
    loss = loss_for_trajectory(agent, tok, episode, device, vars_)
    grads = torch.autograd.grad(loss, vars_)
    return loss.detach(), [g.detach() for g in grads]

def successful_episode(episode):
    return (episode['status'] == 'complete' and episode['reward'] > 0
            and bool(episode['trajectory']))


def direct_write_source(direct_episode, selected_episode, *, shared):
    """Shared-source controls must not substitute their own successful rollout."""
    candidate = selected_episode if shared else direct_episode
    return candidate if successful_episode(candidate) else None


def compile_trajectory_basis(agent, tok, episode, target_observation, device):
    """Compile a verified trajectory for a separate target context."""
    records = records_from_episode(episode)
    source = contextual_source_fields(agent, tok, records, device, 2048,
                                      pooling='both')
    target = contextual_text_fields(
        agent, tok,
        task_context_text(target_observation, target_observation),
        device, 2048, pooling='both')
    with torch.no_grad():
        agent.set_source(source, target_fields=target)
        basis = [a.b.detach().clone() for a in agent.adapters]
        agent.set_source(None)
    return basis


def run(args):
    if args.shared_write_source and not ((args.hindsight_basis or args.hindsight_prototype or args.hindsight_composition) and
            args.verified_source in ('base', 'frozen') and
            args.success_only_updates and args.direct_init_frozen):
        raise ValueError('shared source requires hindsight basis, base/frozen source, '
                         'success-only updates and equal frozen initialization')
    if args.verified_source != 'persistent' and not (args.hindsight_basis or
                                                     args.hindsight_prototype or
                                                     args.hindsight_composition):
        raise ValueError('external write sources require a hindsight controller')
    if args.deferred_target_conditioning and not (args.hindsight_basis or
                                                  args.hindsight_composition):
        raise ValueError('deferred target conditioning requires a basis controller')
    if args.tied_signed and not args.signed_composition:
        raise ValueError('--tied-signed requires --signed-composition')
    if args.output.exists(): raise FileExistsError(args.output)
    if args.lookahead_gate:
        raise ValueError('lookahead is a development-only protocol; disabled pending rollback audit')
    review=json.loads(args.review.read_text())
    if args.hindsight_basis or args.hindsight_prototype or args.hindsight_composition:
        from .freeze_hindsight_v139 import validate_manifest
        validate_manifest(review, vars(args), args.data_root)
    run_binding = {'runner_sha256': filehash(Path(__file__)),
                   'controller_sha256': filehash(Path(__file__).with_name('basis_state_controller_v101.py')),
                   'review_sha256': filehash(args.review),
                   'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    agent,tok=load_agent(args.model,args.checkpoint,args.device,args.gpu_fraction)
    def trajectory_loss(episode, matrices, teacher=None):
        if args.trajectory_kl:
            if teacher is None and args.kl_teacher == 'compiled_basis':
                raise ValueError('compiled-basis KL requires an explicit teacher')
            return loss_for_trajectory_kl(
                agent, tok, episode, args.device, matrices,
                teacher if teacher is not None else teacher_adapter,
                discount=args.credit_discount, credit_steps=args.credit_steps,
                temperature=args.kl_temperature)
        return loss_for_trajectory(agent, tok, episode, args.device, matrices,
                                   discount=args.credit_discount,
                                   credit_steps=args.credit_steps)
    ext_dir=args.external_dir
    ext=torch.load(ext_dir/'trained.pt',map_location='cpu',weights_only=True)
    named=dict(agent.named_parameters()); trainable={n for n,p in named.items() if p.requires_grad}
    if trainable != set(ext['hyper']): raise ValueError('external hyper architecture mismatch')
    for n,v in ext['hyper'].items(): named[n].data.copy_(v.to(args.device))
    agent.requires_grad_(False);agent.eval()
    cache=torch.load(ext_dir/'cache.pt',map_location='cpu',weights_only=True)
    def f(k): return {x:v.to(args.device) for x,v in cache['fields'][k].items()}
    bases=[]
    for row in cache['prepared'][:4]:
        with torch.no_grad():
            agent.set_source(f(row['source']),target_fields=f(row['query']))
            bases.append([a.b.detach().clone() for a in agent.adapters])
    controllers={}
    delta_controllers={}
    signed_controllers={}
    feature_router = None
    fallback = [sum(0.5 / len(bases) * bases[i][j] for i in range(len(bases)))
                for j in range(len(bases[0]))]
    # The zero mounted adapter is the fixed teacher for KL hindsight credit.
    teacher_adapter = [torch.zeros_like(b) for b in fallback]
    prototype_memory = SuccessPrototypeMemory(
        fallback, capacity=args.prototype_capacity,
        threshold=args.prototype_threshold) if (args.prototype_memory or
                                                args.hindsight_prototype) else None
    keyed_memory = PersistentBasisMemory(
        bases, capacity=args.memory_capacity, temperature=args.memory_temperature,
        slot_similarity=args.memory_slot_similarity,
        lr=args.state_lr, decay=args.decay) if args.keyed_memory else None
    frozen_initial = [torch.zeros_like(b) for b in fallback] if args.keyed_memory else [b.detach().clone() for b in fallback]
    direct_start = fallback if args.direct_init_frozen else [torch.zeros_like(b) for b in fallback]
    direct=[torch.nn.Parameter(x.detach().clone()) for x in direct_start]
    direct_opt=torch.optim.SGD(direct,lr=args.direct_lr)
    pending_sources = {}
    rows=[]
    for index, target in enumerate(review['targets'][:args.limit]):
        game=args.data_root/target['game']
        family=Path(target['game']).parts[-3].split('-')[0]
        if keyed_memory is not None or args.feature_router or prototype_memory is not None or args.hindsight_prototype or args.anchored_delta or args.hindsight_basis or args.signed_composition:
            controller = None
        elif args.family_state:
            controller=controllers.setdefault(family, BasisStateController(
                bases,lr=args.state_lr,decay=args.decay,
                anchor_strength=args.anchor_strength))
        else:
            controller=controllers.setdefault('__global__', BasisStateController(
                bases,lr=args.state_lr,decay=args.decay,
                anchor_strength=args.anchor_strength))
        zero=[torch.zeros_like(b) for b in bases[0]]
        with torch.no_grad():
            initial_key=None
        base=run_episode(agent,tok,game,{},adapter=False,device=args.device,max_steps=args.max_steps,max_new_tokens=args.max_new_tokens,constrain_actions=True)
        with torch.no_grad():
            # Reuse the public initial observation as the task key.
            key_text = task_context_text(base['initial_observation'], base['initial_observation'])
            key=contextual_text_fields(agent,tok,key_text,args.device,2048,pooling='mean')['contextual']
            key = key.flatten()
            if args.structured_key or args.prototype_memory:
                key = torch.cat((key, structured_task_key(base['initial_observation'], args.device)))
        deferred_basis = None
        deferred_read_basis = None
        deferred_gate = None
        deferred_action_gate = None
        deferred_persistent_update = None
        deferred_write_gate = None
        deferred_source_episode = pending_sources.get(family)
        if args.deferred_target_conditioning and family in pending_sources:
            deferred_basis = compile_trajectory_basis(
                agent, tok, pending_sources[family],
                base['initial_observation'], args.device)
            if args.deferred_basis_read == 'direct':
                deferred_read_basis = [args.deferred_basis_scale * x.detach().clone()
                                       for x in deferred_basis]
                if args.deferred_loss_gate:
                    with torch.no_grad():
                        candidate_loss = float(trajectory_loss(
                            deferred_source_episode, deferred_read_basis).detach())
                        baseline_loss = float(trajectory_loss(
                            deferred_source_episode, frozen_initial).detach())
                    accepted = candidate_loss <= baseline_loss
                    deferred_gate = {'candidate_loss': candidate_loss,
                                     'baseline_loss': baseline_loss,
                                     'accepted': accepted}
                    if not accepted:
                        deferred_read_basis = None
            if args.trajectory_write_gate:
                with torch.no_grad():
                    candidate_loss = float(trajectory_loss(
                        deferred_source_episode, deferred_basis).detach())
                    baseline_loss = float(trajectory_loss(
                        deferred_source_episode, frozen_initial).detach())
                deferred_write_gate = {
                    'candidate_loss': candidate_loss,
                    'baseline_loss': baseline_loss,
                    'accepted': candidate_loss <= baseline_loss,
                    'source': 'deferred_basis',
                }
            if args.hindsight_composition:
                if deferred_write_gate is None or deferred_write_gate['accepted']:
                    controller.append_basis(deferred_basis,
                                            initial_logit=args.success_basis_logit)
                    deferred_persistent_update = controller.update_mounted(
                        lambda bs: trajectory_loss(deferred_source_episode, bs,
                                                   teacher=deferred_basis))
        if args.signed_composition:
            signed_controller = signed_controllers.setdefault(
                family, SignedBasisController(bases, lr=args.state_lr,
                                              bound=args.signed_bound, tied=args.tied_signed,
                                              preserve_l1=args.signed_preserve_l1,
                                              positive=args.signed_positive))
            bb, prototype_read = signed_controller.matrices(), None
            memory_read = None
        elif args.hindsight_prototype:
            bb, prototype_read = prototype_memory.read(key)
            memory_read = None
        elif args.anchored_delta or args.hindsight_basis:
            delta_controller = delta_controllers.setdefault(
                family, AnchoredDeltaController(fallback, lr=args.state_lr,
                                                max_fraction=args.delta_fraction))
            if deferred_basis is not None:
                delta_controller.add_success_delta(
                    deferred_basis, initial_weight=args.delta_initial_weight)
                deferred_persistent_update = delta_controller.update_mounted(
                    lambda bs: trajectory_loss(deferred_source_episode, bs))
            bb, prototype_read = delta_controller.matrices(), None
            memory_read = None
        elif args.prototype_memory:
            bb, prototype_read = prototype_memory.read(key)
            memory_read = None
        elif args.feature_router:
            if feature_router is None:
                feature_router = FeatureBasisRouter(bases, key.numel(), lr=args.state_lr,
                                                    decay=args.decay)
            bb, memory_read = feature_router.matrices(key), None
            prototype_read = None
        elif keyed_memory is not None:
            prototype_read = None
            bb, memory_read = keyed_memory.read(key)
            if bb is None:
                bb = [torch.zeros_like(b) for b in bases[0]]
        else:
            memory_read = None
            bb=controller.matrices()
        if deferred_read_basis is not None:
            bb = deferred_read_basis
        if deferred_read_basis is not None and args.deferred_action_gate:
            commands = list(base['trajectory'][0].get('available_commands', []))
            messages = [{'role': 'system', 'content': ACTOR_SYSTEM},
                        {'role': 'user', 'content':
                         str(base['initial_observation']) +
                         '\nAvailable commands:\n' + '\n'.join(commands)}]
            def mount(values):
                for layer, value in zip(agent.adapters, values, strict=True):
                    layer.b = value * 1.0
            def margin(scores):
                values = sorted(scores.values(), reverse=True)
                return values[0] - values[1] if len(values) > 1 else values[0]
            mount(frozen_initial)
            frozen_scores = score_commands(agent, tok, messages, commands,
                                           args.device)
            mount(deferred_read_basis)
            candidate_scores = score_commands(agent, tok, messages, commands,
                                              args.device)
            frozen_margin, candidate_margin = margin(frozen_scores), margin(candidate_scores)
            accepted = candidate_margin > frozen_margin
            deferred_action_gate = {'accepted': accepted,
                                    'frozen_margin': frozen_margin,
                                    'candidate_margin': candidate_margin,
                                    'frozen_top': max(frozen_scores, key=frozen_scores.get),
                                    'candidate_top': max(candidate_scores, key=candidate_scores.get)}
            if not accepted:
                bb = [x.detach().clone() for x in frozen_initial]
        frozen_ep=run_episode(agent,tok,game,{},adapter=False,fixed_adapter=frozen_initial,device=args.device,max_steps=args.max_steps,max_new_tokens=args.max_new_tokens,constrain_actions=True)
        persistent=run_episode(agent,tok,game,{},adapter=False,fixed_adapter=bb,device=args.device,max_steps=args.max_steps,max_new_tokens=args.max_new_tokens,constrain_actions=True)
        deferred_direct_update = None
        if args.deferred_target_conditioning and deferred_source_episode is not None:
            direct_opt.zero_grad()
            dl = trajectory_loss(deferred_source_episode, direct,
                                 teacher=deferred_basis)
            dl.backward()
            direct_opt.step()
            deferred_direct_update = {'loss': float(dl.detach()), 'source': 'deferred'}
        direct_ep=run_episode(agent,tok,game,{},adapter=False,fixed_adapter=[x.detach() for x in direct],device=args.device,max_steps=args.max_steps,max_new_tokens=args.max_new_tokens,constrain_actions=True)
        update={'persistent':False,'direct':False,'lookahead':None}
        if deferred_persistent_update is not None:
            update['deferred_persistent_update'] = deferred_persistent_update
        if deferred_gate is not None:
            update['deferred_loss_gate'] = deferred_gate
        if deferred_write_gate is not None:
            update['deferred_write_gate'] = deferred_write_gate
        if deferred_action_gate is not None:
            update['deferred_action_gate'] = deferred_action_gate
        if deferred_direct_update is not None:
            update['deferred_direct_update'] = deferred_direct_update
        # In the hindsight protocol a verified successful trajectory may come
        # from the frozen/base arm.  This lets the stream write experience
        # even when the current persistent arm has not solved the task.
        source_episode, source_arm = persistent, 'persistent'
        if args.verified_source == 'base':
            source_episode, source_arm = base, 'base'
        elif args.verified_source == 'frozen':
            source_episode, source_arm = frozen_ep, 'frozen'
        elif args.verified_source == 'any':
            for candidate, label in ((persistent, 'persistent'),
                                     (base, 'base'), (frozen_ep, 'frozen')):
                if candidate['status'] == 'complete' and candidate['reward'] > 0 and candidate['trajectory']:
                    source_episode, source_arm = candidate, label
                    break
        update['verified_source'] = source_arm
        source_is_success = (source_episode['status']=='complete' and
                             source_episode['reward']>0 and
                             source_episode['trajectory'])
        if source_is_success and not args.deferred_target_conditioning and (args.hindsight_basis or args.hindsight_prototype or args.hindsight_composition or
                                  (persistent['status']=='complete' and persistent['reward']>0)):
            if args.signed_composition:
                update['persistent'] = signed_controller.update_mounted(
                    lambda bs: trajectory_loss(source_episode, bs))
            elif args.hindsight_prototype:
                records = records_from_episode(source_episode)
                sfresh = contextual_source_fields(agent, tok, records, args.device,
                                                   2048, pooling='both')
                tfresh = contextual_text_fields(agent, tok,
                    task_context_text(source_episode['initial_observation'], source_episode['initial_observation']),
                    args.device, 2048, pooling='both')
                with torch.no_grad():
                    agent.set_source(sfresh, target_fields=tfresh)
                    own_basis = [a.b.detach().clone() for a in agent.adapters]
                    agent.set_source(None)
                update['prototype_added'] = prototype_memory.add(key, own_basis)
                update['persistent'] = {'prototype': True, 'verified_source': source_arm}
            elif args.anchored_delta or args.hindsight_basis:
                records = records_from_episode(source_episode)
                sfresh = contextual_source_fields(agent, tok, records, args.device,
                                                   2048, pooling='both')
                tfresh = contextual_text_fields(agent, tok,
                    task_context_text(source_episode['initial_observation'], source_episode['initial_observation']),
                    args.device, 2048, pooling='both')
                with torch.no_grad():
                    agent.set_source(sfresh, target_fields=tfresh)
                    own_basis = [a.b.detach().clone() for a in agent.adapters]
                    agent.set_source(None)
                update['delta_added'] = delta_controller.add_success_delta(
                    own_basis, initial_weight=args.delta_initial_weight)
                update['persistent'] = delta_controller.update_mounted(
                    lambda bs: trajectory_loss(source_episode, bs))
                update['verified_source'] = source_arm
            elif args.prototype_memory:
                records = records_from_episode(source_episode)
                sfresh = contextual_source_fields(agent, tok, records, args.device,
                                                   2048, pooling='both')
                tfresh = contextual_text_fields(agent, tok,
                    task_context_text(source_episode['initial_observation'], source_episode['initial_observation']),
                    args.device, 2048, pooling='both')
                with torch.no_grad():
                    agent.set_source(sfresh, target_fields=tfresh)
                    own_basis = [a.b.detach().clone() for a in agent.adapters]
                    agent.set_source(None)
                update['prototype_added'] = prototype_memory.add(key, own_basis)
                update['persistent'] = {"prototype": True}
            elif args.feature_router:
                update['persistent'] = feature_router.update_mounted(
                    key, lambda bs: trajectory_loss(source_episode, bs))
            elif keyed_memory is not None:
                update['persistent'] = keyed_memory.write_mounted_success(
                    key, lambda bs: trajectory_loss(source_episode, bs))
            else:
                old_coeff = controller.state.coefficients.clone()
                old_gate = controller.state.gate_logit.clone()
                # Confirmed own trajectory becomes a new immutable basis vector.
                records = records_from_episode(source_episode)
                sfresh = contextual_source_fields(agent, tok, records, args.device,
                                                   2048, pooling='both')
                tfresh = contextual_text_fields(agent, tok,
                    task_context_text(source_episode['initial_observation'], source_episode['initial_observation']),
                    args.device, 2048, pooling='both')
                with torch.no_grad():
                    agent.set_source(sfresh, target_fields=tfresh)
                    own_basis = [a.b.detach().clone() for a in agent.adapters]
                    agent.set_source(None)
                before_basis_count = len(controller.bases)
                controller.append_basis(own_basis, initial_logit=args.success_basis_logit)
                update['basis_added'] = True
                update['basis_count_before'] = before_basis_count
                update['basis_count_after'] = len(controller.bases)
                update['persistent']=controller.update_mounted(
                    lambda bs: trajectory_loss(source_episode, bs))
            # Utility gate: test candidate and pre-update state on the next
            # reviewed task with the same environment seed before committing.
            if controller is not None and index + 1 < min(args.limit, len(review['targets'])) and args.lookahead_gate:
                nxt = review['targets'][index + 1]
                ngame = args.data_root / nxt['game']
                candidate = [x.detach() for x in controller.matrices()]
                old = [x.detach() for x in controller.bases[0]]
                cseed = 7800000 + index
                cand_ep = run_episode(agent,tok,ngame,{},adapter=False,
                    fixed_adapter=candidate,device=args.device,max_steps=args.max_steps,
                    max_new_tokens=args.max_new_tokens,constrain_actions=True)
                old_ep = run_episode(agent,tok,ngame,{},adapter=False,
                    fixed_adapter=old,device=args.device,max_steps=args.max_steps,
                    max_new_tokens=args.max_new_tokens,constrain_actions=True)
                accepted = cand_ep['reward'] >= old_ep['reward']
                if not accepted:
                    controller.state.coefficients = old_coeff
                    controller.state.gate_logit = old_gate
                update['lookahead']={'next_game':nxt['game'],'candidate_reward':cand_ep['reward'],
                    'pre_update_reward':old_ep['reward'],'accepted':accepted}
        elif persistent['status']=='complete' and persistent['reward'] <= 0 and persistent['trajectory']:
            # Negative evidence uses only actions the actor actually executed.
            if args.failure_residual and keyed_memory is None and not args.feature_router and not args.prototype_memory and not args.hindsight_prototype and not args.anchored_delta and not args.hindsight_basis and not args.signed_composition:
                _, residual = failure_residual(agent, tok, persistent, args.device,
                                                controller.matrices())
                update['failure_residual_scale'] = controller.append_residual_basis(
                    residual, max_fraction=args.residual_fraction,
                    initial_logit=args.failure_residual_logit,
                    orthogonalize=args.orthogonal_residual)
                update['failure_residual_basis'] = True
            if args.success_only_updates:
                update['persistent_failure'] = {"updated": False, "reason": "success_only"}
            elif args.signed_composition:
                update['persistent_failure'] = signed_controller.update_mounted(
                    lambda bs: trajectory_loss(persistent, bs), failure=True)
            elif args.anchored_delta or args.hindsight_basis or args.hindsight_prototype:
                update['persistent_failure'] = delta_controller.update_mounted(
                    lambda bs: trajectory_loss(persistent, bs), failure=True)
            elif args.prototype_memory:
                update['persistent_failure'] = {"prototype": False}
            elif args.feature_router:
                update['persistent_failure'] = feature_router.update_mounted(
                    key, lambda bs: trajectory_loss(persistent, bs),
                    failure=True)
            elif keyed_memory is not None:
                update['persistent_failure'] = keyed_memory.write_mounted_failure(
                    key, lambda bs: trajectory_loss(persistent, bs))
            else:
                update['persistent_failure'] = controller.update_failure_mounted(
                    lambda bs: trajectory_loss(persistent, bs))
        if args.deferred_target_conditioning and successful_episode(base):
            pending_sources[family] = base
            update['deferred_source_written'] = True
            update['deferred_source_family'] = family
        direct_source = direct_write_source(direct_ep, source_episode,
                                           shared=args.shared_write_source)
        if direct_source is not None and not args.deferred_target_conditioning:
            direct_opt.zero_grad()
            dl = trajectory_loss(direct_source, direct)
            dl.backward()
            direct_opt.step()
            update['direct'] = {'loss': float(dl.detach()),
                                'source': source_arm if args.shared_write_source else 'direct'}
        if args.shared_write_source:
            source_digest = hashlib.sha256(json.dumps(
                source_episode, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            update['shared_source_sha256'] = source_digest
            update['shared_source_success'] = successful_episode(source_episode)
        rows.append({'game':target['game'],'base':base,'persistent':persistent,'direct':direct_ep,'frozen':frozen_ep,'update':update})
        # Infrastructure failures are recorded as zero reward rather than
        # aborting the whole matched evaluation during summary construction.
        def reward_of(ep):
            return float(ep.get('reward', 0.0))
        report={'protocol':'ALFWorld pilot; frozen external hypernetwork; reward-gated actual mounted-B updates','checkpoint_sha256':file_hash(args.checkpoint),'external_sha256':filehash(ext_dir/'trained.pt'),'rows':rows,'summary':{'base':sum(reward_of(x['base']) for x in rows),'persistent':sum(reward_of(x['persistent']) for x in rows),'direct':sum(reward_of(x['direct']) for x in rows),'persistent_updates':sum(bool(x['update'].get('persistent') or x['update'].get('deferred_persistent_update')) for x in rows),'failure_residual_bases':sum(bool(x['update'].get('failure_residual_basis')) for x in rows),'memory_slots':len(keyed_memory.entries) if keyed_memory is not None else None,'feature_router_updates':sum(bool(x['update'].get('persistent')) for x in rows) if args.feature_router else 0,'prototype_entries':len(prototype_memory.entries) if prototype_memory is not None else None}}
        report['run_binding'] = run_binding
        report['summary']['frozen'] = sum(reward_of(x['frozen']) for x in rows)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'game':target['game'],'base':reward_of(base),'persistent':reward_of(persistent),'direct':reward_of(direct_ep),'summary':report['summary']}),flush=True)
    return report

if __name__=='__main__':
        p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'));p.add_argument('--checkpoint',type=Path,default=Path('results/trajectory_hyperlora/alfworld_online_own_trajectory_v52b_20261010.pt'));p.add_argument('--review',type=Path,default=Path('data/annotations/alf_own_only_valid_unseen12_reviewed_20261007.json'));p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'));p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-fraction',type=float,default=.35);p.add_argument('--limit',type=int,default=6);p.add_argument('--max-steps',type=int,default=20);p.add_argument('--max-new-tokens',type=int,default=48);p.add_argument('--state-lr',type=float,default=.2);p.add_argument('--direct-lr',type=float,default=.1);p.add_argument('--direct-init-frozen',action='store_true');p.add_argument('--decay',type=float,default=.02);p.add_argument('--anchor-strength',type=float,default=0.0);p.add_argument('--lookahead-gate',action='store_true');p.add_argument('--family-state',action='store_true');p.add_argument('--success-basis-logit',type=float,default=.5);p.add_argument('--failure-residual',action='store_true');p.add_argument('--residual-fraction',type=float,default=.25);p.add_argument('--failure-residual-logit',type=float,default=0.0);p.add_argument('--orthogonal-residual',action='store_true');p.add_argument('--keyed-memory',action='store_true');p.add_argument('--memory-capacity',type=int,default=16);p.add_argument('--memory-temperature',type=float,default=.05);p.add_argument('--memory-slot-similarity',type=float,default=.98);p.add_argument('--feature-router',action='store_true');p.add_argument('--structured-key',action='store_true');p.add_argument('--prototype-memory',action='store_true');p.add_argument('--hindsight-prototype',action='store_true');p.add_argument('--hindsight-composition',action='store_true');p.add_argument('--deferred-target-conditioning',action='store_true');p.add_argument('--prototype-capacity',type=int,default=16);p.add_argument('--prototype-threshold',type=float,default=.25);p.add_argument('--anchored-delta',action='store_true');p.add_argument('--hindsight-basis',action='store_true');p.add_argument('--verified-source',choices=('persistent','base','frozen','any'),default='persistent');p.add_argument('--shared-write-source',action='store_true');p.add_argument('--credit-steps',type=int,default=4);p.add_argument('--credit-discount',type=float,default=.9);p.add_argument('--delta-fraction',type=float,default=.25);p.add_argument('--delta-initial-weight',type=float,default=.25);p.add_argument('--success-only-updates',action='store_true');p.add_argument('--signed-composition',action='store_true');p.add_argument('--signed-bound',type=float,default=1.0);p.add_argument('--tied-signed',action='store_true');p.add_argument('--signed-preserve-l1',action='store_true');p.add_argument('--signed-positive',action='store_true');p.add_argument('--trajectory-kl',action='store_true');p.add_argument('--kl-temperature',type=float,default=1.0);p.add_argument('--kl-teacher',choices=('base','compiled_basis'),default='base');p.add_argument('--deferred-basis-read',choices=('composition','direct'),default='composition');p.add_argument('--deferred-basis-scale',type=float,default=1.0);p.add_argument('--deferred-loss-gate',action='store_true');p.add_argument('--trajectory-write-gate',action='store_true');p.add_argument('--deferred-action-gate',action='store_true');p.add_argument('--external-dir',type=Path,default=ROOT/'results/trajectory_hyperlora/external_hyper_v100_20261010');run(p.parse_args())
