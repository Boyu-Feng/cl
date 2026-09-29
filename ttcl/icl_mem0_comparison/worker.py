from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import time
import traceback
from ttcl.icl_mem0_comparison.protocol import (
    ARMS, Client, NativeMemory, append, normalize_prompt, public_messages, read, save, search_query,
)
from ttcl.structured_memory import run_benchmark as base


def make_task(name, seed):
    if name == 'blind_spectrum_monitoring':
        from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask
        return BlindSpectrumMonitoringTask(seed=seed, schedule='default', response_timeout_seconds=0)
    return base.make_task(name, seed, independent=True)


class System(base.ContinualLearningSystem):
    def __init__(self, plan, client, arm, history, memory, output, brief, index, count):
        self.plan, self.model, self.arm = plan, client, arm
        self.native_memory, self.output, self.brief = memory, output, brief
        self.index, self.count, self.mode = index, count, arm
        self.calls = self.input_tokens = self.output_tokens = self.max_input_tokens = self.turn = 0
        self.last = self.previous = None
        self.public_steps = []
        instruction = ('Solve the current task using its permitted tools. Return exactly one JSON '
            'action matching the schema. Tool results are evidence; your earlier guesses '
            'are not verified facts. Keep each action within the stated budget.')
        if brief:
            instruction += '\n\n' + brief
        self.messages = [{'role': 'system', 'content': instruction}]
        if arm == 'full_history_icl':
            self.messages.extend(copy.deepcopy(history))
        self.prior_message_count = len(self.messages) - 1

    @property
    def name(self):
        return 'qwen4b_' + self.arm

    def reset(self):
        raise RuntimeError('Instances are constructed explicitly; do not erase online memory')

    def respond(self, query):
        prompt = normalize_prompt(query.prompt, self.brief, self.index, self.count)
        content = prompt
        if self.arm == 'mem0':
            retrieval = self.native_memory.search(search_query(prompt, self.previous))
            save(self.output/f'retrieval_{self.turn+1:03}.json', retrieval)
            facts = [m['memory'] for m in retrieval if m.get('memory')]
            if facts:
                content = ('=== RELEVANT MEMORIES ===\n' + '\n'.join('- '+m for m in facts)
                           + '\n=========================\n\n' + content)
        self.turn += 1
        if self.turn > self.plan['max_turns_per_instance']:
            raise RuntimeError('Safety cap exceeded; official outcome unavailable')
        self.messages.append({'role': 'user', 'content': content + '\n\nReturn only JSON. Action schema:\n'
            + json.dumps(query.response_schema.model_json_schema(), ensure_ascii=False, sort_keys=True)})
        for retry in range(1 + self.plan['action_retries']):
            turn_key = self.turn if retry == 0 else f'{self.turn}:format_retry:{retry}'
            random_seed = base.generation_seed(self.plan['task_seed'], query.instance_id, turn_key)
            completion = self.model.generate(copy.deepcopy(self.messages), random_seed)
            self.calls += 1
            self.input_tokens += completion['input_tokens']; self.output_tokens += completion['output_tokens']
            self.max_input_tokens = max(self.max_input_tokens, completion['input_tokens'])
            event = dict(call=self.calls, turn=self.turn, format_retry=retry,
                instance_id=query.instance_id, query=prompt, messages=copy.deepcopy(self.messages), **completion)
            try:
                action, repair = base.normalize_action(completion['raw_response'], query.response_schema)
            except ValueError as exc:
                event['parse_error'] = str(exc); append(self.output/'responses.jsonl', event)
                # Preserve even the final malformed output in full-history ICL.
                self.messages.append({'role': 'assistant', 'content': completion['raw_response']})
                if retry == self.plan['action_retries']:
                    raise
                self.messages.append({'role': 'user', 'content':
                    'Your reply does not match the action schema above. Return one complete JSON '
                    'object with all required fields and the exact nesting shown. Put tool arguments '
                    'alongside the tool field, not inside tool_call_params. Do not change the task '
                    'or invent tool results. This is a formatting retry; no action was executed.'})
                continue
            event.update(action=action.model_dump(), parse_error=None, packaging_repair=repair)
            append(self.output/'responses.jsonl', event)
            self.messages.append({'role': 'assistant', 'content': completion['raw_response']})
            self.last = (prompt, action.model_dump())
            return base.Response(action=action)

    def observe(self, observation, next_query=None):
        complete = base.observation_marks_instance_complete(observation)
        feedback = observation.content
        if self.last is not None:
            prompt, action = self.last
            step = dict(step=self.turn, query=prompt, action=action,
                        public_feedback=feedback, instance_complete=complete)
            self.public_steps.append(step)
            append(self.output/'public_observations.jsonl', step)
            if self.arm == 'mem0':
                update = self.native_memory.add(public_messages(prompt, action, feedback))
                save(self.output/f'memory_update_{self.turn:03}.json', update)
            self.previous = (prompt, action, feedback)
        if feedback:
            self.messages.append({'role': 'user', 'content': 'Tool feedback:\n' + feedback})

    def get_run_artifacts(self):
        return dict(arm=self.arm, prior_message_count=self.prior_message_count,
                    public_steps=self.public_steps)


def run_episode(plan, client, task_name, index, arm, history, memory, output):
    output.mkdir(parents=True, exist_ok=False)
    task = make_task(task_name, plan['task_seed'])
    query = task.reset_baseline_instance(index)
    brief = task.get_agent_brief()
    brief = base.format_task_agent_brief(brief) if brief else ''
    system = System(plan, client, arm, history, memory, output, brief, index, plan['tasks'][task_name])
    recorder = base.Recorder(output, system, 1)
    record = dict(task=task_name, repeat=client.repeat, arm=arm, canonical_index=index,
        episode=index+1, instance_id=query.instance_id, status='failed', reward=None,
        initial_query_sha256=__import__('hashlib').sha256(query.prompt.encode()).hexdigest(),
        prior_message_count=system.prior_message_count)
    writer_before = (memory.calls, memory.input_tokens, memory.output_tokens, len(memory.errors)) if memory else (0,0,0,0)
    try:
        result = base.run_task(task, system, trace_recorder=recorder, show_progress=False,
                               reset_system=False, initial_query=query)
        if len(result.instance_outcomes) != 1:
            raise ValueError('Expected exactly one official outcome')
        outcome = result.instance_outcomes[0]
        record.update(status='complete', reward=float(outcome.reward), success=outcome.success)
    except Exception as exc:
        record.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        connection = getattr(task, '_conn', None)
        if connection is not None:
            connection.close()
    writer_after = (memory.calls, memory.input_tokens, memory.output_tokens, len(memory.errors)) if memory else (0,0,0,0)
    record.update(actor_calls=system.calls, actor_input_tokens=system.input_tokens,
        actor_output_tokens=system.output_tokens, max_input_tokens=system.max_input_tokens,
        elapsed_seconds=recorder.metrics()['elapsed_seconds'],
        writer_calls=writer_after[0]-writer_before[0], writer_input_tokens=writer_after[1]-writer_before[1],
        writer_output_tokens=writer_after[2]-writer_before[2], memory_warnings=writer_after[3]-writer_before[3])
    save(output/'public_trajectory.json', system.public_steps)
    if arm == 'full_history_icl':
        history[:] = system.messages[1:]
        save(output/'history_after.json', history)
    if memory:
        memory.snapshot(output/'memory_after.json')
    events_path = output/'responses.jsonl'
    if events_path.exists():
        first = json.loads(events_path.open().readline())
        record['first_prompt_sha256'] = first['rendered_prompt_sha256']
    save(output/'row.json', record)
    return record


def work(root, task_name, repeat):
    plan = read(root/'plan.json')
    os.chdir(os.environ['TTCL_BENCH'])
    directory = root/'runs'/task_name/str(repeat)
    directory.mkdir(parents=True, exist_ok=False)
    client = Client(plan, repeat)
    memory = NativeMemory(plan, client, directory/'mem0_store')
    history = []
    try:
        for index in range(plan['tasks'][task_name]):
            rows = []
            for arm in ARMS:
                save(directory/'progress.json', dict(phase='running', canonical_index=index,
                    arm=arm, updated_at=time.time()))
                output = directory/arm/f'episode_{index+1:03}'
                row = run_episode(plan, client, task_name, index, arm, history,
                                  memory if arm == 'mem0' else None, output)
                rows.append(row)
                print(json.dumps({k:row.get(k) for k in ['task','repeat','episode','arm','status','reward','error']}), flush=True)
                if row['status'] != 'complete' and not any(s in row.get('error','') for s in
                        ['no schema-valid JSON action', 'Safety cap exceeded', 'Context overflow']):
                    raise RuntimeError(row.get('error', 'Unknown environment or infrastructure failure'))
            if len({(r['instance_id'], r['initial_query_sha256']) for r in rows}) != 1:
                raise ValueError('Paired task identity or public prompt mismatch')
            if index == 0:
                hashes = {r.get('first_prompt_sha256') for r in rows}
                if len(hashes) != 1:
                    raise ValueError('Empty-memory first task prompts must be identical')
        save(directory/'progress.json', dict(phase='complete', cells=plan['tasks'][task_name]*len(ARMS),
                                            updated_at=time.time()))
    except Exception as exc:
        save(directory/'progress.json', dict(phase='failed', error=repr(exc),
            traceback=traceback.format_exc(), updated_at=time.time()))
        raise
    finally:
        memory.close()
