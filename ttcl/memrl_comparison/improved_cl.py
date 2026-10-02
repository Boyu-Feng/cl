"""Task-aware, public-evidence projection for the CLBench MemRL adaptation.

The original MemRL store and Q updater are retained. The added structured
state and action projection use only prior public inputs, tool observations
and raw submissions. Native MemRL retrieval/Q still follow the original
reward-aware adaptation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from ttcl.icl_mem0_comparison.protocol import read, save
from .memory import Memory, digest


PEAK = re.compile(
    r"\| freq: (?P<freq>[0-9.]+) MHz \| power: (?P<power>-?[0-9.]+) dBm"
    r" \| width: (?P<width>[0-9.]+) MHz"
)
STUDY = re.compile(r"## Study \d+/\d+: ([^\n]+)")
REGION = re.compile(r"\*\*Region\(s\) in this dataset:\*\* ([^\n]+)")
GROUP = re.compile(
    r"^\s*([^:\n]+): n=(\d+) \([^\n]*?\)\s+S\(12m\)=([0-9.]+)"
    r"\s+S\(24m\)=([0-9.]+)\s+S\(36m\)=([0-9.]+)", re.M
)


class ImprovedCLMemory(Memory):
    """MemRL with a bounded, evidence-only projection for two CLBench tasks."""

    def __init__(self, plan, client, directory, calibration, task, embedder=None):
        if task not in {'blind_spectrum_monitoring', 'cohort_studies'}:
            raise ValueError(task)
        self.task = task
        self.scan_index = 0
        self.scan_registry = []
        self.cohort_evidence = []
        self.cohort_reports = []
        self.pending_actor_report = None
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='public_evidence_v3', task=task))

    def _tokens(self, value):
        return len(self.client.tokenizer.encode(value, add_special_tokens=False))

    def _append_bounded(self, context, entry):
        proposed = (context + '\n' + entry).strip()
        return proposed if self._tokens(proposed) <= self.plan['memory_tokens'] else context

    def _scan_context(self):
        min_observations = max(2, math.ceil(.05 * self.scan_index))
        recurrent = [r for r in self.scan_registry if r['count'] >= min_observations]
        recurrent.sort(key=lambda r: (-r['count'], -r['last_seen'], r['center']))
        lines = [
            'Historical public scans (observations, not ground truth).',
            'The report must cover persistent transmitters, including credible'
            ' channels absent from the current scan. Keep credible current peaks;'
            ' use prior recurrences to add dormant channels with currently_active=false.',
        ]
        for r in recurrent[:30]:
            lines.append(f"{r['center']:.1f} MHz, width {r['width']:.1f} MHz,"
                         f" seen {r['count']} times, last prior scan {r['last_seen']}.")
        return '\n'.join(lines) if recurrent else ''

    def _cohort_context(self):
        if not self.cohort_evidence:
            return ''
        lines = [
            'Earlier studies: public tool measurements, not population truth or grader feedback.',
            'Study sampling can be biased. Use these as cross-study evidence and'
            ' still submit all 36 required cohorts, including unobservable ones.',
        ]
        # One recent study per stage preserves cross-study coverage in a small budget.
        latest = {}
        for item in self.cohort_evidence:
            latest[item['study']] = item
        for item in sorted(latest.values(), key=lambda x: x['index']):
            lines.append(f"Study {item['study']}, region {item['region']}:"
                         f" groups defined by {item['expression']};"
                         f" observed S(12/24/36): {item['groups']}.")
        return '\n'.join(lines)

    def retrieve(self, query):
        # On Cohort, the first four (same-stage) prompts produced an unusably
        # strict absolute threshold. Keep native similarity/Q ranking but allow
        # cross-stage candidates; the actor sees only the evidence projection.
        original = self.service.rl_config.sim_threshold
        if self.task == 'cohort_studies':
            self.service.rl_config.sim_threshold = 0.
        try:
            base = super().retrieve(query)
        finally:
            self.service.rl_config.sim_threshold = original
        context = self._scan_context() if self.task == 'blind_spectrum_monitoring' else self._cohort_context()
        if self._tokens(context) > self.plan['memory_tokens']:
            raise ValueError('Public evidence exceeds the declared memory budget')
        # The native memory is still queried and updated. Expose one short,
        # Q-ranked lesson only when it was generated from a successful episode;
        # failure reflections in these tasks often assert unsupported causes.
        ids = []
        for mid in base['selected_before_budget']:
            meta = self.store.get(mid).metadata.model_dump()
            if meta.get('success') is not True:
                continue
            lesson = str(meta.get('public_abstract') or '').strip()
            if not lesson:
                continue
            snippet = 'Prior successful strategy (verify against current evidence): ' + lesson[:900]
            proposed = self._append_bounded(context, snippet)
            if proposed != context:
                context = proposed
                ids.append(mid)
            break
        return dict(base, context=context, ids=ids, tokens=self._tokens(context),
                    dropped_whole_entries=[mid for mid in base['selected_before_budget'] if mid not in ids],
                    context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                    projection='public_evidence_v3')

    def _record_scan(self, query):
        self.scan_index += 1
        for match in PEAK.finditer(query):
            freq, width = float(match['freq']), float(match['width'])
            candidates = [r for r in self.scan_registry
                          if r['last_seen'] < self.scan_index
                          if abs(r['center'] - freq) <= max(2.5, .25 * r['width'])
                          and abs(r['width'] - width) <= max(2., .5 * r['width'])]
            if candidates:
                r = min(candidates, key=lambda x: abs(x['center'] - freq))
                n = r['count']
                r['center'] = (n * r['center'] + freq) / (n + 1)
                r['width'] = (n * r['width'] + width) / (n + 1)
                r['count'] += 1
                r['last_seen'] = self.scan_index
            else:
                self.scan_registry.append(dict(center=freq, width=width, count=1,
                                               last_seen=self.scan_index))

    def decorate_system(self, system):
        original_respond = system.respond

        if self.task == 'cohort_studies':
            def respond_with_prior_predictions(query):
                response = original_respond(query)
                actor_action = response.action.model_dump()
                if not actor_action or any(not key.endswith(('__s12', '__s24', '__s36'))
                                           for key in actor_action):
                    return response
                # Prior raw submissions are a public observation of the
                # frozen actor's beliefs. Averaging across studies reduces
                # dependence on the current study's selection bias.
                report = ({key: sum(float(prior[key]) for prior in self.cohort_reports) /
                           len(self.cohort_reports) for key in actor_action}
                          if self.cohort_reports else actor_action)
                action = query.response_schema.model_validate(report)
                if system.last is None:
                    raise RuntimeError('No actor action recorded for cohort aggregation')
                system.last = (system.last[0], action.model_dump())
                self.pending_actor_report = actor_action
                save(system.output / 'policy_action.json', dict(actor=actor_action,
                     final=action.model_dump(), prior_report_count=len(self.cohort_reports)))
                return type(response)(action=action, metadata=response.metadata)

            system.respond = respond_with_prior_predictions
            return

        def respond_with_history(query):
            response = original_respond(query)
            actor_action = response.action.model_dump()
            transmitters = list(actor_action['transmitters'])
            current = []
            for match in PEAK.finditer(query.prompt):
                peak = dict(center_freq=float(match['freq']),
                            bandwidth=float(match['width']), currently_active=True,
                            estimated_power=float(match['power']))
                current.append(peak)
            added_current, added_history = [], []
            for peak in current:
                if any(abs(float(t['center_freq']) - peak['center_freq']) <= 1.0
                       for t in transmitters):
                    continue
                transmitters.append(peak)
                added_current.append(peak)
            min_observations = max(2, math.ceil(.05 * self.scan_index))
            for item in self.scan_registry:
                if item['count'] < min_observations:
                    continue
                if any(abs(float(t['center_freq']) - item['center']) <=
                       max(2.5, .25 * item['width']) for t in transmitters):
                    continue
                prior = dict(center_freq=round(item['center'], 2),
                             bandwidth=round(item['width'], 2),
                             currently_active=False, estimated_power=-50.)
                transmitters.append(prior)
                added_history.append(prior)
            transmitters.sort(key=lambda x: float(x['center_freq']))
            action = query.response_schema.model_validate(dict(transmitters=transmitters))
            if system.last is None:
                raise RuntimeError('No actor action recorded for spectrum augmentation')
            system.last = (system.last[0], action.model_dump())
            save(system.output / 'policy_action.json', dict(actor=actor_action,
                 final=action.model_dump(), added_current=added_current,
                 added_history=added_history, prior_scan_count=self.scan_index))
            return type(response)(action=action, metadata=response.metadata)

        system.respond = respond_with_history

    def _record_cohort(self, query, public_trace):
        match = STUDY.search(query)
        if not match:
            raise ValueError('Cohort study marker absent from public input')
        region = REGION.search(query)
        trace = json.loads(public_trace)
        for step in reversed(trace):
            call = step.get('action', {}).get('tool_call', {})
            if call.get('tool') != 'estimate_survival_by_group':
                continue
            feedback = step.get('public_feedback') or ''
            groups = GROUP.findall(feedback)
            if not groups:
                continue
            compact = '; '.join(f'{name.strip()} n={n}: {s12}/{s24}/{s36}'
                                for name, n, s12, s24, s36 in groups[:5])
            self.cohort_evidence.append(dict(index=len(self.cohort_evidence) + 1,
                                             study=match.group(1).strip(),
                                             region=region.group(1).strip() if region else 'unknown',
                                             expression=str(call.get('group_expression') or '')[:240],
                                             groups=compact))
            return

    def update(self, query, public_trace, reward, success, retrieval, binding):
        result = super().update(query, public_trace, reward, success, retrieval, binding)
        if self.task == 'blind_spectrum_monitoring':
            self._record_scan(query)
        else:
            self._record_cohort(query, public_trace)
            if self.pending_actor_report is not None:
                self.cohort_reports.append(self.pending_actor_report)
                self.pending_actor_report = None
        return result

    def snapshot(self, path):
        super().snapshot(path)
        data = read(path)
        data['public_evidence'] = dict(task=self.task, scan_index=self.scan_index,
                                       scan_registry=self.scan_registry,
                                       cohort_evidence=self.cohort_evidence,
                                       cohort_reports=self.cohort_reports)
        save(path, data)

    def restore(self, path):
        super().restore(path)
        data = read(path).get('public_evidence')
        if not data or data.get('task') != self.task:
            raise ValueError('Improved CL memory state missing or task changed')
        self.scan_index = data['scan_index']
        self.scan_registry = data['scan_registry']
        self.cohort_evidence = data['cohort_evidence']
        self.cohort_reports = data['cohort_reports']
        self.pending_actor_report = None
