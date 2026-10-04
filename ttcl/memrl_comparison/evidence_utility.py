"""Conservative online utility gate for model-selected trajectory evidence.

Only a paired task from the same native-memory state can credit one evidence
card. A source is activated after wins on two distinct target contents, with
no observed loss. Repeats on one target diagnose sampling noise but cannot
manufacture independent transfer evidence.
"""
from __future__ import annotations

import math


class PairedUtilityGate:
    def __init__(self):
        self.observations = {}
        self.seen = set()

    def observe(self, *, source_content_sha256, target_input_sha256,
                target_domain, actor_seed, native_state_sha256,
                paired_native_state_sha256, baseline_reward,
                with_evidence_reward, shown_card_ids):
        if (not all(isinstance(x, str) and len(x) == 64 for x in
                    (source_content_sha256, target_input_sha256,
                     native_state_sha256, paired_native_state_sha256)) or
                not isinstance(target_domain, str) or not target_domain or
                type(actor_seed) is not int):
            raise ValueError('Missing input/content/state binding')
        if native_state_sha256 != paired_native_state_sha256:
            raise ValueError('Cannot assign card credit across different native memory states')
        if shown_card_ids != [source_content_sha256]:
            raise ValueError('Card-level credit needs exactly one exposed card')
        if any(not isinstance(x, (int, float)) or not math.isfinite(x)
               for x in (baseline_reward, with_evidence_reward)):
            raise ValueError('Official finite paired rewards required')
        key = (source_content_sha256, target_domain)
        unique = (key, target_input_sha256, actor_seed)
        if unique in self.seen:
            raise ValueError('Duplicate paired observation')
        self.seen.add(unique)
        delta = float(with_evidence_reward) - float(baseline_reward)
        sign = 1 if delta > 0 else -1 if delta < 0 else 0
        record = dict(source_content_sha256=source_content_sha256,
            target_input_sha256=target_input_sha256,
            target_domain=target_domain, actor_seed=actor_seed,
            native_state_sha256=native_state_sha256,
            baseline_reward=float(baseline_reward),
            with_evidence_reward=float(with_evidence_reward),
            delta=delta, sign=sign)
        self.observations.setdefault(key, []).append(record)
        return dict(observation=record, status=self.status(
            source_content_sha256, target_domain))

    def status(self, source_content_sha256, target_domain):
        rows = self.observations.get((source_content_sha256, target_domain), [])
        wins = {row['target_input_sha256'] for row in rows if row['sign'] > 0}
        losses = {row['target_input_sha256'] for row in rows if row['sign'] < 0}
        if losses:
            state = 'quarantine'
        elif len(wins) >= 2:
            state = 'active'
        else:
            state = 'probation'
        return dict(state=state, independent_win_targets=len(wins),
                    independent_loss_targets=len(losses), paired_trials=len(rows))
