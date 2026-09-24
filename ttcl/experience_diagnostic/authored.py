"""Assistant-authored diagnostic candidates; never used as learned-writer code.

These literal notes were composed from the four declared past trajectories.
They are an assisted intervention, not an independently sampled model output.
"""

import json
from pathlib import Path
import re


def entry(kind, title, scope, lesson, application, limitations, episode, steps):
    return dict(type=kind, title=title, scope=scope, lesson=lesson,
                application=application, limitations=limitations,
                evidence=[dict(episode=episode, steps=steps)])


def operation(op, identity, value=None, reason='Retain only the claim supported by the cited public observations.'):
    result = dict(op=op, id=identity, reason=reason)
    if value is not None:
        result['entry'] = value
    return result


def updates():
    return {
        ('database_exploration', 13): dict(
            trajectory_summary='Name guesses failed. sqlite_master exposed the schema. Queries using the g1 statistics table returned empty populations; the submitted zero percentage was explicitly incorrect.',
            reward_interpretation='Reward 0 accompanies an explicitly incorrect answer after ten exploratory queries. It does not validate any attempted zero-review definition or category mapping.',
            decision='UPDATE', operations=[
                operation('ADD', 'E1', entry('procedure', 'Inspect actual table names before category-name guesses', 'This SQLite database, if its schema has not changed.',
                    'Searching table names for electronics, office, and musical returned nothing; unrestricted sqlite_master inspection returned items_g1/g2/g3, fdbk_g1/g2/g3 and auxiliary tables.',
                    "Use SELECT name, sql FROM sqlite_master WHERE type='table' when schema is unknown. Inspect category values separately; table suffixes alone do not establish a category.",
                    'The trajectory did not establish which suffix maps to electronics. Recheck names after a schema change.', 13, [1, 2, 3, 6])),
                operation('ADD', 'E2', entry('fact', 'Separate product and feedback schemas', 'The schema printed in episode 13, step 6.',
                    'items_g2 has prc and prc_usd but no rtg_ct or desc_txt. fdbk_g2 has ref_id, rtg, body, ts and vrf, with a declared ref_id foreign key to items_g2. fdbk_stats_g1 references items_g1 only.',
                    'For review data inspect the matching feedback table; use its declared ref_id link to products. Verify category mapping, value encodings and price units before querying.',
                    'Column names/types do not establish units or vrf value encoding. A g1 statistics table is not evidence about g2/g3 products.', 13, [6])),
                operation('ADD', 'E3', entry('hypothesis', 'Check absence of reviews with a matching-table anti-join', 'Questions about products with no feedback rows, after category and foreign-key checks.',
                    'All attempted zero-review populations were empty, including cross-group matches through fdbk_stats_g1. The final zero-percentage answer was rejected.',
                    'Test NOT EXISTS against the corresponding fdbk table and ref_id, count the resulting denominator, and inspect sample matches before computing percentages.',
                    'This alternative was not executed in the supplied trajectory. Confirm the task definition of zero reviews and key consistency; do not treat it as a verified solution.', 13, [6, 7, 8, 9, 10, 11])),
            ]),
        ('database_exploration', 17): dict(
            trajectory_summary='Schema inspection found items_g3. Its dominant category was Musical Instruments. Grouping str_nm returned one NULL group, and the Unknown answer was rejected.',
            reward_interpretation='The zero reward and explicit incorrect feedback reject the final answer. They do not reveal the correct brand storage location or establish why str_nm is null.',
            decision='UPDATE', operations=[
                operation('ADD', 'E2', entry('fact', 'Observed instrument category and missing str_nm values', 'items_g3 in the database inspected in episode 17.',
                    "Grouping main_cat returned Musical Instruments with count 17162. Restricting to that category and grouping str_nm returned NULL with count 17162. The table list also contains attrs_g3.",
                    'Use the observed category mapping as a starting point if unchanged. Do not use str_nm as a verified brand source for this subset. Inspect other available schemas before choosing a brand field.',
                    'The trajectory did not inspect attrs_g3 columns or establish where brand is stored. NULL values do not establish that brand information is absent everywhere.', 17, [2, 5, 6, 7])),
                operation('ADD', 'E3', entry('procedure', 'Do not infer missing information from an unverified field', 'Unknown-schema questions where a candidate column gives only NULL values.',
                    "The actor treated str_nm as brand without validating its semantics; the query returned NULL and the final Unknown answer was explicitly rejected.",
                    'Before concluding that requested information is unavailable, verify the candidate field meaning and inspect other listed tables. If the source remains unresolved, preserve that uncertainty in memory.',
                    'No successful alternative query was demonstrated. Existing time-based review notes remain separately scoped.', 17, [7, 8])),
            ]),
        ('cohort_studies', 13): dict(
            trajectory_summary='FORGE metadata described 232 Industrial patients. Two CASE partitions produced survival estimates and fits for six observable cohorts. A lower second fit KL was followed by negative overall reward.',
            reward_interpretation='Reward -0.094383 is worse than the stated flat population baseline. The local fit decrease from .0372 to .0298 did not establish a positive population-level gain or causal validity.',
            decision='UPDATE', operations=[
                operation('ADD', 'E1', entry('procedure', 'Use observed survival, not risk-label names', 'When interpreting predict_cohort_survival for a user-defined CASE partition.',
                    "In the first partition, the group called high_risk had S12=.6369 while low_risk had S12=.2520. Those labels were assigned by the actor and do not enforce a risk ordering.",
                    'Read the returned survival curves and the exact CASE conditions together. Do not assign low survival merely because a group name says high_risk.',
                    'The values concern this selected FORGE sample and this partition; they do not establish population ordering or an exposure effect.', 13, [3, 4])),
                operation('ADD', 'E2', entry('procedure', 'Separate observable-cohort fit from population reward', 'Using predict_cohort_survival output to prepare a complete cohort report.',
                    'The tool reported fit for six observable cohorts, listed 28 as unobservable, and mean KL .0372 then .0298. Overall submitted reward was negative despite that decrease.',
                    'Keep track of which exact cohorts were measured and distinguish model estimates from displayed cohort KM. Treat predictions for unobservable cohorts as extrapolation, not validated results.',
                    'This trajectory does not identify an optimal extrapolation method. A trajectory-level score cannot identify the effect of any individual estimate.', 13, [3, 4, 6])),
                operation('ADD', 'E3', entry('fact', 'FORGE coding and enrollment constrain transfer', 'The FORGE sample whose metadata was observed in episode 13.',
                    'Enrollment was workplace-based; all 232 patients were Industrial. genotype_group distinguished GG-1/non-GG-1. Serum VOC used mg/L, ambient VOC used ppb, and Acron usage used hours/week.',
                    'Inspect the next study metadata before reusing these fields or thresholds. Do not identify GG-3 from the non-GG-1 category, or equate serum and ambient exposure.',
                    'No mapping to other studies or population representativeness was established by this episode.', 13, [1, 2])),
            ]),
        ('cohort_studies', 17): dict(
            trajectory_summary='CADENCE metadata exposed PRB1 and MMSE but not the prior FORGE exposure/genotype fields. The actor assumed PRB1 was a genotype proxy, fitted one partition, then submitted a report.',
            reward_interpretation='Reward +.137278 is a positive aggregate gain versus the specified baseline. It does not validate the assumed PRB1/genotype equivalence, every submitted number, or the mechanism inferred from local fit.',
            decision='UPDATE', operations=[
                operation('REVISE', 'E1', entry('procedure', 'Recheck variable meaning before transferring a prior partition', 'Transferring a procedure from an occupational study to another study.',
                    'Current CADENCE metadata lists prb1_ratio, mmse_score and family_history; it does not list genotype_group or voc_mg_l. The actor called PRB1 analogous to genotype without a supporting tool observation.',
                    'Inspect the current metadata. Reuse a prior grouping only if its required variables and meanings are available; otherwise form a new explicitly provisional grouping instead of substituting PRB1 for genotype.',
                    'The absence of a verified proxy mapping does not prove there is no association. No equivalence or calibration was established here.', 17, [1, 2, 3])),
                operation('REMOVE', 'E2', reason='Redundant prior high-risk interpretation; current evidence supplies no genotype-proxy validation. Consolidate the scope check in E1.'),
                operation('REMOVE', 'E3', reason='Duplicates the old high-risk interpretation and retains a zero-survival title despite nonzero numbers. Preserve a scoped current procedure in E1 instead.'),
                operation('REVISE', 'E4', entry('procedure', 'Check per-cohort support before trusting an aggregate fit', 'Reading a predict_cohort_survival result in a new study.',
                    'CADENCE fit covered five cohorts and omitted 31. A cohort with n=2 had KL=.3885 even though mean KL was .0992. The old bank described low KL as biological plausibility or robustness, which this tool output does not establish.',
                    'Inspect the n, model and KM columns per cohort. Separate observed sample fits from unobservable-cohort predictions; flag sparse support rather than claiming robustness from the mean alone.',
                    'No shrinkage rule or universally reliable grouping is established. These current sample values must not be copied as population truths for a different study.', 17, [1, 3, 5])),
            ]),
    }


def write_materials(root):
    for (task, ep), update in updates().items():
        directory = root / 'inputs' / task / str(ep)
        episode = json.loads((directory / 'trajectory.json').read_text())
        by_step = {s['step']: s['public_feedback'] for s in episode['steps']}
        specs = []
        if task == 'database_exploration' and ep == 13:
            for match in re.finditer(r'CREATE TABLE [^\n]+', by_step[6]):
                specs.append({'step': 6, 'text': match.group().rstrip()})
        elif task == 'database_exploration':
            for step in [2, 4, 5, 6, 7]:
                specs.append({'step': step, 'text': by_step[step]})
        else:
            specs.append({'step': 1, 'text': by_step[1]})
            for step in ([3, 4] if ep == 13 else [3]):
                specs.append({'step': step, 'text': by_step[step].split('\nUnobservable cohorts')[0].rstrip()})
        (directory / 'audited_update.json').write_text(json.dumps(update, indent=2, ensure_ascii=False))
        (directory / 'raw_spec.json').write_text(json.dumps(specs, indent=2, ensure_ascii=False))
        audit = {
            'origin': 'Assistant-authored in this conversation from declared past inputs, before any new scoring.',
            'verification': 'Every retained observation manually checked against cited public tool feedback; proposed unexecuted alternatives explicitly marked hypothesis. No semantic oracle used.',
            'history_contamination_limit': 'Researcher has seen earlier reports about this development benchmark; not a blinded writer evaluation.',
            'claim_checks': [{'id': op['id'], 'operation': op['op'], 'reason': op['reason'],
                              'evidence': op.get('entry', {}).get('evidence', []),
                              'limits': op.get('entry', {}).get('limitations', 'Removal of redundancy/unsupported interpretation; not proof that the original observation was false.')}
                             for op in update['operations']],
        }
        (directory / 'semantic_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    import sys
    write_materials(Path(sys.argv[1]).resolve())
