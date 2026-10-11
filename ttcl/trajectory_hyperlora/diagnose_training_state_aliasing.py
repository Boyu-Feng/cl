"""Audit information omitted by the reader query on the training prefix only."""
import json
from pathlib import Path
from collections import defaultdict,Counter
from ttcl.trajectory_hyperlora.prepare_transition_reader_v82 import build,REVIEW,PARENT
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash


def main():
    output=Path('results/trajectory_hyperlora/alfworld_training_state_aliasing_20261010.json')
    if output.exists():raise FileExistsError(output)
    review,targets=build();assert review==json.loads(REVIEW.read_text())
    parent=json.loads(PARENT.read_text())
    groups=defaultdict(list);actors=defaultdict(list)
    for k,(t,transition) in enumerate(zip(targets,review['transitions'],strict=True)):
        groups[digest(transition['key_text'])].append(k)
        actors[digest(t['actor_messages'])].append(k)
    def conflicts(mapping):
        result=[]
        for key,indices in mapping.items():
            if len({targets[k]['target_action'] for k in indices})<2:continue
            result.append({'input_sha256':key,'source_indices':sorted({targets[k]['stream_index'] for k in indices}),
                'targets':[{'target_index':k,'step':targets[k]['target_step'],'action':targets[k]['target_action'],
                            'binding':targets[k]['input_content_sha256']} for k in indices]})
        return result
    reader_conflicts=conflicts(groups);actor_conflicts=conflicts(actors)
    episodes=[]
    for row in parent['rows'][:120]:
        ep=row['episode'];seen={};repeats=[];before=ep['initial_observation']
        for step,t in enumerate(ep['trajectory']):
            # Same complete immediate observation and command, not just same location.
            key=digest([before,t['command']])
            if key in seen:repeats.append({'step':step,'previous_step':seen[key],'distance':step-seen[key]})
            seen[key]=step;before=t['observation']
        episodes.append({'index':row['index'],'reward':ep['reward'],'steps':ep['steps'],
            'repeat_observation_action_pairs':len(repeats),
            'repeats_beyond_two_action_history':sum(x['distance']>2 for x in repeats),
            'repeat_positions':repeats})
    def summarize(rows):
        return {'n':len(rows),'mean_steps':sum(x['steps'] for x in rows)/max(1,len(rows)),
            'episodes_with_repeats':sum(x['repeat_observation_action_pairs']>0 for x in rows),
            'episodes_with_long_gap_repeats':sum(x['repeats_beyond_two_action_history']>0 for x in rows),
            'repeat_pairs':sum(x['repeat_observation_action_pairs'] for x in rows),
            'long_gap_repeat_pairs':sum(x['repeats_beyond_two_action_history'] for x in rows)}
    result={'parent_report_sha256':file_hash(PARENT),'review_sha256':file_hash(REVIEW),'runner_sha256':file_hash(Path(__file__)),
        'training_target_count':len(targets),'reader_conflicting_groups':reader_conflicts,
        'actor_conflicting_groups':actor_conflicts,'episodes':episodes,
        'summary':{'reader_conflicts':len(reader_conflicts),'actor_conflicts':len(actor_conflicts),
                   'success':summarize([x for x in episodes if x['reward']==1]),
                   'failure':summarize([x for x in episodes if x['reward']==0])},
        'scope':'Audited first120 training tasks only. Reader queries omit action history; actor prompts checked separately. Identical immediate observations do not imply identical hidden environment state. Repeated pairs and label conflicts do not prove harmful loops or causal performance loss.'}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary']))

if __name__=='__main__':main()
