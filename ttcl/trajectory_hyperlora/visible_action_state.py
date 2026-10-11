"""Deterministic summary of actions already executed in the current attempt.

Accept only the visible prefix, never a full episode plus a target step. This
adds no inferred state, outcome label, legal-action filter or next-action hint.
Callers must bind the prefix and account for the resulting prompt tokens.
"""
import json


def action_state_text(completed_transitions):
    counts={};order=[]
    for transition in completed_transitions:
        command=transition['command']
        if not isinstance(command,str) or not command:raise ValueError('Invalid past command')
        if command not in counts:order.append(command);counts[command]=0
        counts[command]+=1
    # JSON escaping preserves command boundaries and literal text exactly.
    state={'executed_commands':[{'command':c,'count':counts[c]} for c in order],
           'last_four_commands':[x['command'] for x in completed_transitions[-4:]]}
    return 'Actions already executed in this attempt:\n'+json.dumps(state,ensure_ascii=False,separators=(',',':'))
