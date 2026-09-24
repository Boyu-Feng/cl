from __future__ import annotations

import json
import random
import re
import sqlite3

from ttcl.experience_evolution.core import digest, seed


def parse_action(text):
    decoder=json.JSONDecoder()
    for match in re.finditer(r'\{',text):
        try:
            value,_=decoder.raw_decode(text[match.start():])
            if isinstance(value,dict):return value
        except ValueError:pass
    raise ValueError('No JSON object action')


def world(domain,number,split):
    rng=random.Random(seed('world',domain,number,split))
    tag=f'{split}_{number:04}'
    suffix=f'{rng.randrange(10000,99999)}'
    if domain=='sql':
        return {'domain':domain,'id':tag,'table':'stock_'+suffix,'key':'code_'+suffix,
                'category':'group_'+suffix,'amount':'cents_'+suffix,
                'rows':[[f'p{i}', ['red','blue','green'][i%3],rng.randrange(10,2000)] for i in range(18)]}
    return {'domain':'tools','id':tag,'read':'fetch_'+suffix,'convert':'convert_'+suffix,
            'key_arg':'item_'+suffix,'value_arg':'value_'+suffix,'factor':rng.choice([2,3,5,7]),
            'rows':{f'p{i}':rng.randrange(10,900) for i in range(18)}}


def task(world,index):
    domain=world['domain']
    return {'domain':domain,'id':f'{domain}:{world["id"]}:{index}', 'world':world,'index':index,
            'goal': (f'Find the { ["sum","maximum","count"][index%3] } of amounts for category { ["red","blue","green"][(index//3)%3] }.'
                     if domain=='sql' else f'Fetch item p{index%18}, convert its value using this service, and submit the converted integer.')}


class LocalTask:
    """Small fully local environments; all tool execution is bounded and deterministic."""
    def __init__(self,spec):
        self.spec=spec;self.world=spec['world'];self.done=False;self.reward=0.0
        self.calls=0;self.fetched=False;self.converted=False
        self.initial=f'Environment {self.world["id"]}. {spec["goal"]}\n'
        if spec['domain']=='sql':
            w=self.world;self.db=sqlite3.connect(':memory:')
            self.db.execute(f'CREATE TABLE {w["table"]} ({w["key"]} TEXT, {w["category"]} TEXT, {w["amount"]} INTEGER)')
            self.db.executemany(f'INSERT INTO {w["table"]} VALUES (?,?,?)',w['rows'])
            self.initial+='Use JSON {"action":"QUERY","sql":"..."} to inspect SQLite or {"action":"ANSWER","value":number}. Only read-only SELECT and table_info PRAGMA are allowed. You have 5 actions including the answer. A correct answer earns 1; all other completed attempts earn 0.'
            self.expected=self.db.execute(self.solution_query()).fetchone()[0]
            # Restrict capabilities even if the model emits unexpected SQL.
            allowed={sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION}
            def authorize(action,arg1,arg2,db,trigger):
                if action==sqlite3.SQLITE_PRAGMA and str(arg1).lower()=='table_info':return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
            self.db.set_authorizer(authorize)
        else:
            self.expected=self.world['rows'][f'p{spec["index"]%18}']*self.world['factor']
            self.initial+='Use JSON {"tool":"describe","arguments":{}} to discover this service. Invoke tools using their exact names and arguments. Submit with {"tool":"submit","arguments":{"answer":integer}}. Fetch and convert the requested item before submission. You have 5 actions. Full completion earns 1; otherwise 0.'

    def solution_query(self):
        w=self.world;i=self.spec['index'];op=['SUM','MAX','COUNT'][i%3];category=['red','blue','green'][(i//3)%3]
        return f"SELECT {op}({w['amount']}) FROM {w['table']} WHERE {w['category']}='{category}'"

    def step(self,action):
        if self.done:raise RuntimeError('Task already ended')
        self.calls+=1;valid=True
        try:
            if self.spec['domain']=='sql':
                if action.get('action')=='QUERY':
                    query=action.get('sql','')
                    if not isinstance(query,str) or not re.match(r'^\s*(SELECT\b|PRAGMA\s+table_info\s*\()',query,re.I):
                        raise ValueError('Only SELECT or PRAGMA table_info is allowed')
                    cursor=self.db.execute(query);names=[d[0] for d in cursor.description]
                    result=json.dumps({'columns':names,'rows':cursor.fetchmany(30)})
                elif action.get('action')=='ANSWER':
                    self.done=True
                    try:ok=math_equal(action.get('value'),self.expected)
                    except (ValueError,TypeError):ok=False
                    self.reward=float(ok)
                    result=f'CORRECT. Reward 1.' if ok else f'INCORRECT. Submitted {action.get("value")!r}; correct answer {self.expected}. Reward 0.'
                else:raise ValueError('Expected action QUERY or ANSWER')
            else:
                w=self.world;name=action.get('tool');args=action.get('arguments',{})
                if not isinstance(args,dict):raise ValueError('arguments must be an object')
                if name=='describe':
                    result=json.dumps({'environment':w['id'],'tools':{
                        w['read']:{'arguments':{w['key_arg']:'item identifier'},'effect':'fetch requested raw value'},
                        w['convert']:{'arguments':{w['value_arg']:'fetched integer'},'effect':'convert fetched raw value'},
                        'submit':{'arguments':{'answer':'converted integer'},'effect':'submit after fetch and conversion'}}})
                elif name==w['read']:
                    key=args.get(w['key_arg'])
                    if key not in w['rows']:raise ValueError(f'Required argument {w["key_arg"]}; use a valid item identifier')
                    self.last_raw=w['rows'][key];self.fetched=key==f'p{self.spec["index"]%18}';self.converted=False
                    result=json.dumps({'item':key,'raw_value':self.last_raw})
                elif name==w['convert']:
                    if not self.fetched:raise ValueError('Fetch the requested item before conversion')
                    if not math_equal(args.get(w['value_arg']),self.last_raw):raise ValueError(f'Required argument {w["value_arg"]} must equal fetched value')
                    self.converted=True;result=json.dumps({'converted_value':self.expected})
                elif name=='submit':
                    self.done=True;self.reward=float(self.fetched and self.converted and math_equal(args.get('answer'),self.expected))
                    result='CORRECT. Reward 1.' if self.reward else 'INCORRECT. Required successful fetch, conversion and exact converted value. Reward 0.'
                else:raise ValueError('Unknown tool for this environment. Call describe for the current service interface.')
        except (ValueError,TypeError,sqlite3.Error) as exc:
            valid=False;result='ERROR: '+str(exc)
        if self.calls>=5 and not self.done:
            self.done=True;result+='\nAction budget exhausted. Reward 0.'
        return result,valid

    def close(self):
        if self.spec['domain']=='sql':self.db.close()


def math_equal(left,right):
    if isinstance(left,bool) or left is None:return False
    try:return float(left)==float(right)
    except (ValueError,TypeError):return False


def scripted_episode(spec,mode):
    """Execute scripted source actions; these are declared synthetic demonstrations."""
    env=LocalTask(spec);w=spec['world'];events=[]
    if spec['domain']=='sql':
        inspect={'action':'QUERY','sql':"SELECT name,sql FROM sqlite_master WHERE type='table'"}
        if mode in {'correction','scope'}:
            actions=[{'action':'QUERY','sql':'SELECT SUM(price) FROM products'},inspect,
                     {'action':'QUERY','sql':env.solution_query()}, {'action':'ANSWER','value':env.expected}]
        elif mode=='ambiguous':
            actions=[inspect]*5
        else:actions=[inspect,{'action':'QUERY','sql':env.solution_query()},{'action':'ANSWER','value':env.expected}]
    else:
        raw=w['rows'][f'p{spec["index"]%18}']
        good=[{'tool':w['read'],'arguments':{w['key_arg']:f'p{spec["index"]%18}'}},
              {'tool':w['convert'],'arguments':{w['value_arg']:raw}}, {'tool':'submit','arguments':{'answer':env.expected}}]
        if mode in {'correction','scope'}:actions=[{'tool':'fetch','arguments':{'id':f'p{spec["index"]%18}'}}, {'tool':'describe','arguments':{}},*good]
        elif mode=='ambiguous':actions=[{'tool':'describe','arguments':{}}]*5
        else:actions=[{'tool':'describe','arguments':{}},*good]
    # Some correction sources end in an explicit wrong answer despite useful evidence.
    if mode=='correction' and spec['index']%2==0:
        if spec['domain']=='sql':actions[-1]={'action':'ANSWER','value':env.expected+1}
        else:actions[-1]={'tool':'submit','arguments':{'answer':raw}}
    for action in actions:
        observation,valid=env.step(action)
        events.append({'action':json.dumps(action),'observation':observation,'valid_command':valid})
        if env.done:break
    value={'initial_observation':env.initial,'trajectory':events,'reward':env.reward,'steps':len(events)}
    env.close();return value


def run_local(spec,memory,random_seed,client):
    env=LocalTask(spec)
    system='Solve the current task using the described tools. Return exactly one JSON action per turn. Use prior experience only within its stated environment and conditions.'
    if memory:system+='\nPast experience:\n'+memory
    messages=[{'role':'system','content':system}];observation=env.initial;events=[];generations=[]
    try:
        for turn in range(5):
            messages.append({'role':'user','content':observation})
            completion=client.complete(messages,model='frozen-actor',random_seed=seed(random_seed,turn),tokens=256,temperature=.7,top_p=1.0)
            generations.append({'text':completion['raw_response'],'usage':{'prompt_tokens':completion['input_tokens'],
                                'completion_tokens':completion['output_tokens']},'seed':completion['actual_generation_seed']})
            try:action=parse_action(completion['raw_response'])
            except ValueError:action={}
            observation,valid=env.step(action)
            messages.append({'role':'assistant','content':completion['raw_response']})
            events.append({'action':completion['raw_response'],'observation':observation,'valid_command':valid})
            if env.done:break
        return {'game':spec['id'],'memory':memory,'memory_sha256':digest(memory),'seed':random_seed,
                'initial_observation':env.initial,'initial_commands_sha256':digest(env.initial),
                'trajectory':events,'generations':generations,'reward':env.reward,'steps':len(events),
                'actor_adapter_enabled':False,'status':'complete','termination':'success' if env.reward else 'budget_or_wrong_answer'}
    finally:env.close()
