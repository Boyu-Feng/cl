from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace
from ttcl.icl_mem0_comparison.protocol import append, read, save, seed, sha
from .upstream import TextualMemoryItem, load


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


class LocalStore:
    def __init__(self):
        self.items = {}

    def add(self, items):
        for item in items:
            # IDs only identify storage entries, never supply labels.
            item.id = f'memory_{len(self.items):06d}'
            self.items[item.id] = copy.deepcopy(item)

    def get(self, memory_id):
        return copy.deepcopy(self.items[memory_id])

    def update(self, memory_id, data):
        if memory_id not in self.items or data['id'] != memory_id:
            raise ValueError('Invalid memory update identity')
        self.items[memory_id] = TextualMemoryItem.model_validate(data)


class Embedder:
    """BGE-M3 CLS pooling; explicit chunking instead of silent truncation."""
    def __init__(self, path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval().cpu()
        self.cache = {}

    def embed(self, texts):
        pending=[];vectors={}
        for text in dict.fromkeys(texts):
            if text in self.cache:continue
            ids=self.tokenizer.encode(text,add_special_tokens=False)
            chunks=[ids[i:i+8190] for i in range(0,len(ids),8190)] or [[]]
            vectors[text]=[]
            pending.extend((text,self.tokenizer.prepare_for_model(chunk)) for chunk in chunks)
        while pending:
            batch_items=[pending.pop(0)]
            while pending and len(batch_items)<8 and max(
                    len(x[1]['input_ids']) for x in batch_items+[pending[0]])*(len(batch_items)+1)<=8192:
                batch_items.append(pending.pop(0))
            batch=self.tokenizer.pad([x[1] for x in batch_items],padding=True,return_tensors='pt')
            batch={k:v for k,v in batch.items() if k in {'input_ids','attention_mask'}}
            with self.torch.no_grad():
                encoded=self.model(**batch).last_hidden_state[:,0]
                encoded=self.torch.nn.functional.normalize(encoded,dim=-1)
            for (text,_),vector in zip(batch_items,encoded):vectors[text].append(vector)
        for text,chunks in vectors.items():
            self.cache[text]=self.torch.nn.functional.normalize(self.torch.stack(chunks).mean(0),dim=0).tolist()
        return [self.cache[text] for text in texts]


class Memory:
    def __init__(self, plan, client, directory, calibration, embedder=None):
        self.plan, self.client, self.directory = plan, client, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        upstream, hashes = load(plan['upstream'])
        save(self.directory/'upstream_bindings.json', hashes)
        self.signature=digest(dict(upstream=hashes,adapter_sha256=sha(Path(__file__)),rl_config=plan['rl_config'],calibration=calibration,
            memory_tokens=plan['memory_tokens'],writer_max_tokens=plan['writer_max_tokens'],
            q_min_threshold=plan.get('q_min_threshold',-10),model=plan.get('model'),embedding=plan.get('embedding'),repeat=client.repeat))
        self.calls = self.input_tokens = self.output_tokens = 0
        self.limit_hits=0
        self.errors, self.abstracts = [], []
        owner = self
        class Writer(upstream.BaseLLM):
            default_temperature=0.
            default_max_tokens=plan['writer_max_tokens']
            generate_script=upstream.generate_script
            def generate(self, messages, **kwargs):
                try:
                    result = owner.client.complete(messages, seed(client.repeat,'memrl_writer',owner.calls),
                        tokens=plan['writer_max_tokens'], temperature=kwargs.get('temperature',0.),top_p=1.)
                    owner.calls += 1
                    owner.input_tokens += result['input_tokens']; owner.output_tokens += result['output_tokens']
                    append(owner.directory/'writer.jsonl', dict(messages=messages, **result))
                    if not result['raw_response'].strip():
                        raise ValueError('Empty MemRL writer output')
                    # Native provider returns text at its token cap. Keep that
                    # budgeted output and flag it; never retry with extra tokens.
                    owner.last_limit_hit=result['finish_reason']=='length'
                    owner.limit_hits+=int(owner.last_limit_hit)
                    owner.abstracts.append(result['raw_response'])
                    return result['raw_response']
                except Exception as exc:
                    owner.errors.append(repr(exc)); raise
            def extract_keywords(self, *args, **kwargs):
                raise RuntimeError('Query retrieval does not use keyword extraction')
        self.store = LocalStore()
        mos = SimpleNamespace(mem_cubes={'local':SimpleNamespace(text_mem=self.store)},
            get=lambda **kw:self.store.get(kw['memory_id']))
        service = upstream.MemoryService.__new__(upstream.MemoryService)
        cfg = upstream.RLConfig(**plan['rl_config'])
        cfg.q_min_threshold = plan.get('q_min_threshold',-10)
        cfg.sim_threshold = calibration['threshold']
        service.__dict__.update(mos=mos, user_id='benchmark', default_cube_id='local',
            strategy_config=upstream.StrategyConfiguration.main_combination(),
            num_workers=1, llm_provider=Writer(), embedding_provider=embedder or Embedder(plan['embedding']),
            memory_confidence=100., add_similarity_threshold=.9, enable_value_driven=True,
            rl_config=cfg, sim_norm_mean=calibration['mean'], sim_norm_std=calibration['std'],
            use_z_score_normalization=True, dedup_by_task_id=False, weight_sim=.5, weight_q=.5,
            _mem_cache={}, _q_cache={}, _mem_cache_max_size=10000, _q_cache_max_size=1000000,
            _db_gate=threading.BoundedSemaphore(1), dict_memory={}, query_embeddings={},
            _q_updater=upstream.QValueUpdater(mos,'benchmark',cfg,default_cube_id='local'))
        self.service = service

    def retrieve(self, query):
        value = self.service.retrieve_query(query,k=5,threshold=self.service.rl_config.sim_threshold)
        result, similarities = value if isinstance(value,tuple) else (value,[])
        context, ids, dropped = '', [], []
        candidates = []
        for item in result['candidates']:
            candidates.append({k:item[k] for k in ('memory_id','similarity','q_estimate','score')})
        for item in result['selected']:
            md = self.store.get(item['memory_id']).metadata.model_dump()
            # Budget-matched adaptation: scripts/reflections are exposed; the
            # native full trajectory is kept in storage and audit snapshots.
            text = 'Task: '+md['task_description']+'\nExperience: '+md['public_abstract']
            proposed = (context+'\n\n'+text).strip()
            if len(self.client.tokenizer.encode(proposed,add_special_tokens=False)) > self.plan['memory_tokens']:
                dropped.append(item['memory_id']); continue
            context = proposed; ids.append(item['memory_id'])
        return dict(query=query, context=context, ids=ids, candidates=candidates,
            selected_before_budget=result['actions'], dropped_whole_entries=dropped,
            similarities=similarities, tokens=len(self.client.tokenizer.encode(context,add_special_tokens=False)),
            context_sha256=hashlib.sha256(context.encode()).hexdigest())

    def update(self, query, public_trace, reward, success, retrieval, binding):
        if not isinstance(reward,(int,float)) or not math.isfinite(reward):
            raise ValueError('No Q updates without an official finite reward')
        if retrieval['query'] != query:
            raise ValueError('Retrieval/update query mismatch')
        before = len(self.errors)
        updates = {}
        for mid in retrieval['ids']:
            new = self.service.update_value(mid,float(reward))
            if new is None:
                raise RuntimeError('Upstream swallowed a Q update error')
            self.service._q_cache[mid] = new
            updates[mid] = new
        self.abstracts = []
        results = self.service.add_memories([query],[public_trace],
            [success if success is not None else True],
            retrieved_memory_queries=[retrieval['similarities']],
            retrieved_memory_ids_list=[retrieval['ids']],
            metadatas=[{'success':success,'official_reward':reward,'input_binding':binding,
                        'public_trace_sha256':hashlib.sha256(public_trace.encode()).hexdigest()}])
        if len(self.errors)!=before or not results or len(results)!=1 or not results[0][1] or len(self.abstracts)!=1:
            raise RuntimeError(f'MemRL memory write failed: {self.errors[before:]}')
        mid = results[0][1]
        item = self.store.get(mid).model_dump()
        item['metadata']['public_abstract'] = self.abstracts[0]
        item['metadata']['writer_token_limit_hit']=self.last_limit_hit
        self.store.update(mid,item)
        if any(mid not in self.store.items for group in self.service.dict_memory.values() for mid in group):
            raise RuntimeError('Invalid upstream query index')
        return dict(new_memory_id=mid,q_updates=updates,input_binding=binding)

    def snapshot(self, path):
        save(path,dict(protocol_signature=self.signature,items={k:v.model_dump() for k,v in self.store.items.items()},
            dict_memory=self.service.dict_memory, query_embeddings=self.service.query_embeddings,
            q_cache=self.service._q_cache, calls=self.calls,input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,limit_hits=self.limit_hits,errors=self.errors))

    def restore(self, path):
        state = read(path)
        if state.get('protocol_signature')!=self.signature:
            raise ValueError('Memory snapshot protocol or seed changed')
        self.store.items = {k:TextualMemoryItem.model_validate(v) for k,v in state['items'].items()}
        self.service.dict_memory = state['dict_memory']; self.service.query_embeddings=state['query_embeddings']
        self.service._q_cache = state['q_cache']; self.service._mem_cache.clear()
        for key in ('calls','input_tokens','output_tokens','limit_hits','errors'):
            setattr(self,key,state[key])
