"""Exact E5 / FAISS IP search over pinned wiki25; index stays in host RAM."""
from __future__ import annotations
import argparse
import bisect
import gc
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import numpy as np
from .common import read,save,workspace


class Corpus:
    def __init__(self,paths):
        self.paths=sorted(map(Path,paths));self.starts=[];total=0
        for p in self.paths:self.starts.append(total);total+=p.stat().st_size
        self.total=total

    def lines(self):
        carry=b'';offset=0
        for p in self.paths:
            with p.open('rb') as f:
                while data:=f.read(8*1024**2):
                    pieces=(carry+data).split(b'\n');carry=pieces.pop()
                    for line in pieces:
                        size=len(line)+1
                        if line.strip():yield offset,size,json.loads(line)
                        offset+=size
        if carry.strip():yield offset,len(carry),json.loads(carry)

    def read(self,start,length):
        pieces=[]
        while length:
            i=bisect.bisect_right(self.starts,start)-1
            local=start-self.starts[i];n=min(length,self.paths[i].stat().st_size-local)
            with self.paths[i].open('rb') as f:f.seek(local);pieces.append(f.read(n))
            start+=n;length-=n
        return json.loads(b''.join(pieces))


class Encoder:
    def __init__(self,path,device):
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModel.from_pretrained(path,local_files_only=True).to(device).eval()
        if str(device).startswith('cuda'):self.model.to(dtype=torch.bfloat16)

    def encode(self,texts,prefix):
        t=self.torch
        inputs=self.tokenizer([prefix+x for x in texts],return_tensors='pt',padding=True,
                              truncation=True,max_length=512).to(self.device)
        with t.inference_mode():
            hidden=self.model(**inputs).last_hidden_state.float()
            mask=inputs['attention_mask'][...,None]
            embedding=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            embedding=t.nn.functional.normalize(embedding,p=2,dim=1)
        return embedding.cpu().numpy().astype('float32')

    def cpu(self):
        self.model=self.model.float().cpu();self.device='cpu';gc.collect();self.torch.cuda.empty_cache()


def serve(root,port,device):
    import faiss
    import torch
    torch.set_num_threads(8);faiss.omp_set_num_threads(8)
    spec=read(root/'asset_plan.json');w=workspace()
    paths=[w/spec['wiki']['directory']/x['path'] for x in spec['wiki']['files']]
    progress=read(root/'asset_progress.json')
    done=progress['completed']
    if not all(any(d.get('file')==x['path'] for d in done) for x in spec['wiki']['files']):
        raise RuntimeError('All wiki shards must pass SHA-256 verification before indexing')
    if not any(d.get('repo')=='intfloat/e5-base-v2' for d in done):
        raise RuntimeError('E5 download incomplete')
    encoder=Encoder(w/'models/apex/base/e5-base-v2',device)
    corpus=Corpus(paths);index=faiss.IndexFlatIP(768);offsets=[];lengths=[];batch=[];started=time.time()
    for offset,length,row in corpus.lines():
        offsets.append(offset);lengths.append(length);batch.append(row['contents'])
        if len(batch)==256:
            index.add(encoder.encode(batch,'passage: '));batch=[]
            if index.ntotal%10240==0:
                save(root/'retrieval_progress.json',{'phase':'building_exact_index','documents':index.ntotal,
                    'corpus_bytes_read':offset+length,'corpus_bytes_total':corpus.total,
                    'seconds':time.time()-started,'gpu_device':device})
                print('indexed',index.ntotal,'bytes',offset+length,flush=True)
    if batch:index.add(encoder.encode(batch,'passage: '))
    offsets=np.asarray(offsets,dtype=np.int64);lengths=np.asarray(lengths,dtype=np.int32)
    encoder.cpu();lock=threading.Lock()
    save(root/'retrieval_index.json',{'documents':index.ntotal,'dimension':768,'index':'IndexFlatIP',
        'normalized':True,'persistent_on_disk':False,'note':'Exact index retained in host RAM; no lossy index compression.',
        'wiki_revision':spec['wiki']['revision'],'build_seconds':time.time()-started})
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,data):
            b=json.dumps(data,ensure_ascii=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
        def do_GET(self):self.reply(200,{'ready':True,'documents':index.ntotal})
        def do_POST(self):
            try:
                data=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
                queries=data['queries'];topk=int(data.get('topk',3))
                if not isinstance(queries,list) or not queries or not all(isinstance(q,str) and q.strip() for q in queries):raise ValueError('Invalid queries')
                if not 1<=topk<=10 or len(queries)>10:raise ValueError('Retrieval request exceeds bounds')
                with lock:
                    embeddings=encoder.encode(queries,'query: ');scores,indices=index.search(embeddings,topk)
                result=[]
                for ss,ii in zip(scores,indices):
                    result.append([{'score':float(score),'document':corpus.read(int(offsets[i]),int(lengths[i]))}
                                   for score,i in zip(ss,ii) if i>=0])
                self.reply(200,{'result':result})
            except Exception as e:self.reply(500,{'error':repr(e)})
        def log_message(self,*args):pass
    save(root/'retrieval_ready.json',{'port':port,'documents':index.ntotal,'time':time.time(),'device':'cpu'})
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=18430);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();serve(a.root.resolve(),a.port,a.device)
