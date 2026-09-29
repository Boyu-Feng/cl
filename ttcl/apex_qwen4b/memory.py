"""APEx category skills and balanced retrieval, with feedback-bound fresh memories."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .common import binding,save,workspace


class MemoryBank:
    def __init__(self,judge,encoder=None):
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.judge=judge;self.torch=torch;self.entries=[];self.vectors=[];self.caption_vectors=[];self.skills={}
        path=workspace()/'models/apex/base/sup-simcse-bert-base-uncased'
        if encoder:self.encoder,self.tokenizer=encoder
        else:
            self.encoder=AutoModel.from_pretrained(path,local_files_only=True).cpu().eval()
            self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)

    def encode(self,text):
        torch=self.torch
        if not text.strip():return np.zeros(self.encoder.config.hidden_size,dtype='float32')
        x=self.tokenizer(text,return_tensors='pt',truncation=True,max_length=512)
        with torch.inference_mode():
            # Upstream encodes one string at a time and mean-pools last hidden states.
            vec=self.encoder(**x).last_hidden_state.mean(1)[0]
            vec=torch.nn.functional.normalize(vec,p=2,dim=-1)
        return vec.numpy()

    def add(self,question,caption,trace,correct,plan='',used=(),memory_id=None):
        category=self.judge.category(question);modality='text-image' if caption else 'text-only'
        summary=self.judge.summary(trace)
        entry={'data_id':memory_id or binding([question,caption,trace]),'question':question,'image_caption':caption,
            'workflow_summary':summary,'judgement':'correct' if correct else 'incorrect','plan':plan,
            'category':category,'modality':modality,'usage_count':1,'success_count':1,'win_rate':1.,
            'source_binding':binding({'question':question,'caption':caption,'public_trace':trace,'feedback_correct':correct})}
        q=self.encode(question);c=self.encode(caption)
        same=[i for i,e in enumerate(self.entries) if (e['category'],e['modality'])==(category,modality)]
        if same:
            scores=np.asarray([.8*np.dot(q,self.vectors[i])+.2*np.dot(c,self.caption_vectors[i]) if caption else np.dot(q,self.vectors[i]) for i in same])
            best=same[int(scores.argmax())]
            if float(scores.max())>=.9999:
                old=self.entries[best]
                if old['judgement']=='incorrect' or len(summary.split())<len(old['workflow_summary'].split()):
                    self.entries[best]=entry;self.vectors[best]=q;self.caption_vectors[best]=c
                return category,modality
        self.entries.append(entry);self.vectors.append(q);self.caption_vectors.append(c)
        for i in used:
            e=self.entries[i];e['usage_count']+=1;e['success_count']+=int(correct);e['win_rate']=round(e['success_count']/e['usage_count'],4)
        return category,modality

    def retrieve(self,question,caption='',exclude_binding=None):
        category=self.judge.category(question);modality='text-image' if caption else 'text-only'
        candidates=[i for i,e in enumerate(self.entries) if e['category']==category and e['modality']==modality
            and binding({'question':e['question'],'caption':e['image_caption']})!=exclude_binding]
        chosen=[]
        if candidates:
            q=self.encode(question);c=self.encode(caption)
            sim=np.asarray([.8*np.dot(q,self.vectors[i])+.2*np.dot(c,self.caption_vectors[i]) if caption else np.dot(q,self.vectors[i]) for i in candidates])
            scores=.7*(sim-sim.min())/(sim.max()-sim.min()+1e-8)+.3*np.asarray([self.entries[i]['win_rate'] for i in candidates])
            for outcome in ['correct','incorrect']:
                choices=[(float(s),i) for s,i in zip(scores,candidates) if self.entries[i]['judgement']==outcome]
                chosen.extend(i for _,i in sorted(choices,reverse=True)[:2])
        context='\n\n'.join(f"[{self.entries[i]['judgement']}] Question: {self.entries[i]['question']}\nWorkflow: {self.entries[i]['workflow_summary']}" for i in chosen)
        skill=self.skills.get(category+'_'+modality)
        if skill and skill.get('win_rate',0)*min(skill.get('evidence_count',0)/10,1)<=.1:skill=None
        return context,skill,chosen,category,modality

    def update_skill(self,category,modality,parsed):
        skill=parsed.get('skill') if isinstance(parsed,dict) else None
        if not isinstance(skill,dict) or parsed.get('operation') not in {'synthesize','refine','create','skip'}:return False
        key=category+'_'+modality
        if parsed['operation']=='skip' and key in self.skills:return False
        if parsed['operation']=='create' and key in self.skills:key+='_v2'
        entries=[e for e in self.entries if e['category']==category and e['modality']==modality]
        skill=dict(skill,category=category,modality=modality,evidence_count=len(entries),
            win_rate=sum(e['judgement']=='correct' for e in entries)/max(1,len(entries)))
        self.skills[key]=skill;return True

    def save(self,path):
        save(path,{'entries':self.entries,'skills':self.skills})

    def load(self,path):
        data=json.loads(Path(path).read_text());self.entries=data['entries'];self.skills=data['skills']
        self.vectors=[self.encode(e['question']) for e in self.entries]
        self.caption_vectors=[self.encode(e['image_caption']) for e in self.entries]
