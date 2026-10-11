"""Train a trajectory-to-LoRA utility hypernetwork with paired future-action ranking."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ttcl.trajectory_hyperlora.direct_composition_pilot import DirectRelationHyperLoRA
from ttcl.trajectory_hyperlora.joint_domains_v93 import MODEL, digest, filehash, read, write

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results/trajectory_hyperlora/external_clbench_corpus_v100_20261010"
OUT = ROOT / "results/trajectory_hyperlora/external_utility_hyper_v151_20261011"

def main(device, steps=None, lr=1e-4, b_penalty=1e-2, anchor_weight=0.1,
         data_dir=None, out_root=None):
    global OUT, DATA_DIR
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    if out_root is not None:
        OUT = Path(out_root)
    if steps is not None:
        OUT = OUT.with_name(f"{OUT.name}_steps{steps}_lr{lr:g}_bp{b_penalty:g}_aw{anchor_weight:g}")
    else:
        OUT = OUT.with_name(f"{OUT.name}_full_lr{lr:g}_bp{b_penalty:g}_aw{anchor_weight:g}")
    if (OUT / "complete.json").exists(): raise RuntimeError("refusing overwrite")
    data = read(DATA_DIR / "dataset.json"); manifest = read(DATA_DIR / "manifest.json")
    assert digest(data) == manifest["dataset_sha256"]
    if (manifest.get('status', '').startswith('candidate_') or
            manifest.get('protocol') == 'external_paired_return_v158'):
        raise ValueError('Unreviewed candidate supervision: see causal-pair audit v161; '
                         'a copied review status does not validate action utility')
    if (OUT / "complete.json").exists(): raise RuntimeError("refusing overwrite frozen utility run")
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "protocol.json", {"protocol":"external_utility_hyper_v151_norm_anchor", "dataset_sha256":manifest["dataset_sha256"], "manifest_sha256":filehash(DATA_DIR/"manifest.json"), "frozen_before_transfer":True, "updates_per_arm":len(data["schedule"]), "objective":"paired future-action ranking with adapter norm and base-behavior anchoring; negatives are deterministic answer permutations", "b_penalty":b_penalty, "anchor_weight":anchor_weight, "target_datasets_read":manifest.get("target_datasets_read", []), "target_datasets_forbidden":manifest.get("target_datasets_forbidden", [])})
    torch.manual_seed(10002); torch.cuda.manual_seed_all(10002)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device)
    agent = DirectRelationHyperLoRA(model, rank=4, layers=2, encoder_kind="contextual", task_conditioned=True).to(device); agent.eval()
    hyper = {n:p for n,p in agent.named_parameters() if p.requires_grad}
    ha = [a.a for a in agent.adapters]
    sa = [torch.nn.Parameter(a.detach().clone()) for a in ha]
    sb = [torch.nn.Parameter(torch.zeros(1,a.base.out_features,a.rank,device=device)) for a in agent.adapters]
    cache={}
    def enc(text):
        k=digest(text)
        if k not in cache:
            ids=tok(text,add_special_tokens=False).input_ids; used=ids if len(ids)<=2048 else ids[:1024]+ids[-1024:]
            agent.set_source(None)
            with torch.no_grad(): h=model(input_ids=torch.tensor([used],device=device),output_hidden_states=True,use_cache=False).hidden_states[-1]
            cache[k]={"contextual":h[:,-1].float().detach(),"pair_contextual":h.float().mean(1).detach()}
        return k
    prepared=[]
    answers = [tok(row["answer"] + tok.eos_token, add_special_tokens=False).input_ids
               for row in data["rows"]]
    for i, row in enumerate(data["rows"]):
        pre=tok.apply_chat_template(row["messages"],tokenize=True,add_generation_prompt=True)
        # Paired-return corpora provide an exact-query failure action.  The
        # old corpus has no such label and falls back to a deterministic
        # permutation only for diagnostic compatibility.
        negative = (tok(row["negative"] + tok.eos_token,
                        add_special_tokens=False).input_ids
                    if "negative" in row else answers[(i + 1) % len(answers)])
        if negative == answers[i]:
            negative = answers[(i + 2) % len(answers)]
        prepared.append({"prefix":pre,"answer":answers[i],"negative":negative,
                         "source":enc(row["source"]),"query":enc(row["query"])})
    torch.save({"fields":cache,"prepared":prepared},OUT/"cache.pt")
    oh=torch.optim.AdamW(list(hyper.values()),lr=lr,weight_decay=0); ol=torch.optim.AdamW(sa+sb,lr=lr,weight_decay=0); history=[]; started=time.time()
    schedule = data["schedule"] if steps is None else data["schedule"][:steps]
    for step,idx in enumerate(schedule,1):
        r=prepared[idx]; losses={}
        def sequence_score(answer):
            ids=torch.tensor([r["prefix"]+answer],device=device)
            logits=model(input_ids=ids,use_cache=False).logits
            token_logits=logits[:,len(r["prefix"])-1:-1]
            logp=torch.log_softmax(token_logits,dim=-1)
            target=torch.tensor(answer,device=device).view(1,-1)
            return logp.gather(-1,target.unsqueeze(-1)).mean()
        # Measure the unadapted reference once.  The utility objective must not
        # buy a ranking margin by destroying the base model's likelihood.
        agent.set_source(None)
        with torch.no_grad():
            base_positive = sequence_score(r["answer"])
        for arm in ("hyper","lora"):
            if arm=="hyper":
                for a,p in zip(agent.adapters,ha,strict=True): a.a=p
                agent.set_source({k:v.to(device) for k,v in cache[r["source"]].items()},target_fields={k:v.to(device) for k,v in cache[r["query"]].items()}); opt=oh; params=list(hyper.values())
            else:
                for a,p,b in zip(agent.adapters,sa,sb,strict=True): a.a=p; a.b=b*1.
                opt=ol; params=sa+sb
            opt.zero_grad(set_to_none=True)
            positive, negative = sequence_score(r["answer"]), sequence_score(r["negative"])
            margin = positive - negative
            rank_loss = torch.nn.functional.softplus(1.0 - margin)
            # Generated B is the executable parameter experience.  Constrain
            # its scale so the hypernetwork cannot win by destabilizing logits.
            b_norm = torch.stack([a.b.float().pow(2).mean() for a in agent.adapters]).mean()
            # Keep the positive action at least as likely as the frozen base;
            # this is a soft trust-region term, applied to both arms.
            anchor = torch.relu(base_positive.detach() - positive).pow(2)
            loss = rank_loss + b_penalty * b_norm + anchor_weight * anchor
            assert torch.isfinite(loss)
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); opt.step()
            losses[arm]={"loss":float(loss.detach()),"rank_loss":float(rank_loss.detach()),"positive":float(positive.detach()),"negative":float(negative.detach()),"base_positive":float(base_positive.detach()),"margin":float(margin.detach()),"b_norm":float(b_norm.detach()),"anchor":float(anchor.detach()),"grad_norm":float(norm)}
        history.append({"step":step,"row_index":idx,**losses})
        if step==1 or step%10==0:
            print(json.dumps(history[-1]),flush=True); write(OUT/"status.json",{"stage":"training","updates":step,"total":len(data["schedule"]),"elapsed":time.time()-started})
        if step%100==0:
            torch.save({"step":step,"hyper":{n:p.detach().cpu() for n,p in hyper.items()},"static_a":[p.detach().cpu() for p in sa],"static_b":[p.detach().cpu() for p in sb]},OUT/f"checkpoint_{step:04d}.pt"); write(OUT/"training.json",history)
    torch.save({"step":len(history),"hyper":{n:p.detach().cpu() for n,p in hyper.items()},"static_a":[p.detach().cpu() for p in sa],"static_b":[p.detach().cpu() for p in sb],"manifest_sha256":filehash(DATA_DIR/"manifest.json")},OUT/"trained.pt")
    write(OUT/"training.json",history); write(OUT/"complete.json",{"training_complete":True,"updates_per_arm":len(history),"checkpoint_sha256":filehash(OUT/"trained.pt"),"manifest_sha256":filehash(DATA_DIR/"manifest.json"),"frozen_before_transfer":True}); write(OUT/"status.json",{"stage":"training_complete","updates":len(history),"frozen":True})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:3");p.add_argument("--steps",type=int,default=None);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--b-penalty",type=float,default=1e-2);p.add_argument("--anchor-weight",type=float,default=0.1);p.add_argument("--data-dir",type=Path,default=None);p.add_argument("--out-root",type=Path,default=None);args=p.parse_args();main(args.device,args.steps,args.lr,args.b_penalty,args.anchor_weight,args.data_dir,args.out_root)
