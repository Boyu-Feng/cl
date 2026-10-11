"""Measure total effective BA/r change, including the learned shared A."""
import json
import math
from pathlib import Path
import torch
from ttcl.trajectory_hyperlora.transition_reader_v84 import TransitionReader
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
ROOT=Path('results/trajectory_hyperlora')


def inner(bs,aa,cs,dd):
    result=0.
    for b,a,c,d in zip(bs,aa,cs,dd,strict=True):
        b=b.squeeze(0).double();a=a.double();c=c.squeeze(0).double();d=d.double()
        result+=float(((b.T@c)*(a@d.T)).sum())/(a.shape[0]*d.shape[0])
    return result


def distance(bs,aa,cs,dd):
    return math.sqrt(max(0.,inner(bs,aa,bs,aa)+inner(cs,dd,cs,dd)-2*inner(bs,aa,cs,dd)))


def main():
    torch.set_num_threads(2);out=ROOT/'alfworld_shared_a_v89_geometry_20261010.json'
    if out.exists():raise FileExistsError(out)
    # Small dense check validates rank factors and the cross term independently.
    torch.manual_seed(89);b=[torch.randn(1,5,2)];a=[torch.randn(2,7)];c=[torch.randn(1,5,3)];d=[torch.randn(3,7)]
    assert math.isclose(distance(b,a,c,d),float((b[0].double()@a[0].double()/2-c[0].double()@d[0].double()/3).norm()),rel_tol=1e-10)
    folder=ROOT/'alfworld_transition_reader_v89_20261010';p=folder/'reader.pt';s=torch.load(p,map_location='cpu',weights_only=True)
    report=json.loads((folder/'training.json').read_text());assert report['complete'] and not report['failures'] and report['reader_sha256']==file_hash(p)
    bootpath=ROOT/'alfworld_centered_experience_v75b_20261010/bootstrap.pt';boot=torch.load(bootpath,map_location='cpu',weights_only=True)
    assert s['bootstrap_sha256']==file_hash(bootpath)
    parent=ROOT/'alfworld_transition_reader_v88_20261010';qp=parent/'eval_query_features.pt';queries=torch.load(qp,map_location='cpu',weights_only=True)
    ep=ROOT/'alfworld_transition_reader_v88_eval_20261010.json';evaluation=json.loads(ep.read_text());assert evaluation['complete'] and evaluation['query_features_sha256']==file_hash(qp)
    review=json.loads(Path('data/annotations/alfworld_transition_reader_v88_eval_20261010.json').read_text())
    cache=torch.load(parent/'features.pt',map_location='cpu',weights_only=True)
    fields=[torch.cat([cache['values'][cache['rows'][k][field]] for k in review['memory_indices']],dim=0) for field in ('key_text','action_text','feedback_text')]
    reader=TransitionReader(s['feature_dim'],s['factor_shapes'],s['width'],s['top_k'],s['temperature']);reader.load_state_dict(s['state']);reader.eval()
    norm=math.sqrt(inner(boot['B'],boot['A'],boot['B'],boot['A']));rows=[]
    with torch.no_grad():
        for row in evaluation['rows']:
            key=row['arms']['real']['parameter_reads'][0]['query_text_sha256'];q=queries[key][0]
            values,info=reader.adapted_factors(q,*fields,s['shared_B'],s['shared_A'])
            rows.append({'index':row['index'],'query_sha256':key,'total_relative_effective_change':distance(values,s['shared_A'],boot['B'],boot['A'])/norm,
                'B_increment_relative_to_learned_A_reference':float(info['raw_effective_norm']*info['scale']/info['reference_effective_norm'])})
    static=torch.load(parent/'static.pt',map_location='cpu',weights_only=True)
    result={'reader_sha256':file_hash(p),'bootstrap_sha256':file_hash(bootpath),'query_features_sha256':file_hash(qp),
        'runner_sha256':file_hash(Path(__file__)),'rows':rows,
        'empty_learned_A_relative_change':distance(s['shared_B'],s['shared_A'],boot['B'],boot['A'])/norm,
        'static_relative_change':distance(static['factors'],static['A'],boot['B'],boot['A'])/norm,
        'real_total_relative_change_range':[min(x['total_relative_effective_change'] for x in rows),max(x['total_relative_effective_change'] for x in rows)],
        'scope':'CPU float64 Gram geometry, reader forward float32, same24 visible initial queries saved in v88. Geometry is not task performance;25% limits B increment at learned A only.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
