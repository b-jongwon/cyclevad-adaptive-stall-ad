"""Paired source-file bootstrap, stratified by scene; no frame resampling."""
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from cycle_vad.data import write_json
ROOT=Path(__file__).resolve().parents[1]
SCENES=['R01','R02','R03','R04']
N=2000

def weighted_metrics(y, score, owner, counts):
    # Tied scores form one threshold, matching sklearn AP/AUROC exactly.
    vals, group=np.unique(-score, return_inverse=True)
    positive=np.zeros((len(vals), counts.shape[0]))
    negative=np.zeros_like(positive)
    np.add.at(positive,(group[y==1],owner[y==1]),1)
    np.add.at(negative,(group[y==0],owner[y==0]),1)
    aps=[];aucs=[]
    for start in range(0,counts.shape[1],100):
        p=positive@counts[:,start:start+100];n=negative@counts[:,start:start+100]
        cp=np.cumsum(p,axis=0);cn=np.cumsum(n,axis=0)
        pt=cp[-1];nt=cn[-1]
        precision=np.divide(cp,cp+cn,out=np.zeros_like(cp),where=cp+cn>0)
        aps.extend(np.divide((p*precision).sum(0),pt,out=np.full_like(pt,np.nan),where=pt>0))
        aucs.extend(np.divide((p*(nt-cn+.5*n)).sum(0),pt*nt,out=np.full_like(pt,np.nan),where=pt*nt>0))
    return np.array(aps),np.array(aucs)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',choices=['C-21','C-22'],default='C-21');args=parser.parse_args()
    candidate=args.candidate
    rng=np.random.default_rng(20261009)
    apdiff=[];aucdiff=[];normal={};per_scene={}
    for scene in SCENES:
        ys=[];old=[];new=[];owners=[];ids=[]
        for p in sorted((ROOT/'runs/followup'/scene/'fold0/scores').glob('testing_*.npz')):
            with np.load(p) as d:
                mask=d['labels']>=0
                if not mask.any():continue
                ids.append(p.stem)
                ys.append(d['labels'][mask]);old.append(d['C-00__appearance_conditional'][mask]);new.append(d[f'{candidate}__appearance_conditional'][mask]);owners.append(np.full(mask.sum(),len(ids)-1))
        y,old,new,owner=map(np.concatenate,(ys,old,new,owners))
        counts=np.column_stack([np.ones(len(ids),dtype=int),rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=N).T])
        a,u=weighted_metrics(y,old,owner,counts);b,v=weighted_metrics(y,new,owner,counts)
        np.testing.assert_allclose([a[0],u[0],b[0],v[0]], [average_precision_score(y,old),roc_auc_score(y,old),average_precision_score(y,new),roc_auc_score(y,new)],rtol=1e-10,atol=1e-10)
        apdiff.append(100*(b[1:]-a[1:]));aucdiff.append(100*(v[1:]-u[1:]))
        per_scene[scene]={'valid_source_files':len(ids),'ap_gain_pp':100*(b[0]-a[0]),'ap_gain_pp_ci95':np.nanquantile(apdiff[-1],[.025,.975]).tolist()}
        videos=[]
        for fold in range(5):
            rows=json.loads((ROOT/f'results/E8/{scene}/fold{fold}/per_video.json').read_text())
            rows=[r for r in rows if r['partition']=='normal_evaluation' and r['readout']=='appearance_conditional']
            by={r['id']:{x['cell']:x for x in rows if x['id']==r['id']} for r in rows}
            for i,arms in by.items():videos.append((arms['C-00']['valid_frames'],arms['C-00']['fpr'],arms[candidate]['fpr']))
        f,a,b=np.array(videos).T
        w=rng.multinomial(len(f),np.full(len(f),1/len(f)),size=N)
        delta=100*((w@(f*b))/(w@f)-(w@(f*a))/(w@f))
        normal[scene]={'files':len(f),'fpr_gain_pp':float(100*((f*b).sum()-(f*a).sum())/f.sum()),'fpr_gain_pp_ci95':np.quantile(delta,[.025,.975]).tolist()}
    ap=np.mean(apdiff,axis=0);auc=np.mean(aucdiff,axis=0)
    write_json(ROOT/'results/E8'/('paired_bootstrap.json' if candidate=='C-21' else 'paired_bootstrap_C22.json'),{'resamples':N,'seed':20261009,'unit':'source_file','stratified_by':'scene','paired':True,
        'readout':'appearance_conditional','contrast':f'{candidate} minus C-00','historical_fold':0,
        'macro_ap_gain_pp_ci95':np.nanquantile(ap,[.025,.975]).tolist(),'macro_auroc_gain_pp_ci95':np.nanquantile(auc,[.025,.975]).tolist(),
        'finite_macro_ap_resamples':int(np.isfinite(ap).sum()),'scene_historical':per_scene,'normal_oof':normal,
        'group_independence_verified':False,'uncertainty_scope':'Fixed fitted models and thresholds; does not include retraining uncertainty or prove recording independence.',
        'metric_equivalence_to_sklearn_passed':True})
    print('Paired bootstrap complete:',np.nanquantile(ap,[.025,.975]))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
