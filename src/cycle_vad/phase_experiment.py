"""E8B: phase interventions with full mean/subspace refitting and held-out calibration."""
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
from .data import balanced_sample, hold, labels_for, write_json, fingerprint
from .model import ResidualSpace, TailCalibrator, descriptor
from .tracker import fourier
from .pipeline import load_rows, code_fingerprint
from .metrics import frame_metrics


def offset(identity, seed):
    value=json.dumps([seed,identity],separators=(',',':'))
    return (int(hashlib.sha256(value.encode()).hexdigest()[:13],16)+.5)/2**52


def phase(trace, ids, period, arm, identity):
    if arm=='M1':return (trace['observation_angle'][0]+(ids-ids[0])/period)%1
    if arm=='M2':return trace['observation_angle']
    if arm.startswith('M4_'):return (trace['angle']+offset(identity,int(arm.split('_')[1])))%1
    return trace['angle']


def fit_mean(z, angles, harmonics, ridge, limit, seed):
    rows=balanced_sample([np.concatenate([fourier(a,harmonics),x],1).astype(np.float32) for x,a in zip(z,angles)],limit,seed)
    cols=1+2*harmonics;b,x=rows[:,:cols],rows[:,cols:]
    penalty=np.eye(cols)*ridge*len(b);penalty[0,0]=1e-8
    return np.linalg.solve(b.T@b+penalty,b.T@x).astype(np.float32)


def compact(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text('[\n'+',\n'.join(json.dumps(r,separators=(',',':'),allow_nan=False) for r in rows)+'\n]\n');tmp.replace(path)


def run(root, rows, scene, fold, seed, splits):
    start=time.perf_counter();suffix='' if seed==42 else f'seed{seed}'
    base=joblib.load(root/'runs/followup'/suffix/scene/f'fold{fold}'/'train2.joblib')
    cfg=base.config;limit=cfg['fit_samples'];rank=cfg['subspace_rank']
    lookup={r['id']:r for r in rows}
    groups={k:list(v) for k,v in splits.items()}
    if fold==0:groups['historical_test']=[r['id'] for r in rows if r['scene']==scene and r['partition']=='testing']
    ids=[i for group in groups.values() for i in group];selected=[lookup[i] for i in ids]
    data=dict(zip(ids,load_rows(root/'runs/stride2_seed42',selected)))
    for d in data.values():
        assert np.all(np.diff(d['indices'])==2)
        assert json.loads(str(d['encoder_identity']))['resolved_revision']=='f9e44c814b77203eaa57a6bdbbd535f21ede1415'
    z={i:descriptor(d) for i,d in data.items()}
    traces={i:base.tracker.predict(z[i],data[i]['indices']) for i in ids}
    appraw={}
    for i in ids:
        out,inside,_=base.pooled.score(z[i]);local,_=base.local.score(data[i]['patches'])
        appraw[i]={'pooled':out,'pooled_inside':inside,'local':local}
    appcal={k:TailCalibrator().fit(np.concatenate([appraw[i][k] for i in splits['reference']])) for k in appraw[ids[0]]}
    app={i:np.maximum.reduce([appcal[k].score(v) for k,v in appraw[i].items()]) for i in ids}
    fitids=splits['fit'];fitz=[z[i] for i in fitids]
    sample=balanced_sample(fitz,limit,seed)
    const=sample.mean(0)
    shifted=ResidualSpace().fit(sample-const,rank,seed=seed)
    # PCA sign is arbitrary; compare the invariant scores, not basis signs.
    checks={}
    for label,j in [('fit',fitids[0]),('normal_evaluation',splits['normal_evaluation'][0])]:
        a=base.pooled.score(z[j]);b=shifted.score(z[j]-const)
        for axis in [0,1]:np.testing.assert_allclose(a[axis],b[axis],rtol=5e-4,atol=1e-6)
        checks[label]=float(max(np.max(np.abs(a[q]-b[q])) for q in (0,1)))
    arms=['M0','M1','M2','M3']+[f'M4_{s}' for s in range(1101,1106)]
    if seed==42:arms+=['M0_fixedrank','M3_fixedrank','M1_retuned','M2_retuned','M3_retuned']
    per_video=[];metrics=[];models=[];bank={i:{} for k,g in groups.items() if k not in ('fit','validation') for i in g}
    folder=root/'runs/E8B'/f'seed{seed}'/scene/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
    for arm in arms:
        source=arm.split('_')[0] if not arm.startswith('M4') else arm
        angle={i:phase(traces[i],data[i]['indices'],base.tracker.period,source,i) for i in ids}
        h,lam=base.harmonics,base.ridge;selection=[]
        if source=='M0':
            h=0;coef=const[None]
        elif arm.endswith('retuned'):
            best=None
            for hh in [2,4,8]:
                for rr in [.01,.1,1.]:
                    candidate=fit_mean(fitz,[angle[i] for i in fitids],hh,rr,limit,seed)
                    loss=float(np.mean([np.mean((z[i]-fourier(angle[i],hh)@candidate)**2) for i in splits['validation']]))
                    selection.append({'harmonics':hh,'ridge':rr,'validation_mse':loss})
                    if best is None or loss<best[0]:best=(loss,hh,rr,candidate)
            _,h,lam,coef=best
        elif source=='M3':coef=base.coef
        else:coef=fit_mean(fitz,[angle[i] for i in fitids],h,lam,limit,seed)
        if source=='M0' and arm=='M0':space=base.pooled
        elif arm=='M3':space=base.conditional
        else:
            residuals=[z[i]-fourier(angle[i],h)@coef for i in fitids]
            space=ResidualSpace().fit(balanced_sample(residuals,limit,seed),rank,variance=1.0 if arm.endswith('fixedrank') else .95,seed=seed)
        if arm=='M3':
            refit=fit_mean(fitz,[angle[i] for i in fitids],h,lam,limit,seed)
            np.testing.assert_allclose(coef,refit,rtol=1e-5,atol=1e-7)
        raw={};mse={}
        for i in ids:
            residual=z[i]-fourier(angle[i],h)@coef
            mse[i]=np.mean(residual**2,axis=1)
            out,inside,_=space.score(z[i] if arm=='M0' else residual)
            raw[i]=(out,inside)
        cal=[TailCalibrator().fit(np.concatenate([raw[i][k] for i in splits['reference']])) for k in (0,1)]
        streams={}
        for i in ids:
            cond=np.maximum(cal[0].score(raw[i][0]),cal[1].score(raw[i][1]))
            streams[i]={'conditional_only':cond,'appearance_conditional':np.maximum(app[i],cond)}
            if arm=='M0':np.testing.assert_array_equal(streams[i]['appearance_conditional'],app[i])
        thresholds={name:float(np.quantile(np.concatenate([hold(data[i]['indices'],streams[i][name],lookup[i]['frames']) for i in splits['threshold']]),.99,method='higher')) for name in streams[ids[0]]}
        for part,partids in groups.items():
            if part in ('fit','validation'):continue
            ys=[];scores={n:[] for n in thresholds}
            for i in partids:
                d=data[i];n=lookup[i]['frames'];y=labels_for(lookup[i]);ys.append(y)
                for name in thresholds:
                    score=hold(d['indices'],streams[i][name],n);scores[name].append(score);bank[i][f'{arm}__{name}']=score
                    per_video.append({'scene':scene,'fold':fold,'seed':seed,'id':i,'partition':part,'arm':arm,'readout':name,
                        **frame_metrics(y,score,thresholds[name]),
                        'mu_mse':float(hold(d['indices'],mse[i],n).mean()),'outside_mse':float(hold(d['indices'],raw[i][0],n).mean()),
                        'conditional_activation':float(hold(d['indices'],streams[i]['conditional_only']>app[i],n).mean()),'rank':len(space.basis)})
            for name in thresholds:
                recs=[r for r in per_video if r['arm']==arm and r['partition']==part and r['readout']==name]
                fp=[r['fpr'] for r in recs if r['fpr'] is not None]
                metrics.append({'scene':scene,'fold':fold,'seed':seed,'partition':part,'arm':arm,'readout':name,
                    **frame_metrics(np.concatenate(ys),np.concatenate(scores[name]),thresholds[name]),
                    'video_mean_fpr':float(np.mean(fp)) if fp else None,'video_mean_mu_mse':float(np.mean([r['mu_mse'] for r in recs])),
                    'video_mean_outside_mse':float(np.mean([r['outside_mse'] for r in recs]))})
        models.append({'arm':arm,'harmonics':h,'ridge':lam,'rank':len(space.basis),'selection':selection,'thresholds':thresholds})
        joblib.dump({'coef':coef,'space':space,'calibrators':cal,'h':h,'thresholds':thresholds},folder/f'{arm}.joblib',compress=3)
    for i,values in bank.items():
        _,part,seq=i.split('/');p=folder/'scores'/f'{part}_{seq}.npz';p.parent.mkdir(exist_ok=True)
        np.savez_compressed(p,labels=labels_for(lookup[i]),**values)
    dest=root/'results/E8B'/f'seed{seed}'/scene/f'fold{fold}'
    compact(dest/'metrics.json',metrics);compact(dest/'per_video.json',per_video)
    write_json(dest/'run.json',{'status':'complete','scene':scene,'seed':seed,'fold':fold,'splits':splits,'models':models,
        'M0_translation_score_max_abs_error':checks,'M0_fusion_equivalence':True,'M3_mean_equivalence':True,
        'code_fingerprint':code_fingerprint(),'protocol_sha256':fingerprint(json.loads((root/'configs/experiments/followup/phase_process_v2.json').read_text())),
        'cache_signatures':{i:str(d['signature']) for i,d in data.items()},'wall_seconds':time.perf_counter()-start})
    print(f'E8B seed={seed} {scene} fold={fold}: {len(arms)} arms, {time.perf_counter()-start:.1f}s',flush=True)
