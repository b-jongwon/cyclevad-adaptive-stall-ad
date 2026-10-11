"""Unified phase/process experiment and controlled appearance-head factorial."""
import hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
from .data import balanced_sample,hold,labels_for,write_json,fingerprint
from .model import descriptor,ResidualSpace,TailCalibrator
from .tracker import fourier
from .phase_experiment import fit_mean,compact
from .process_experiment import dynamics,edit_map,halfup,starts,weighted_quantile,QUANTILES
from .pipeline import load_rows,code_fingerprint
from .metrics import frame_metrics,event_metrics
ARMS=['B0','B1','B2','B3','C','P']

def read(p):return json.loads(Path(p).read_text())
def maxwindows(scores,locations,length):
    return np.stack([scores[o:o+length].max(0) for o in locations]) if len(locations) else np.empty((0,scores.shape[1]))
def calibrate(scores,splits,T,D):
    values=[];weights=[];counts={}
    for i in splits['threshold']:
        v=maxwindows(scores[i],starts(len(scores[i]),T,D),D);counts[i]=len(v)
        if len(v):values.append(v);weights.extend(np.full(len(v),1/len(v)))
    if len(values)<2:raise ValueError('Insufficient independent threshold sources')
    v=np.concatenate(values);w=np.asarray(weights)
    thresholds=np.array([[weighted_quantile(v[:,a],w,q) for a in range(v.shape[1])] for q in QUANTILES])
    f=np.quantile(np.concatenate([scores[i] for i in splits['threshold']]),.99,axis=0,method='higher')
    return thresholds,f,counts

def load_context(root,rows,scene,fold,seed,splits,historical=True):
    lookup={r['id']:r for r in rows};groups={k:list(v) for k,v in splits.items()}
    if historical and fold==0:groups['historical_test']=[r['id'] for r in rows if r['scene']==scene and r['partition']=='testing']
    ids=sum(groups.values(),[]);assert len(ids)==len(set(ids))
    data=dict(zip(ids,load_rows(root/'runs/stride1_seed42',[lookup[i] for i in ids])))
    for d in data.values():
        assert np.array_equal(d['indices'],np.arange(int(d['frame_count'])))
        identity=read_identity(d);assert identity['device']=='cuda' and identity['resolved_revision']=='f9e44c814b77203eaa57a6bdbbd535f21ede1415'
    base=joblib.load(root/'runs/E9S/stride2'/f'seed{seed}'/scene/f'fold{fold}'/'model.joblib')
    z={i:descriptor(d) for i,d in data.items()};x={i:base['tracker'].reducer.transform(v)/base['tracker'].scale for i,v in z.items()}
    idx={i:np.arange(0,len(v),2) for i,v in z.items()};trace={i:base['tracker'].predict_latent(x[i][idx[i]],idx[i]) for i in ids}
    appraw={}
    for i in ids:
        out,inside,_=base['pooled'].score(z[i]);loc,_=base['local'].score(data[i]['patches']);appraw[i]={'pooled':out,'inside':inside,'local':loc}
    appcal={k:TailCalibrator().fit(np.concatenate([appraw[i][k][::2] for i in splits['reference']])) for k in ['pooled','inside','local']}
    app={i:np.maximum.reduce([appcal[k].score(v) for k,v in vals.items()]) for i,vals in appraw.items()}
    return {'base':base,'data':data,'z':z,'x':x,'idx':idx,'trace':trace,'app':app,'appcal':appcal,'groups':groups,'lookup':lookup}
def read_identity(d):return json.loads(str(d['encoder_identity']))
def choose_mean(ctx,splits,seed,h=None,ridge=None):
    fit=[ctx['z'][i][::2] for i in splits['fit']];angles=[ctx['trace'][i]['angle'] for i in splits['fit']];selection=[];best=None
    for hh in ([h] if h else [2,4,8]):
        for rr in ([ridge] if ridge is not None else [.01,.1,1.]):
            coef=fit_mean(fit,angles,hh,rr,4096,seed)
            loss=float(np.mean([np.mean((ctx['z'][i][::2]-fourier(ctx['trace'][i]['angle'],hh)@coef)**2) for i in splits['validation']]))
            selection.append({'harmonics':hh,'ridge':rr,'validation_mse':loss})
            if best is None or loss<best[0]:best=(loss,hh,rr,coef)
    return best[1],best[2],best[3],selection

def fuse(a,c,p):return np.stack([a,np.maximum(a,c),np.maximum(a,p),np.maximum.reduce([a,c,p]),c,p],axis=1)
def conditional_raw(z,theta,coef,h,space):
    # Float64 score computation avoids float32 GEMM batch-shape changes being
    # amplified by empirical reference ranks. Stored fitted parameters stay fixed.
    residual=z.astype(np.float64)-fourier(theta,h).astype(np.float64)@coef.astype(np.float64)
    return space.score(residual)
def conditional(z,theta,coef,h,space,cal):
    out,inside,_=conditional_raw(z,theta,coef,h,space)
    return np.maximum(cal[0].score(out),cal[1].score(inside))
def event_rows(identity,labels,scores,threshold,D):
    positive=np.asarray(labels)==1;lo=np.flatnonzero(np.diff(np.r_[False,positive,False].astype(int))==1);hi=np.flatnonzero(np.diff(np.r_[False,positive,False].astype(int))==-1)
    rows=[]
    for j,(a,b) in enumerate(zip(lo,hi)):
        window=scores[a:min(b,a+D)];delays=[]
        for k in range(scores.shape[1]):
            hit=np.flatnonzero(window[:,k]>threshold[k]);delays.append(int(hit[0]) if len(hit) else None)
        rows.append({'id':f'{identity}:event{j}','source_id':identity,'onset':int(a),'end':int(b),'deadline_frames':int(min(D,b-a)),
                     'delay_primary':delays,'hit_full_interval':(scores[a:b].max(0)>threshold).tolist()})
    return rows

def run_e11(root,rows,scene,fold,seed,splits):
    began=time.perf_counter();cfg=read(root/'configs/experiments/followup/paper_stage1_v1.json');ctx=load_context(root,rows,scene,fold,seed,splits)
    if seed==42:h,lam,coef,selection=choose_mean(ctx,splits,seed)
    else:
        original=read(root/'results/E11/seed42'/scene/f'fold{fold}'/'run.json');h,lam,coef,selection=choose_mean(ctx,splits,seed,original['harmonics'],original['ridge'])
    fitres=[ctx['z'][i][::2]-fourier(ctx['trace'][i]['angle'],h)@coef for i in splits['fit']]
    space=ResidualSpace().fit(balanced_sample(fitres,4096,seed),64,seed=seed)
    refraw=[conditional_raw(ctx['z'][i][::2],ctx['trace'][i]['angle'],coef,h,space) for i in splits['reference']]
    ccal=[TailCalibrator().fit(np.concatenate([r[k] for r in refraw])) for k in [0,1]]
    pcal={k:TailCalibrator().fit(np.concatenate([ctx['trace'][i][k] for i in splits['reference']])) for k in ['alignment','innovation','progress']}
    scores={};sampled={}
    for i,theta in ctx['trace'].items():
        c=conditional(ctx['z'][i][::2],theta['angle'],coef,h,space,ccal);p=np.maximum.reduce([pcal[k].score(theta[k]) for k in pcal])
        sampled[i]=fuse(ctx['app'][i][::2],c,p);scores[i]=hold(ctx['idx'][i],sampled[i],len(ctx['z'][i]))
    T=ctx['base']['tracker'].period;D,W=halfup(.2*T),halfup(.4*T);thresholds,frame_threshold,counts=calibrate(scores,splits,T,D);primary=thresholds[3]
    output=root/'runs/E11'/f'seed{seed}'/scene/f'fold{fold}';output.mkdir(parents=True,exist_ok=True)
    joblib.dump({'base':ctx['base'],'appcal':ctx['appcal'],'coef':coef,'h':h,'ridge':lam,'space':space,'ccal':ccal,'pcal':pcal,'thresholds':thresholds,'frame_thresholds':frame_threshold},output/'model.joblib',compress=3)
    pervideo=[];metrics=[];normal=[];events=[]
    for part,ids in ctx['groups'].items():
        if part in ['fit','validation']:continue
        for i in ids:
            y=labels_for(ctx['lookup'][i]);seq=i.split('/')[-1];partition=i.split('/')[1]
            np.savez_compressed(output/f'{partition}_{seq}.npz',labels=y,scores=scores[i])
            for a,arm in enumerate(ARMS):
                pervideo.append({'id':i,'partition':part,'arm':arm,**frame_metrics(y,scores[i][:,a],primary[a]),**event_metrics(y,scores[i][:,a],primary[a]),
                    'frame_q99':frame_metrics(y,scores[i][:,a],frame_threshold[a])})
            if part=='historical_test':events+=event_rows(i,y,scores[i],primary,D)
            if part=='normal_evaluation':
                vmax=maxwindows(scores[i],starts(len(scores[i]),T,D),D)
                normal.append({'id':i,'frames':len(y),'grid_windows':len(vmax),'grid_alarm_rates':(vmax[:,None,:]>thresholds[None,:,:]).mean(0).tolist(),
                    'frame_fpr_primary':(scores[i]>primary).mean(0).tolist(),'frame_fpr_frame_q99':(scores[i]>frame_threshold).mean(0).tolist()})
        for a,arm in enumerate(ARMS):
            y=np.concatenate([labels_for(ctx['lookup'][i]) for i in ids]);s=np.concatenate([scores[i][:,a] for i in ids])
            metrics.append({'scene':scene,'seed':seed,'fold':fold,'partition':part,'arm':arm,**frame_metrics(y,s,primary[a]),'frame_q99':frame_metrics(y,s,frame_threshold[a])})
    manifest=read(root/'configs/experiments/followup/e9_cases_v2.json');cases=[];identity_errors=[];prefix_errors=[]
    for i in splits['normal_evaluation']:
        n=len(ctx['z'][i]);take=ctx['idx'][i];z=ctx['z'][i][take];trace=ctx['base']['tracker'].predict_latent(ctx['x'][i][take],take)
        replay=fuse(ctx['app'][i][take],conditional(z,trace['angle'],coef,h,space,ccal),np.maximum.reduce([pcal[k].score(trace[k]) for k in pcal]))
        np.testing.assert_allclose(replay,sampled[i],atol=1e-10,rtol=1e-10);identity_errors.append(float(np.max(np.abs(replay-sampled[i]))))
        for spec in [c for c in manifest if c['eligible'] and c['scene']==scene and c['fold']==fold and c['source_id']==i]:
            assert spec['T']==T and spec['deadline']==D and spec['window']==W
            o,L=spec['onset'],spec['length'];mapping=edit_map(n,o,L,spec['edit']);take=np.arange(0,o+W,2);src=mapping[take]
            tr=ctx['base']['tracker'].predict_latent(ctx['x'][i][src],take)
            c=conditional(ctx['z'][i][src],tr['angle'],coef,h,space,ccal);p=np.maximum.reduce([pcal[k].score(tr[k]) for k in pcal])
            ss=fuse(ctx['app'][i][src],c,p);score=hold(take,ss,o+W)
            # Float32 GEMM batch shape can slightly change calibrated scores; the
            # tracker prefix itself must remain invariant at numerical precision.
            pre=take<o
            for key in tr:np.testing.assert_allclose(tr[key][pre],ctx['trace'][i][key][:pre.sum()],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(score[:o],scores[i][:o],rtol=1e-9,atol=1e-9)
            prefix_errors.append(float(np.max(np.abs(score[:o]-scores[i][:o]))))
            delayed=[]
            for a in range(6):
                hit=np.flatnonzero(score[o:o+D,a]>primary[a]);delayed.append(int(hit[0]) if len(hit) else None)
            ilo=o+32;ihi=min(o+(2*L if spec['edit']=='swap_adjacent_blocks' else L),o+W)
            interior=score[ilo:ihi] if spec['edit']!='skip' and ihi>ilo else None
            cases.append({'id':spec['id'],'source_id':i,'scene':scene,'fold':fold,'seed':seed,'edit':spec['edit'],'severity':spec['severity'],'onset_fraction':spec['onset_fraction'],
                'edited_max_W':score[o:o+W].max(0).tolist(),'original_max_W':scores[i][o:o+W].max(0).tolist(),
                'edited_max_D':score[o:o+D].max(0).tolist(),'original_max_D':scores[i][o:o+D].max(0).tolist(),'delay_primary':delayed,
                'interior_frames':len(interior) if interior is not None else 0,'interior_alarm_primary':(interior.max(0)>primary).tolist() if interior is not None else None,
                'initial_transient_max':score[o:min(o+32,o+W)].max(0).tolist()})
    dest=root/'results/E11'/f'seed{seed}'/scene/f'fold{fold}'
    for name,values in [('metrics',metrics),('per_video',pervideo),('normal',normal),('cases',cases),('historical_events',events)]:compact(dest/f'{name}.json',values)
    write_json(dest/'run.json',{'status':'complete','scene':scene,'fold':fold,'seed':seed,'arms':ARMS,'splits':splits,'T':T,'D':D,'W':W,'harmonics':h,'ridge':lam,'mean_selection':selection,
        'window_thresholds':thresholds.tolist(),'frame_thresholds':frame_threshold.tolist(),'threshold_window_counts':counts,'edited_cases':len(cases),'identity_max_error':max(identity_errors),
        'score_prefix_max_error':max(prefix_errors),'tracker_prefix_invariance':True,'conditional_rank':len(space.basis),'protocol_sha256':fingerprint(cfg),
        'code_fingerprint':code_fingerprint(),'cache_signatures':{i:str(d['signature']) for i,d in ctx['data'].items()},'wall_seconds':time.perf_counter()-began})
    print(f'E11 seed{seed} {scene}/fold{fold}: {len(cases)} edits, {time.perf_counter()-began:.1f}s',flush=True)
