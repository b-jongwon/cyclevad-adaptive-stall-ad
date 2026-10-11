"""Frozen C/P recalibration and predeclared alarm policies (E15B0/B/C)."""
from pathlib import Path
import time
import joblib
import numpy as np
from .data import hold, write_json, fingerprint
from .model import TailCalibrator
from .paper_experiments import read, maxwindows, conditional_raw
from .paper_diagnostics import sha
from .process_experiment import starts, weighted_quantile, edit_map
from .phase_experiment import compact
from .pipeline import code_fingerprint

ALPHAS = [.01, .05, .10]
POLICIES = ['C', 'MAX', 'OR75', 'OR50', 'ORfree']
SCENES = ['R01', 'R02', 'R03', 'R04']
COMPONENTS = ['C_outside', 'C_inside', 'P_alignment', 'P_innovation', 'P_progress']

def suffix(seed, scene, fold):
    return Path(f'seed{seed}') / scene / f'fold{fold}'

def load_bank(root, rel, diagnostic=False):
    stage = 'E15A' if diagnostic else 'E11B'
    return {f'{rel.parts[1]}/{p.stem.replace("_", "/", 1)}':dict(np.load(p))
            for p in sorted((root/'runs'/stage/rel).glob('*.npz')) if p.stem != 'edited_windows'}

def pool_windows(bank, ids, T, D):
    vals=[]; weights=[]; counts={}
    for i in ids:
        v=maxwindows(bank[i], starts(len(bank[i]), T, D), D); counts[i]=len(v)
        if len(v): vals.append(v); weights.extend(np.full(len(v),1/len(v)))
    if len(vals)<2: raise ValueError('At least two independent threshold sources required')
    w=np.asarray(weights); return np.concatenate(vals), w/w.sum(), counts

def policy_thresholds(v, w, alphas=ALPHAS):
    result=[]
    for alpha in alphas:
        c,p=[weighted_quantile(v[:,j],w,1-alpha) for j in range(2)]
        joint=weighted_quantile(v.max(1),w,1-alpha)
        row=[[c,np.inf],[joint,joint]]
        for share in [.75,.5]:
            row.append([weighted_quantile(v[:,0],w,1-alpha*share),
                        weighted_quantile(v[:,1],w,1-alpha*(1-share))])
        row.append([c,p]); result.append(row)
    return np.array(result)

def alarms(scores, thresholds):
    # Input [...,2]; output [...,alpha,policy]. No surrogate ranking score for OR.
    return (scores[...,None,None,:]>thresholds).any(-1)

def serial_thresholds(th):
    return [[[None if not np.isfinite(x) else float(x) for x in pair] for pair in row] for row in th]

def deserialize_thresholds(th):
    return np.array([[[np.inf if x is None else x for x in pair] for pair in row] for row in th])

def event_records(identity, labels, scores, thresholds, D):
    delta=np.diff(np.r_[False,np.asarray(labels)==1,False].astype(int)); records=[]
    for j,(lo,hi) in enumerate(zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1))):
        a=alarms(scores[lo:min(hi,lo+D)], thresholds)
        delay=np.where(a.any(0),a.argmax(0),-1)
        records.append({'id':f'{identity}:event{j}','source_id':identity,'onset':int(lo),'end':int(hi),
                        'deadline_frames':int(len(a)),'hit':a.any(0).astype(int).tolist(),
                        'delay':delay.tolist(),'peak_CP':scores[lo:min(hi,lo+D)].max(0).tolist()})
    return records

def normal_record(identity, scores, thresholds, T, D):
    v=maxwindows(scores,starts(len(scores),T,D),D)
    hit=alarms(v,thresholds); frame=alarms(scores,thresholds)
    c=v[:,0]>thresholds[0,0,0]; p=v[:,1]>thresholds[0,4,1]
    return {'id':identity,'frames':len(scores),'windows':len(v),'grid_far':hit.mean(0).tolist(),
            'frame_fpr':frame.mean(0).tolist(),
            'overlap_q99':{'C_only':float((c&~p).mean()),'P_only':float((p&~c).mean()),
                           'both':float((c&p).mean()),'neither':float((~c&~p).mean())}}

class VideoEqualTail:
    """Reweight reference ECDF only; retain frozen smoothing and extrapolation."""
    def fit(self, sequences, frozen):
        if not sequences or any(not len(x) for x in sequences): raise ValueError('Empty reference')
        values=np.concatenate(sequences).astype(float)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite reference')
        self.n=len(values); weights=np.concatenate([np.full(len(x),1/(len(sequences)*len(x))) for x in sequences])
        order=np.argsort(values,kind='stable'); self.values=values[order]
        self.cumulative=np.r_[0,np.cumsum(weights[order])]; self.cumulative[-1]=1.
        np.testing.assert_allclose(self.values,frozen.reference,atol=1e-9,rtol=1e-9)
        self.scale=frozen.scale; self.maximum=float(frozen.reference[-1]); return self
    def score(self, values):
        values=np.asarray(values)
        lo=np.searchsorted(self.values,values,'left'); hi=np.searchsorted(self.values,values,'right')
        cdf=(self.cumulative[lo]+self.cumulative[hi])/2
        survival=(self.n*(1-cdf)+1)/(self.n+1)
        return -np.log(survival)+np.maximum(values-self.maximum,0)/self.scale

def branches(raw, calibrators):
    c=np.column_stack([cal.score(raw[:,j]) for j,cal in enumerate(calibrators)])
    return np.column_stack([c[:,:2].max(1),c[:,2:].max(1)])

def run_b0(root, seed, scene, fold):
    rel=suffix(seed,scene,fold); r=read(root/'results/E11B'/rel/'run.json'); data=load_bank(root,rel)
    bank={i:d['scores'][:,[4,5]] for i,d in data.items()}
    v,w,counts=pool_windows(bank,r['splits']['threshold'],r['T'],r['D']); th=policy_thresholds(v,w)
    old=np.array(r['window_thresholds'])
    np.testing.assert_array_equal(th[:,0,0],old[[3,1,0],4]);np.testing.assert_array_equal(th[:,1,0],old[[3,1,0],6])
    normal=[normal_record(i,bank[i],th,r['T'],r['D']) for i in r['splits']['normal_evaluation']]
    events=[]
    for i in data:
        if '/testing/' in i: events+=event_records(i,data[i]['labels'],bank[i],th,r['D'])
    loo=[]
    for drop in r['splits']['threshold']:
        kept=[i for i in r['splits']['threshold'] if i!=drop]
        if len(kept)<2: loo.append({'excluded':drop,'status':'unavailable'});continue
        vv,ww,cc=pool_windows(bank,kept,r['T'],r['D'])
        loo.append({'excluded':drop,'status':'complete','thresholds':serial_thresholds(policy_thresholds(vv,ww))})
    jumps=[]
    for j in range(3):
        x=v[:,j] if j<2 else v.max(1);unique,inv=np.unique(x,return_inverse=True);mass=np.bincount(inv,weights=w)
        jumps.append({'branch':['C','P','CP'][j],'unique_values':len(unique),'max_ecdf_jump':float(mass.max()),
                      'top_value_mass':float(mass[-1]),'q95_equals_q995':bool(weighted_quantile(x,w,.95)==weighted_quantile(x,w,.995))})
    dest=root/'results/E15B0'/rel
    for name,value in [('normal',normal),('events',events),('leave_one_threshold_source_out',loo)]:compact(dest/f'{name}.json',value)
    write_json(dest/'run.json',{'status':'complete','seed':seed,'scene':scene,'fold':fold,'alphas':ALPHAS,'policies':POLICIES,
                              'T':r['T'],'D':r['D'],'thresholds':serial_thresholds(th),'counts':counts,'ecdf':jumps,
                              'window_stride':max(1,int(np.floor(.1*r['T']+.5))),
                              'calibration_values':v.tolist(),'calibration_weights':w.tolist(),
                              'baseline_sha256':sha(root/'results/E11B'/rel/'run.json')})
    print(f'E15B0 {rel}: {len(v)} windows; {len(events)} historical events',flush=True)


def run_bc(root, rows, seed, scene, fold):
    began=time.perf_counter(); rel=suffix(seed,scene,fold); r=read(root/'results/E11B'/rel/'run.json')
    cfg=read(root/'configs/experiments/followup/e15bc_v1.json'); modelpath=root/'runs/E11'/rel/'model.joblib'
    m=joblib.load(modelpath); frozen=m['ccal']+[m['pcal'][k] for k in ['alignment','innovation','progress']]
    data=load_bank(root,rel,True); refs=r['splits']['reference']; splits=r['splits'];T,D,W=[r[k] for k in ['T','D','W']]
    weighted=[VideoEqualTail().fit([data[i]['raw'][::2,3+j] for i in refs],frozen[j]) for j in range(5)]
    banks=[{i:d['scores'][:,[4,5]] for i,d in data.items()},
           {i:branches(d['raw'][:,3:],weighted) for i,d in data.items()}]
    error=0.
    for i,d in data.items():
        reproduced=branches(d['raw'][:,3:],frozen);np.testing.assert_allclose(reproduced,banks[0][i],atol=1e-9,rtol=1e-9)
        error=max(error,float(abs(reproduced-banks[0][i]).max()))
    thresholds=[];normal=[];events=[];frames=[]; rank=[]
    from sklearn.metrics import average_precision_score,roc_auc_score
    for g,bank in enumerate(banks):
        v,w,counts=pool_windows(bank,splits['threshold'],T,D);th=policy_thresholds(v,w);thresholds.append(th)
        for i in splits['normal_evaluation']:normal.append({'g':g,**normal_record(i,bank[i],th,T,D)})
        for i,d in data.items():
            if '/testing/' not in i:continue
            ev=event_records(i,d['labels'],bank[i],th,D);events.extend({'g':g,**e} for e in ev)
            a=alarms(bank[i],th); y=d['labels']; prior=np.cumsum(y==1)>0
            for name,mask in [('normal',y==0),('anomaly',y==1),('normal_before',(y==0)&~prior),('normal_after',(y==0)&prior)]:
                if mask.any():frames.append({'id':i,'g':g,'stratum':name,'frames':int(mask.sum()),'alarm_count':a[mask].sum(0).tolist()})
        hist=[i for i in data if '/testing/' in i]
        if hist:
            y=np.concatenate([data[i]['labels'] for i in hist]);s=np.concatenate([bank[i] for i in hist]);valid=y>=0
            for name,score in [('C',s[:,0]),('MAX',s.max(1))]:
                rank.append({'g':g,'arm':name,'ap':float(average_precision_score(y[valid],score[valid])),
                             'auroc':float(roc_auc_score(y[valid],score[valid]))})
    np.testing.assert_array_equal(thresholds[0][:,0,0],np.array(r['window_thresholds'])[[3,1,0],4])
    np.testing.assert_array_equal(thresholds[0][:,1,0],np.array(r['window_thresholds'])[[3,1,0],6])
    output=root/'runs/E15BC'/rel;output.mkdir(parents=True,exist_ok=True)
    for i,d in data.items():
        _,part,seq=i.split('/');np.savez_compressed(output/f'{part}_{seq}.npz',labels=d['labels'],scores=np.stack([b[i] for b in banks]),theta=d['theta'])
    # Replay edited observations from frozen dense CUDA features, never from a
    # calibrated branch maximum. FIT/VAL are not loaded or adapted here.
    from .pipeline import load_rows
    from .model import descriptor
    lookup={v['id']:v for v in rows};ids=splits['normal_evaluation']
    features=load_rows(root/'runs/stride1_seed42',[lookup[i] for i in ids]);z={i:descriptor(d) for i,d in zip(ids,features)}
    for d in features:
        from .paper_experiments import read_identity
        enc=read_identity(d);assert enc['device']=='cuda' and enc['resolved_revision']=='f9e44c814b77203eaa57a6bdbbd535f21ede1415'
    x={i:m['base']['tracker'].reducer.transform(v)/m['base']['tracker'].scale for i,v in z.items()}
    specs={c['id']:c for c in read(root/'configs/experiments/followup/e9_cases_v2.json') if c['eligible']}
    previous=read(root/'results/E11B'/rel/'cases.json');oldwindows=np.load(root/'runs/E11B'/rel/'edited_windows.npz')
    assert oldwindows['ids'].tolist()==[v['id'] for v in previous]
    cases=[];raw_windows=[];theta_windows=[];scores_windows=[];prefix_error=0.
    for index,old in enumerate(previous):
        spec=specs[old['id']];i=spec['source_id'];o,L=spec['onset'],spec['length'];take=np.arange(0,o+W,2)
        src=edit_map(len(z[i]),o,L,spec['edit'])[take];tr=m['base']['tracker'].predict_latent(x[i][src],take)
        co,ci,_=conditional_raw(z[i][src],tr['angle'],m['coef'],m['h'],m['space'])
        raw=hold(take,np.column_stack([co,ci]+[tr[k] for k in ['alignment','innovation','progress']]),o+W)
        theta=hold(take,tr['angle'],o+W);g0=branches(raw,frozen);g1=branches(raw,weighted)
        for g,s in enumerate([g0,g1]):
            np.testing.assert_allclose(s[:o],banks[g][i][:o],atol=1e-9,rtol=1e-9)
            prefix_error=max(prefix_error,float(abs(s[:o]-banks[g][i][:o]).max()))
        np.testing.assert_allclose(g0[o:o+W],oldwindows['scores'][index][:,[4,5]],atol=1e-9,rtol=1e-9)
        # Preserve exact old floating values on the reproduction arm.
        windows=np.stack([oldwindows['scores'][index][:,[4,5]],g1[o:o+W]])
        raw_windows.append(raw[o:o+W]);theta_windows.append(theta[o:o+W]);scores_windows.append(windows)
        for g,window in enumerate(windows):
            a=alarms(window,thresholds[g]);orig=alarms(banks[g][i][o:o+D],thresholds[g]);hit=a[:D].any(0)
            rec={k:old[k] for k in ['id','source_id','edit','severity','onset_fraction']}
            rec.update(g=g,hit=hit.astype(int).tolist(),matched_hit=orig.any(0).astype(int).tolist(),
                       delay=np.where(hit,a[:D].argmax(0),-1).tolist(),alarm_fraction_W=a.mean(0).tolist())
            cases.append(rec)
    np.savez_compressed(output/'edited_raw.npz',ids=oldwindows['ids'],raw=np.stack(raw_windows),theta=np.stack(theta_windows),scores=np.stack(scores_windows))
    dest=root/'results/E15BC'/rel
    for name,value in [('normal',normal),('events',events),('frames',frames),('ranking',rank),('cases',cases)]:compact(dest/f'{name}.json',value)
    write_json(dest/'run.json',{'status':'complete','seed':seed,'scene':scene,'fold':fold,'splits':splits,'T':T,'D':D,'W':W,
                              'alphas':ALPHAS,'policies':POLICIES,'calibrations':['G0','G1'],'components':COMPONENTS,
                              'thresholds':[serial_thresholds(th) for th in thresholds],'threshold_window_counts':counts,
                              'edited_cases':len(previous),'replay_error':error,'prefix_error':prefix_error,
                              'frozen_checkpoint_sha256':sha(modelpath),'config_sha256':fingerprint(cfg),
                              'baseline_sha256':sha(root/'results/E11B'/rel/'run.json'),'code_fingerprint':code_fingerprint(),
                              'wall_seconds':time.perf_counter()-began})
    print(f'E15BC {rel}: {len(previous)} edits, 8 budgeted + 2 diagnostic conditions; {time.perf_counter()-began:.1f}s',flush=True)
