"""E9S source-frame edits and causal temporal controls with normal-window calibration."""
import json
import math
import time
from pathlib import Path
import joblib
import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import softmax
from .data import balanced_sample, hold, write_json, fingerprint
from .model import descriptor, ResidualSpace, LocalMemory, TailCalibrator
from .tracker import CycleTracker, circular_difference
from .pipeline import load_rows, code_fingerprint
from .phase_experiment import compact

ARMS=[f'P{i}' for i in range(10)]
QUANTILES=[.90,.95,.975,.99,.995]


def halfup(x):return int(math.floor(x+.5))


def edit_map(n, onset, length, kind):
    o,L=onset,length;ids=np.arange(n)
    if kind=='identity':return ids
    if o<1 or L<2:raise ValueError('Invalid edit prefix/length')
    if kind=='freeze':return np.r_[ids[:o],np.full(L,o-1),ids[o:]]
    if o+(2*L if kind=='swap_adjacent_blocks' else L)>n:raise ValueError('Edit outside source')
    if kind=='reverse':return np.r_[ids[:o],ids[o:o+L][::-1],ids[o+L:]]
    if kind=='skip':return np.r_[ids[:o],ids[o+L:]]
    if kind=='swap_adjacent_blocks':return np.r_[ids[:o],ids[o+L:o+2*L],ids[o:o+L],ids[o+2*L:]]
    raise ValueError(kind)


def speed_map(n,speed):
    if speed<=0:raise ValueError('Positive speed required')
    t=np.arange(math.ceil(n/speed));return np.floor(speed*t).astype(int)[np.floor(speed*t)<n]


def weighted_quantile(values, weights, q):
    values,weights=np.asarray(values),np.asarray(weights)
    if not len(values) or len(values)!=len(weights) or np.any(weights<0) or weights.sum()<=0:raise ValueError('Invalid weighted observations')
    order=np.argsort(values,kind='stable');cum=np.cumsum(weights[order]);idx=min(np.searchsorted(cum,q*cum[-1],side='left'),len(order)-1)
    return float(values[order[idx]])


class TwoTail:
    def fit(self, values):
        self.ref=np.sort(np.asarray(values,dtype=float))
        if not len(self.ref) or not np.isfinite(self.ref).all():raise ValueError('Valid reference required')
        return self
    def score(self,values):
        n=len(self.ref);lo=np.searchsorted(self.ref,values,'left');hi=np.searchsorted(self.ref,values,'right')
        lower=(lo+.5*(hi-lo)+.5)/(n+1)
        return -np.log(np.maximum(np.minimum(1,2*np.minimum(lower,1-lower)),1/(n+1)))


class PhysicalTracker(CycleTracker):
    def __init__(self,stride=2,seed=42):
        super().__init__(128,24,seed);self.stride=stride
    def encode(self,z):
        x=self.reducer.transform(z)/self.scale
        lag=np.maximum(np.arange(len(x))-4//self.stride,0)
        return np.concatenate([x,.5*(x-x[lag])],axis=1)
    def predict_latent(self,x,indices):
        ids=np.asarray(indices);n=len(x);g=self.grid_size
        if len(ids)!=n or ids[0]!=0 or not np.all(np.diff(ids)==self.stride):raise ValueError('Uniform increasing output-time indices required')
        lag=np.maximum(np.arange(n)-4//self.stride,0)
        encoded=np.concatenate([x,.5*(x-x[lag])],axis=1)
        distance=np.mean((encoded[:,None,:]-self.template[None,:,:])**2,axis=2)
        likelihood=softmax(-distance/self.temperature,axis=1)
        observation=(np.angle(np.sum(likelihood*self.unit[None,:],axis=1))/(2*np.pi))%1
        angle=np.zeros(n);innovation=np.zeros(n);one_step=np.zeros(n)
        posterior=np.full(g,1/g);previous=np.full(g,1/g)
        dt=self.stride;offsets=np.arange(-g//4,g//4+1)
        sigma=np.sqrt(dt/2)*max(1,abs(g*2/self.period)*.5)
        weights=softmax(-.5*((offsets-g*dt/self.period)/sigma)**2)
        restart=1-(1-.001)**(dt/2)
        for t,like in enumerate(likelihood):
            if t:
                priors=convolve1d(np.stack([posterior,previous]),weights,axis=1,mode='wrap')
                priors=(1-restart)*priors+restart/g
                prior,other=priors
                innovation[t]=-np.log(max(g*np.dot(prior,like),1e-12))
                one_step[t]=-np.log(max(g*np.dot(other,like),1e-12))
            else:prior=posterior
            posterior=prior*like;posterior/=max(posterior.sum(),1e-300)
            angle[t]=(np.angle(posterior@self.unit)/(2*np.pi))%1;previous=like
        def progress(theta):
            result=np.zeros(n)
            for h in [2,8,32]:
                k=h//self.stride
                if n<=k:continue
                elapsed=(ids[k:]-ids[:-k])/self.period
                delta=np.abs(circular_difference(theta[k:],theta[:-k])-elapsed)/np.maximum(elapsed,1/g)
                result[k:]=np.maximum(result[k:],np.where(elapsed<.45,delta,0))
            return result
        return {'alignment':distance.min(1),'innovation':innovation,'progress':progress(angle),
                'one_innovation':one_step,'one_progress':progress(observation),'angle':angle,'observation_angle':observation}


def fit_ar(xs,validation,stride,seed):
    lag=2//stride
    def pairs(x):return np.concatenate([np.ones((len(x)-lag,1)),x[:-lag],x[lag:]],axis=1)
    rows=balanced_sample([pairs(x) for x in xs],4096,seed);dim=xs[0].shape[1]
    a,y=rows[:,:dim+1],rows[:,dim+1:];best=None;selection=[]
    for ridge in [.01,.1,1.]:
        penalty=np.eye(dim+1)*ridge*len(a);penalty[0,0]=1e-8
        coef=np.linalg.solve(a.T@a+penalty,a.T@y)
        mse=float(np.mean([np.mean((np.c_[np.ones(len(x)-lag),x[:-lag]]@coef-x[lag:])**2) for x in validation]))
        selection.append({'ridge':ridge,'normal_validation_mse':mse})
        if best is None or mse<best[0]:best=(mse,ridge,coef)
    return best[2],selection,best[1]


def dynamics(tracker,x,indices,ar):
    out=tracker.predict_latent(x,indices);valid={k:np.ones(len(x),dtype=bool) for k in out}
    for h in [2,8,32]:
        k=h//tracker.stride;name=f'diff{h}';out[name]=np.zeros(len(x));out[name][k:]=np.mean((x[k:]-x[:-k])**2,axis=1)
        valid[name]=np.arange(len(x))>=k
    k=2//tracker.stride;out['ar']=np.zeros(len(x))
    out['ar'][k:]=np.mean((np.c_[np.ones(max(0,len(x)-k)),x[:-k]]@ar-x[k:])**2,axis=1)
    valid['ar']=np.arange(len(x))>=k
    return out,valid


def combine(app,raw,valid,cal):
    q={k:np.where(valid[k],cal[k].score(raw[k]),0.) for k in cal}
    def mix(*keys):return np.maximum.reduce([app]+[q[k] for k in keys])
    return np.stack([app,mix('alignment'),mix('innovation'),mix('progress'),mix('alignment','innovation','progress'),
                     mix('ar'),mix('diff2'),mix('diff2','diff8','diff32'),mix('innovation','progress'),
                     mix('alignment','one_innovation','one_progress')],axis=1)


def starts(n,T,D):return np.arange(32,n-D+1,max(2,halfup(.1*T)),dtype=int)


def window_max(score,locations,length):
    return np.stack([score[o:o+length].max(0) for o in locations]) if len(locations) else np.empty((0,len(ARMS)))


def run(root,rows,scene,fold,seed,splits,stride=2):
    start=time.perf_counter();cfg=json.loads((root/'configs/experiments/followup/phase_process_v2.json').read_text())
    lookup={r['id']:r for r in rows};ids=[i for v in splits.values() for i in v]
    data=dict(zip(ids,load_rows(root/'runs/stride1_seed42',[lookup[i] for i in ids])))
    for d in data.values():
        assert np.array_equal(d['indices'],np.arange(int(d['frame_count'])))
        assert json.loads(str(d['encoder_identity']))['resolved_revision']==cfg['feature_checkpoint']
    # Framewise representation and appearance are computed once per source and
    # reindexed after edits. All temporal descriptors/states are recomputed.
    z={i:descriptor(data[i]) for i in ids};sampled={i:z[i][::stride] for i in ids}
    indices={i:np.arange(0,len(z[i]),stride) for i in ids}
    tracker=PhysicalTracker(stride,seed)
    tracker.fit([sampled[i] for i in splits['fit']],[indices[i] for i in splits['fit']],[[np.arange(len(sampled[i]))] for i in splits['fit']])
    T=float(np.median([len(z[i]) for i in splits['fit']]));tracker.period=T
    x={i:tracker.reducer.transform(z[i])/tracker.scale for i in ids}
    pooled=ResidualSpace().fit(balanced_sample([sampled[i] for i in splits['fit']],4096,seed),64,seed=seed)
    local=LocalMemory().fit([data[i]['patches'][::stride] for i in splits['fit']],96,seed)
    appraw={}
    for i in ids:
        out,inside,_=pooled.score(z[i]);loc,_=local.score(data[i]['patches'])
        appraw[i]={'pooled':out,'inside':inside,'local':loc}
    appcal={k:TailCalibrator().fit(np.concatenate([appraw[i][k][::stride] for i in splits['reference']])) for k in ['pooled','inside','local']}
    app={i:np.maximum.reduce([appcal[k].score(v) for k,v in appraw[i].items()]) for i in ids}
    ar,selection,chosen=fit_ar([x[i][::stride] for i in splits['fit']],[x[i][::stride] for i in splits['validation']],stride,seed)
    raw={};valid={}
    for i in splits['reference']+splits['threshold']+splits['normal_evaluation']:
        raw[i],valid[i]=dynamics(tracker,x[i][::stride],indices[i],ar)
    keys=['alignment','innovation','progress','one_innovation','one_progress','ar','diff2','diff8','diff32']
    cal={k:(TwoTail() if k.startswith('diff') else TailCalibrator()).fit(np.concatenate([raw[i][k][valid[i][k]] for i in splits['reference']])) for k in keys}
    original={i:hold(indices[i],combine(app[i][::stride],raw[i],valid[i],cal),len(x[i])) for i in raw}
    D,W=halfup(.2*T),halfup(.4*T);threshold_windows=[];weights=[];calibration_counts={}
    for i in splits['threshold']:
        locations=starts(len(x[i]),T,D);values=window_max(original[i],locations,D);calibration_counts[i]=len(values)
        if len(values):threshold_windows.append(values);weights.extend(np.full(len(values),1/len(values)))
    if len(threshold_windows)<2:raise ValueError('Fewer than two normal threshold videos with windows; operating point unavailable')
    windows=np.concatenate(threshold_windows);weights=np.array(weights)
    thresholds=np.array([[weighted_quantile(windows[:,a],weights,q) for a in range(10)] for q in QUANTILES])
    primary=thresholds[QUANTILES.index(.99)];frame_threshold=np.quantile(np.concatenate([original[i] for i in splits['threshold']]),.99,axis=0,method='higher')
    output=root/'runs/E9S'/f'stride{stride}'/f'seed{seed}'/scene/f'fold{fold}';output.mkdir(parents=True,exist_ok=True)
    joblib.dump({'tracker':tracker,'pooled':pooled,'local':local,'ar':ar,'appcal':appcal,'cal':cal,'thresholds':thresholds,'frame_threshold':frame_threshold},output/'model.joblib',compress=3)
    cases=json.loads((root/'configs/experiments/followup/e9_cases_v2.json').read_text())
    selected=[c for c in cases if c['scene']==scene and c['fold']==fold and c['eligible']]
    assert all(c['T']==T and c['window']==W and c['deadline']==D for c in selected)
    case_results=[];normal=[];speed_results=[];identity_errors=[]
    for i in splits['normal_evaluation']:
        n=len(x[i]);identity=edit_map(n,0,0,'identity')
        rid=identity[::stride];trace,mask=dynamics(tracker,x[i][rid],np.arange(len(rid))*stride,ar)
        replay=hold(np.arange(len(rid))*stride,combine(app[i][rid],trace,mask,cal),n)
        np.testing.assert_allclose(replay,original[i],rtol=1e-10,atol=1e-10)
        identity_errors.append(float(np.max(np.abs(replay-original[i]))))
        base=original[i];loc=starts(n,T,D);nmax=window_max(base,loc,D)
        normal.append({'id':i,'frames':n,'grid_windows':len(loc),'grid_alarm_rates':(nmax[:,None,:]>thresholds[None,:,:]).mean(0).tolist() if len(loc) else None,
            'frame_fpr_primary':(base>primary).mean(0).tolist(),'frame_fpr_frame_q99':(base>frame_threshold).mean(0).tolist()})
        # Reference original streams permit exact downstream re-evaluation.
        np.savez_compressed(output/f'{i.split("/")[-1]}_normal.npz',scores=base,grid_starts=loc,grid_maxima=nmax)
        for c in [c for c in selected if c['source_id']==i]:
            o,L=c['onset'],c['length'];mapping=edit_map(n,o,L,c['edit']);assert len(mapping)==c['output_frames']
            # No score after the preregistered window is needed; causal prefixes
            # give exactly the same outputs as running the entire edited stream.
            take=np.arange(0,o+W,stride);source=mapping[take]
            trace,mask=dynamics(tracker,x[i][source],take,ar)
            score=hold(take,combine(app[i][source],trace,mask,cal),o+W)
            event=score[o:o+D];old=base[o:o+D]
            delays=[]
            for a in range(10):
                hits=np.flatnonzero(event[:,a]>primary[a]);delays.append(int(hits[0]) if len(hits) else None)
            interior_end=o+(2*L if c['edit']=='swap_adjacent_blocks' else L)
            interior_lo=o+32;interior_hi=min(interior_end,o+W)
            interior=score[interior_lo:interior_hi] if c['edit']!='skip' and interior_hi>interior_lo else None
            case_results.append({'id':c['id'],'source_id':i,'scene':scene,'fold':fold,'seed':seed,'stride':stride,
                'edit':c['edit'],'severity':c['severity'],'onset_fraction':c['onset_fraction'],
                'edited_max_W':score[o:o+W].max(0).tolist(),'original_max_W':base[o:o+W].max(0).tolist(),
                'edited_max_D':event.max(0).tolist(),'original_max_D':old.max(0).tolist(),
                'delay_primary':delays,'interior_frames':len(interior) if interior is not None else 0,
                'interior_alarm_primary':(interior.max(0)>primary).tolist() if interior is not None else None,
                'initial_transient_max':score[o:min(o+32,o+W)].max(0).tolist()})
        for speed in [.8,1.,1.2]:
            mapping=speed_map(n,speed);take=np.arange(0,len(mapping),stride);source=mapping[take]
            trace,mask=dynamics(tracker,x[i][source],take,ar)
            score=hold(take,combine(app[i][source],trace,mask,cal),len(mapping))
            vmax=window_max(score,starts(len(mapping),T,D),D)
            speed_results.append({'source_id':i,'speed':speed,'output_frames':len(mapping),'windows':len(vmax),
                'frame_alarm_primary':(score>primary).mean(0).tolist(),'window_alarm_primary':(vmax>primary).mean(0).tolist() if len(vmax) else None})
        print(f'  E9S s{stride} seed{seed} {scene}/f{fold} {i}: {sum(c["source_id"]==i for c in selected)} edits',flush=True)
    dest=root/'results/E9S'/f'stride{stride}'/f'seed{seed}'/scene/f'fold{fold}'
    compact(dest/'cases.json',case_results);compact(dest/'normal.json',normal);compact(dest/'speed_stress.json',speed_results)
    write_json(dest/'run.json',{'status':'complete','scene':scene,'seed':seed,'fold':fold,'stride':stride,'splits':splits,
        'T':T,'D':D,'W':W,'arms':ARMS,'threshold_quantiles':QUANTILES,'window_thresholds':thresholds.tolist(),'frame_thresholds':frame_threshold.tolist(),
        'threshold_window_counts':calibration_counts,'AR1_selection':selection,'AR1_selected_ridge':chosen,
        'identity_max_score_error':max(identity_errors),'edited_cases':len(case_results),'normal_sources':len(normal),
        'code_fingerprint':code_fingerprint(),'protocol_sha256':fingerprint(cfg),'cache_signatures':{i:str(d['signature']) for i,d in data.items()},
        'wall_seconds':time.perf_counter()-start,'claim_scope':'Controlled temporal edits, not verified industrial fault types or phase accuracy'})
    print(f'E9S stride={stride} seed={seed} {scene} fold={fold}: {len(case_results)} edits, {time.perf_counter()-start:.1f}s',flush=True)
