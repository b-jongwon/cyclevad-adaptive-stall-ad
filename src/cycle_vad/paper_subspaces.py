"""E12 separates conditional means from shared/phase-specific covariance."""
import hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
from .paper_experiments import load_context,read,calibrate
from .phase_experiment import compact
from .data import balanced_sample,hold,labels_for,write_json,fingerprint
from .model import ResidualSpace,TailCalibrator
from .tracker import fourier
from .metrics import frame_metrics
from .pipeline import code_fingerprint

SPECS=[('S0','global',0),('S1','continuous',0),('S2_K4','bins4',0),('S2_K8','bins8',0),
       ('S3_K4','continuous',4),('S3_K8','continuous',8),('S4_K4','bins4',4),('S4_K8','bins8',8),('S5','continuous',None)]

def nested_subset(ids,fraction,seed):
    order=sorted(ids,key=lambda i:hashlib.sha256(json.dumps([seed,i],separators=(',',':')).encode()).hexdigest())
    return order[:min(len(order),max(2,int(np.ceil(len(order)*fraction))))]
def bins(theta,k):return np.minimum((np.mod(theta,1)*k).astype(int),k-1)
def equal_rank(theta,owner,dim):
    limits=[64,dim-1,len(theta)-2];coverage={};available=True
    for k in [4,8]:
        b=bins(theta,k);counts=np.bincount(b,minlength=k);videos=[len(np.unique(owner[b==j])) for j in range(k)]
        coverage[str(k)]={'sample_counts':counts.tolist(),'source_counts':videos};limits.append(k*(int(counts.min())-2))
        available &= min(videos)>=2
    rank=max(0,8*(min(limits)//8));return rank if available else 0,coverage

def mean_coefficients(z,theta,kind,h,lam):
    if kind=='global':return z.mean(0,keepdims=True)
    if kind.startswith('bins'):
        k=int(kind[4:]);b=bins(theta,k)
        if np.bincount(b,minlength=k).min()==0:raise ValueError('Empty phase mean bin')
        return np.stack([z[b==j].mean(0) for j in range(k)])
    b=fourier(theta,h);pen=np.eye(b.shape[1])*lam*len(b);pen[0,0]=1e-8
    return np.linalg.solve(b.T@b+pen,b.T@z).astype(np.float32)

class AppearanceHead:
    def __init__(self,kind,k,h,lam):self.kind=kind;self.k=k;self.h=h;self.lam=lam
    def mean(self,theta):
        if self.kind=='global':return np.repeat(self.coef,len(theta),axis=0)
        if self.kind.startswith('bins'):return self.coef[bins(theta,int(self.kind[4:]))]
        return fourier(theta,self.h)@self.coef
    def fit(self,z,theta,rank,seed):
        self.coef=mean_coefficients(z,theta,self.kind,self.h,self.lam);res=z-self.mean(theta);self.spaces=[]
        if self.k is None:return self
        groups=[np.arange(len(z))] if self.k==0 else [np.flatnonzero(bins(theta,self.k)==j) for j in range(self.k)]
        for idx in groups:self.spaces.append(ResidualSpace().fit(res[idx],rank if self.k==0 else rank//self.k,variance=1.,seed=seed))
        assert sum(len(s.basis) for s in self.spaces)==rank
        return self
    def raw(self,z,theta):
        residual=z-self.mean(theta);mse=np.mean(residual**2,axis=1)
        if self.k is None:return mse[:,None],mse
        out=np.zeros((len(z),2));b=np.zeros(len(z),int) if self.k==0 else bins(theta,self.k)
        for j,space in enumerate(self.spaces):
            idx=np.flatnonzero(b==j)
            if len(idx):a,c,_=space.score(residual[idx]);out[idx,0]=a;out[idx,1]=c
        return out,mse
    def calibrate(self,arrays):
        v=np.concatenate(arrays);self.cal=[TailCalibrator().fit(v[:,k]) for k in range(v.shape[1])];return self
    def score(self,z,theta):
        raw,_=self.raw(z,theta);return np.maximum.reduce([cal.score(raw[:,k]) for k,cal in enumerate(self.cal)])
    def storage(self):
        model=int(self.coef.nbytes+sum(s.mean.nbytes+s.basis.nbytes+s.eigen.nbytes for s in self.spaces))
        calibration=sum(c.reference.nbytes+8 for c in self.cal)
        return {'model_array_bytes':model,'calibration_array_bytes':int(calibration),'head_array_bytes':int(model+calibration),
                'actual_total_rank':sum(len(s.basis) for s in self.spaces)}

def run_e12(root,rows,scene,fold,seed,splits):
    start=time.perf_counter();cfg=read(root/'configs/experiments/followup/paper_stage1_v1.json');ctx=load_context(root,rows,scene,fold,seed,splits)
    previous=read(root/'results/E11/seed42'/scene/f'fold{fold}'/'run.json');h,lam=previous['harmonics'],previous['ridge']
    out=root/'runs/E12'/f'seed{seed}'/scene/f'fold{fold}';out.mkdir(parents=True,exist_ok=True)
    ids=sum([v for k,v in ctx['groups'].items() if k not in ['fit','validation']],[])
    scorebank={i:{} for i in ids};pervideo=[];metrics=[];models=[];subsets=[]
    for fraction in [.25,.5,1.]:
        chosen=nested_subset(splits['fit'],fraction,seed)
        combined=[np.c_[ctx['trace'][i]['angle'].astype(np.float32),np.full(len(ctx['idx'][i]),j,dtype=np.float32),ctx['z'][i][::2]].astype(np.float32) for j,i in enumerate(chosen)]
        sample=balanced_sample(combined,4096,seed);theta=sample[:,0];owner=sample[:,1].astype(int);z=sample[:,2:]
        rank,coverage=equal_rank(theta,owner,z.shape[1]);subsets.append({'fraction':fraction,'fit_ids':chosen,'fit_samples':len(z),'common_rank':rank,'coverage':coverage})
        for name,kind,k in SPECS:
            key=f'{name}_f{int(fraction*100)}'
            if k is not None and rank<8:
                models.append({'arm':name,'fraction':fraction,'status':'unavailable','reason':'Common rank/phase source coverage insufficient'});continue
            began=time.perf_counter();head=AppearanceHead(kind,k,h,lam).fit(z,theta,rank,seed)
            raw={};mse={}
            for i in ids:raw[i],mse[i]=head.raw(ctx['z'][i][::2],ctx['trace'][i]['angle'])
            head.calibrate([raw[i] for i in splits['reference']]);full={}
            for i in ids:
                c=np.maximum.reduce([cal.score(raw[i][:,j]) for j,cal in enumerate(head.cal)])
                full[i]=hold(ctx['idx'][i],np.stack([c,np.maximum(ctx['app'][i][::2],c)],axis=1),len(ctx['z'][i]))
            T=ctx['base']['tracker'].period;thresholds,fthreshold,counts=calibrate(full,splits,T,int(np.floor(.2*T+.5)))
            for part,group in ctx['groups'].items():
                if part in ['fit','validation']:continue
                for i in group:
                    y=labels_for(ctx['lookup'][i]);scorebank[i][key]=full[i]
                    for j,readout in enumerate(['head_only','appearance_head']):
                        pervideo.append({'id':i,'partition':part,'arm':name,'fraction':fraction,'readout':readout,**frame_metrics(y,full[i][:,j],fthreshold[j]),
                            'window_q99_frame_metrics':frame_metrics(y,full[i][:,j],thresholds[3,j]),
                            'mu_mse':float(hold(ctx['idx'][i],mse[i],len(y)).mean()),'outside_mse':float(hold(ctx['idx'][i],raw[i][:,0],len(y)).mean())})
                for j,readout in enumerate(['head_only','appearance_head']):
                    yy=np.concatenate([labels_for(ctx['lookup'][i]) for i in group]);ss=np.concatenate([full[i][:,j] for i in group])
                    metrics.append({'scene':scene,'fold':fold,'seed':seed,'partition':part,'arm':name,'fraction':fraction,'readout':readout,**frame_metrics(yy,ss,fthreshold[j]),
                        'window_q99_frame_metrics':frame_metrics(yy,ss,thresholds[3,j])})
            path=out/f'{key}.joblib';joblib.dump(head,path,compress=3)
            test=splits['normal_evaluation'][0];zz=ctx['z'][test][::2][:32];tt=ctx['trace'][test]['angle'][:32]
            for j in range(3):head.score(zz[:1],tt[:1])
            timings=[]
            for _ in range(3):
                for j in range(len(zz)):
                    t=time.perf_counter();head.score(zz[j:j+1],tt[j:j+1]);timings.append((time.perf_counter()-t)*1000)
            models.append({'arm':name,'fraction':fraction,'status':'complete','mean_kind':kind,'covariance_bins':k,**head.storage(),
                'compressed_checkpoint_bytes':path.stat().st_size,'score_latency_ms_p50':float(np.median(timings)),'score_latency_ms_p95':float(np.quantile(timings,.95)),
                'latency_scope':'batch1 head mean/PCA/calibration only, no encoder/tracker/A; 96 calls, first held-out normal source; sequential shared-host measurement',
                'frame_thresholds':fthreshold.tolist(),'window_thresholds':thresholds.tolist(),'wall_seconds':time.perf_counter()-began})
        print(f'  E12 seed{seed} {scene}/f{fold} fraction{fraction}: rank{rank}',flush=True)
    for i,values in scorebank.items():
        _,part,seq=i.split('/');np.savez_compressed(out/f'{part}_{seq}.npz',labels=labels_for(ctx['lookup'][i]),**values)
    dest=root/'results/E12'/f'seed{seed}'/scene/f'fold{fold}'
    compact(dest/'metrics.json',metrics);compact(dest/'per_video.json',pervideo)
    write_json(dest/'run.json',{'status':'complete','scene':scene,'seed':seed,'fold':fold,'splits':splits,'subsets':subsets,'models':models,'harmonics':h,'ridge':lam,
        'scope':'Appearance head sample efficiency with frozen full-FIT tracker and A, not end-to-end few-shot','protocol_sha256':fingerprint(cfg),'code_fingerprint':code_fingerprint(),
        'cache_signatures':{i:str(d['signature']) for i,d in ctx['data'].items()},'wall_seconds':time.perf_counter()-start})
    print(f'E12 seed{seed} {scene}/fold{fold}: {sum(m["status"]=="complete" for m in models)} heads, {time.perf_counter()-start:.1f}s',flush=True)
