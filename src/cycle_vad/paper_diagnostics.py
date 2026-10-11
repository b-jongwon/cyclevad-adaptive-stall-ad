"""Frozen E11 seven-arm replay and component-level normal alarm diagnostics."""
import hashlib,time
from pathlib import Path
import joblib
import numpy as np
from .data import hold,labels_for,write_json,fingerprint
from .metrics import frame_metrics,event_metrics
from .paper_experiments import read,load_context,conditional,conditional_raw,fuse,calibrate,maxwindows,event_rows
from .phase_experiment import compact
from .process_experiment import edit_map,starts
from .pipeline import code_fingerprint

ARMS=['B0','B1','B2','B3','C','P','CP']
COMPONENTS=['A_outside','A_inside','A_local','C_outside','C_inside','P_alignment','P_innovation','P_progress']
Q=[.0,.1,.25,.5,.75,.9,.95,.99,1.]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def extend(scores):
    return np.column_stack([scores,np.maximum(scores[:,4],scores[:,5])])
def ties(values):
    mask=values==values.max(1,keepdims=True)
    return mask/mask.sum(1,keepdims=True)
def strata(labels,theta):
    labels=np.asarray(labels);normal=labels==0;prior=np.cumsum(labels==1)>0;t=np.arange(len(labels))
    masks={'normal':normal,'anomaly':labels==1,'normal_no_prior_anomaly':normal&~prior,
           'normal_after_first_anomaly':normal&prior,'normal_startup':normal&(t<32),'normal_steady':normal&(t>=32)}
    for k in range(4):masks[f'normal_phase{k}']=normal&(np.floor(np.mod(theta,1)*4).astype(int)==k)
    return masks
def diagnostic_record(raw,cal,score,mask,own_threshold,full_threshold):
    n=int(mask.sum())
    if not n:return None
    r,c,s=raw[mask],cal[mask],score[mask];branch=s[:,[0,4,5]];alarm=c>full_threshold
    return {'frames':n,'component_raw_quantiles':np.quantile(r,Q,axis=0).T.tolist(),
            'component_calibrated_quantiles':np.quantile(c,Q,axis=0).T.tolist(),
            'own_window_threshold_exceed':(c>own_threshold).mean(0).tolist(),
            'full_threshold_exceed':alarm.mean(0).tolist(),
            'exclusive_full_alarm':(alarm&(alarm.sum(1,keepdims=True)==1)).mean(0).tolist(),
            'component_winner_share':ties(c).mean(0).tolist(),'branch_winner_share':ties(branch).mean(0).tolist(),
            'full_fpr':float((s[:,3]>full_threshold).mean()),
            'branch_winner_given_full_alarm':ties(branch)[s[:,3]>full_threshold].mean(0).tolist() if np.any(s[:,3]>full_threshold) else None}

def run(root,rows,scene,fold,seed,splits):
    began=time.perf_counter();old=root/'results/E11'/f'seed{seed}'/scene/f'fold{fold}';baseline=read(old/'run.json')
    assert baseline['status']=='complete' and baseline['splits']==splits
    cfg=read(root/'configs/experiments/followup/e11b_e15a_v1.json')
    modelpath=root/'runs/E11'/f'seed{seed}'/scene/f'fold{fold}'/'model.joblib';m=joblib.load(modelpath)
    ctx=load_context(root,rows,scene,fold,seed,splits);T,D,W=[baseline[k] for k in ['T','D','W']]
    bank={};rawbank={};calbank={};angles={};labels={};replay_errors=[]
    for part,ids in ctx['groups'].items():
        if part in ['fit','validation']:continue
        for i in ids:
            _,partition,seq=i.split('/');indices=ctx['idx'][i];tr=ctx['trace'][i];z=ctx['z'][i]
            # A uses exactly the same full-source GEMM shape as frozen E11.
            ao,ai,_=m['base']['pooled'].score(z);al,_=m['base']['local'].score(ctx['data'][i]['patches'])
            co,ci,_=conditional_raw(z[indices],tr['angle'],m['coef'],m['h'],m['space'])
            raw=np.column_stack([ao[indices],ai[indices],al[indices],co,ci]+[tr[k] for k in ['alignment','innovation','progress']])
            cals=[m['appcal'][k] for k in ['pooled','inside','local']]+m['ccal']+[m['pcal'][k] for k in ['alignment','innovation','progress']]
            cal=np.column_stack([c.score(raw[:,j]) for j,c in enumerate(cals)])
            scores=hold(indices,fuse(cal[:,:3].max(1),cal[:,3:5].max(1),cal[:,5:].max(1)),len(z))
            with np.load(modelpath.parent/f'{partition}_{seq}.npz') as d:
                np.testing.assert_allclose(scores,d['scores'],atol=1e-9,rtol=1e-9);replay_errors.append(float(np.max(abs(scores-d['scores']))))
                bank[i]=extend(d['scores']);labels[i]=d['labels'].copy()
            rawbank[i]=hold(indices,raw,len(z));calbank[i]=hold(indices,cal,len(z));angles[i]=hold(indices,tr['angle'],len(z))
    thresholds,ft,counts=calibrate(bank,splits,T,D);primary=thresholds[3]
    np.testing.assert_array_equal(thresholds[:,:6],baseline['window_thresholds']);np.testing.assert_array_equal(ft[:6],baseline['frame_thresholds'])
    component_thresholds,_,_=calibrate(calbank,splits,T,D)
    out=root/'runs/E11B'/f'seed{seed}'/scene/f'fold{fold}';out.mkdir(parents=True,exist_ok=True)
    dout=root/'runs/E15A'/f'seed{seed}'/scene/f'fold{fold}';dout.mkdir(parents=True,exist_ok=True)
    pervideo=[];metrics=[];normal=[];events=[];diagnostics=[]
    for part,ids in ctx['groups'].items():
        if part in ['fit','validation']:continue
        for i in ids:
            _,partition,seq=i.split('/');y=labels[i];s=bank[i]
            np.savez_compressed(out/f'{partition}_{seq}.npz',labels=y,scores=s)
            np.savez_compressed(dout/f'{partition}_{seq}.npz',labels=y,raw=rawbank[i],calibrated=calbank[i],theta=angles[i],scores=s)
            for a,arm in enumerate(ARMS):
                pervideo.append({'id':i,'partition':part,'arm':arm,**frame_metrics(y,s[:,a],primary[a]),**event_metrics(y,s[:,a],primary[a]),'frame_q99':frame_metrics(y,s[:,a],ft[a])})
            if part=='historical_test':events+=event_rows(i,y,s,primary,D)
            if part=='normal_evaluation':
                vmax=maxwindows(s,starts(len(s),T,D),D)
                normal.append({'id':i,'frames':len(y),'grid_windows':len(vmax),'grid_alarm_rates':(vmax[:,None,:]>thresholds[None,:,:]).mean(0).tolist(),
                               'frame_fpr_primary':(s>primary).mean(0).tolist(),'frame_fpr_frame_q99':(s>ft).mean(0).tolist()})
            for name,mask in strata(y,angles[i]).items():
                rec=diagnostic_record(rawbank[i],calbank[i],s,mask,component_thresholds[3],primary[3])
                if rec is not None:
                    # Full distributions per normal/anomaly cohort; smaller strata retain rates.
                    if name not in ['normal','anomaly']:
                        rec.pop('component_raw_quantiles');rec.pop('component_calibrated_quantiles')
                    diagnostics.append({'id':i,'partition':part,'stratum':name,**rec})
        for a,arm in enumerate(ARMS):
            y=np.concatenate([labels[i] for i in ids]);s=np.concatenate([bank[i][:,a] for i in ids])
            metrics.append({'scene':scene,'seed':seed,'fold':fold,'partition':part,'arm':arm,**frame_metrics(y,s,primary[a]),'frame_q99':frame_metrics(y,s,ft[a])})
    specs={c['id']:c for c in read(root/'configs/experiments/followup/e9_cases_v2.json') if c['eligible']};cases=[];windows=[];prefix=[]
    for oldcase in read(old/'cases.json'):
        spec=specs[oldcase['id']];i=spec['source_id'];o,L=spec['onset'],spec['length'];n=len(ctx['z'][i]);take=np.arange(0,o+W,2)
        src=edit_map(n,o,L,spec['edit'])[take];tr=m['base']['tracker'].predict_latent(ctx['x'][i][src],take)
        c=conditional(ctx['z'][i][src],tr['angle'],m['coef'],m['h'],m['space'],m['ccal']);p=np.maximum.reduce([m['pcal'][k].score(tr[k]) for k in m['pcal']])
        scores=extend(hold(take,fuse(ctx['app'][i][src],c,p),o+W));np.testing.assert_allclose(scores[:o],bank[i][:o],atol=1e-9,rtol=1e-9)
        prefix.append(float(abs(scores[:o]-bank[i][:o]).max()));window=scores[o:o+W];windows.append(window)
        for key,vals in [('edited_max_D',window[:D].max(0)),('edited_max_W',window.max(0))]:np.testing.assert_allclose(vals[:6],oldcase[key],atol=1e-9,rtol=1e-9)
        delays=[int(v[0]) if len(v:=np.flatnonzero(window[:D,a]>primary[a])) else None for a in range(7)]
        assert delays[:6]==oldcase['delay_primary']
        ilo=o+32;ihi=min(o+(2*L if spec['edit']=='swap_adjacent_blocks' else L),o+W)
        interior=scores[ilo:ihi] if spec['edit']!='skip' and ihi>ilo else None
        rec={k:oldcase[k] for k in ['id','source_id','scene','fold','seed','edit','severity','onset_fraction']}
        rec.update(edited_max_W=window.max(0).tolist(),original_max_W=bank[i][o:o+W].max(0).tolist(),edited_max_D=window[:D].max(0).tolist(),
                   original_max_D=bank[i][o:o+D].max(0).tolist(),delay_primary=delays,alarm_frames_W=(window>primary).sum(0).tolist(),
                   interior_frames=len(interior) if interior is not None else 0,interior_alarm_primary=(interior.max(0)>primary).tolist() if interior is not None else None,
                   initial_transient_max=window[:min(32,W)].max(0).tolist())
        cases.append(rec)
    np.savez_compressed(out/'edited_windows.npz',ids=np.array([c['id'] for c in cases]),scores=np.stack(windows))
    meta={'status':'complete','scene':scene,'fold':fold,'seed':seed,'splits':splits,'T':T,'D':D,'W':W,'arms':ARMS,'window_thresholds':thresholds.tolist(),
          'frame_thresholds':ft.tolist(),'threshold_window_counts':counts,'edited_cases':len(cases),'identity_max_error':max(replay_errors),'score_prefix_max_error':max(prefix),
          'frozen_checkpoint_sha256':sha(modelpath),'baseline_run_sha256':sha(old/'run.json'),'protocol_sha256':fingerprint(cfg),'code_fingerprint':code_fingerprint(),'wall_seconds':time.perf_counter()-began}
    dest=root/'results/E11B'/f'seed{seed}'/scene/f'fold{fold}'
    for name,rows_ in [('per_video',pervideo),('metrics',metrics),('normal',normal),('historical_events',events),('cases',cases)]:compact(dest/f'{name}.json',rows_)
    write_json(dest/'run.json',meta)
    dest=root/'results/E15A'/f'seed{seed}'/scene/f'fold{fold}';compact(dest/'per_video_strata.json',diagnostics)
    write_json(dest/'run.json',{**meta,'components':COMPONENTS,'quantiles':Q,'component_window_thresholds':component_thresholds.tolist(),
                              'scope':'frozen-model descriptive diagnostic; own-component and Full thresholds are different operating points; no calibration changed'})
    print(f'{scene}/f{fold}/seed{seed}: E11B {len(cases)} edits; E15A {len(diagnostics)} strata; {time.perf_counter()-began:.1f}s',flush=True)
