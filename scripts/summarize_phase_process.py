"""Frozen E8B/E9S aggregation and paired source-cluster uncertainty."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from bootstrap_followup import weighted_metrics
from cycle_vad.data import write_json
ROOT=Path(__file__).resolve().parents[1]
SCENES=['R01','R02','R03','R04'];N=2000
OFFSETS=[f'M4_{i}' for i in range(1101,1106)]
READOUTS=['conditional_only','appearance_conditional']
EDITS=['freeze','reverse','skip','swap_adjacent_blocks'];SEVERITIES=[.05,.1,.2]
def read(p):return json.loads(Path(p).read_text())
def interval(x):
    x=np.asarray(x);return {'estimate':float(x[0]),'ci95':np.nanquantile(x[1:],[.025,.975]).tolist(),'finite_resamples':int(np.isfinite(x[1:]).sum())}
def draws(n,rng):return np.column_stack([np.ones(n),rng.multinomial(n,np.full(n,1/n),N).T])
def e8():
    files=sorted((ROOT/'results/E8B').glob('seed*/R*/fold*/run.json'));assert len(files)==60,len(files)
    allmetrics=[];videos=[]
    for p in files:
        r=read(p);assert r['status']=='complete'
        allmetrics+=read(p.with_name('metrics.json'));videos+=read(p.with_name('per_video.json'))
    arms=list(dict.fromkeys(r['arm'] for r in allmetrics));summary=[]
    for seed in [42,43,44]:
        for arm in arms:
            for out in READOUTS:
                hist=[r for r in allmetrics if r['seed']==seed and r['arm']==arm and r['readout']==out and r['partition']=='historical_test']
                if not hist:continue
                assert len(hist)==4
                normal={}
                for scene in SCENES:
                    vs=[r for r in videos if r['seed']==seed and r['arm']==arm and r['readout']==out and r['scene']==scene and r['partition']=='normal_evaluation']
                    assert len(set(r['id'] for r in vs))==len(vs)
                    normal[scene]={'videos':len(vs),'frame_fpr':float(np.average([r['fpr'] for r in vs],weights=[r['frames'] for r in vs])),
                        'video_fpr':float(np.mean([r['fpr'] for r in vs])),'mu_mse':float(np.mean([r['mu_mse'] for r in vs])),
                        'outside_mse':float(np.mean([r['outside_mse'] for r in vs])),'rank_mean':float(np.mean([r['rank'] for r in vs]))}
                summary.append({'seed':seed,'arm':arm,'readout':out,'historical_macro_ap':float(np.mean([r['ap'] for r in hist])),
                    'historical_macro_auroc':float(np.mean([r['auroc'] for r in hist])),'historical_by_scene':{r['scene']:r for r in hist},'normal_by_scene':normal})
    write_json(ROOT/'results/E8B/summary.json',summary)
    rng=np.random.default_rng(20261009);boot={};point_checks=0
    main=['M0','M1','M2','M3']+OFFSETS
    for scene in SCENES:
        ys=[];owners=[];scores={f'{a}__{o}':[] for a in main for o in READOUTS};ids=[]
        for p in sorted((ROOT/'runs/E8B/seed42'/scene/'fold0/scores').glob('testing_*.npz')):
            with np.load(p) as d:
                mask=d['labels']>=0
                if not mask.any():continue
                ids.append(p.stem);ys.append(d['labels'][mask]);owners.append(np.full(mask.sum(),len(ids)-1))
                for k in scores:scores[k].append(d[k][mask])
        y=np.concatenate(ys);owner=np.concatenate(owners);counts=draws(len(ids),rng);boot[scene]={}
        for k,v in scores.items():
            v=np.concatenate(v);ap,auc=weighted_metrics(y,v,owner,counts)
            np.testing.assert_allclose([ap[0],auc[0]],[average_precision_score(y,v),roc_auc_score(y,v)],atol=1e-10)
            arm,out=k.split('__');expected=next(r for r in allmetrics if r['seed']==42 and r['scene']==scene and r['fold']==0 and r['partition']=='historical_test' and r['arm']==arm and r['readout']==out)
            np.testing.assert_allclose([ap[0],auc[0]],[expected['ap'],expected['auroc']],atol=1e-10);point_checks+=2
            boot[scene][k]={'ap':ap,'auroc':auc}
    contrasts=[]
    for out in READOUTS:
        for control in ['M0','M1','M2','M4_mean']:
            controls=OFFSETS if control=='M4_mean' else [control]
            for metric in ['ap','auroc']:
                diffs={s:100*(boot[s][f'M3__{out}'][metric]-np.mean([boot[s][f'{a}__{out}'][metric] for a in controls],axis=0)) for s in SCENES}
                contrasts.append({'readout':out,'contrast':f'M3-{control}','metric':metric+'_gain_pp','macro':interval(np.mean(list(diffs.values()),axis=0)),
                    'by_scene':{s:interval(v) for s,v in diffs.items()}})
    normal_boot=[]
    for scene in SCENES:
        vs=[r for r in videos if r['seed']==42 and r['scene']==scene and r['partition']=='normal_evaluation' and r['arm']=='M0' and r['readout']==READOUTS[0]]
        ids=[r['id'] for r in vs];count=draws(len(ids),rng);weights=np.array([r['frames'] for r in vs])[:,None]*count
        for out in READOUTS:
            rates={}
            for arm in main:
                lookup={r['id']:r['fpr'] for r in videos if r['seed']==42 and r['scene']==scene and r['partition']=='normal_evaluation' and r['arm']==arm and r['readout']==out}
                rates[arm]=np.array([lookup[i] for i in ids])@weights/weights.sum(0)
            for control in ['M0','M4_mean']:
                other=np.mean([rates[a] for a in OFFSETS],axis=0) if control=='M4_mean' else rates[control]
                normal_boot.append({'scene':scene,'readout':out,'contrast':f'M3-{control}','frame_fpr_gain_pp':interval(100*(rates['M3']-other))})
    write_json(ROOT/'results/E8B/paired_bootstrap.json',{'resamples':N,'seed':20261009,'unit':'source file, stratified by scene, conditional on fitted seed42 models; offsets clustered',
        'point_metric_checks':point_checks,'historical':contrasts,'normal_oof':normal_boot})
    print('E8B aggregation: 60 units,',len(summary),'summary rows,',point_checks,'bootstrap/sklearn checks',flush=True)


def e9(stride,seed,bootstrap=False):
    base=ROOT/'results/E9S'/f'stride{stride}'/f'seed{seed}';runs=sorted(base.glob('R*/fold*/run.json'));assert len(runs)==20,(stride,seed,len(runs))
    cases=[];normal=[];stress=[];runmap={};normal_scores={}
    for p in runs:
        r=read(p);assert r['status']=='complete' and r['identity_max_score_error']<=1e-10
        runmap[(r['scene'],r['fold'])]=r
        c=read(p.with_name('cases.json'));assert len(c)==r['edited_cases'];cases+=c
        normal.extend(dict(v,scene=r['scene'],fold=r['fold']) for v in read(p.with_name('normal.json')))
        stress.extend(dict(v,scene=r['scene'],fold=r['fold']) for v in read(p.with_name('speed_stress.json')))
    expected={c['id'] for c in read(ROOT/'configs/experiments/followup/e9_cases_v2.json') if c['eligible']}
    assert len(cases)==len(expected)==3910 and {c['id'] for c in cases}==expected
    assert len(normal)==111 and len(set(v['id'] for v in normal))==111
    rng=np.random.default_rng(20261009);result_groups=[];scene_boot={};normal_result={};speed_result={};nboot=N+1 if bootstrap else 1
    for scene in SCENES:
        nv=sorted([r for r in normal if r['scene']==scene],key=lambda v:v['id']);ids=[v['id'] for v in nv];idmap={v:i for i,v in enumerate(ids)}
        count=draws(len(ids),rng) if bootstrap else np.ones((len(ids),1));scene_metrics={}
        grid=np.array([r['grid_alarm_rates'] for r in nv]);frame=np.array([r['frame_fpr_primary'] for r in nv]);fweights=np.array([r['frames'] for r in nv])
        normal_result[scene]={'videos':len(nv),'windows':sum(v['grid_windows'] for v in nv),'grid_alarm_rates':grid.mean(0).tolist(),
            'frame_fpr_primary':np.average(frame,axis=0,weights=fweights).tolist(),'video_frame_fpr_primary':frame.mean(0).tolist(),
            'frame_fpr_frame_q99':np.average([r['frame_fpr_frame_q99'] for r in nv],axis=0,weights=fweights).tolist()}
        grid_boot=np.einsum('ia,ib->ab',grid[:,3,:],count)/count.sum(0)
        speed_result[scene]={str(speed):{'window_alarm':np.mean([r['window_alarm_primary'] for r in stress if r['scene']==scene and r['speed']==speed],axis=0).tolist(),
            'frame_alarm':np.mean([r['frame_alarm_primary'] for r in stress if r['scene']==scene and r['speed']==speed],axis=0).tolist()} for speed in [.8,1.,1.2]}
        scene_curves=[];scene_frames=[]
        for edit in EDITS:
            for sev in SEVERITIES:
                numer={k:np.zeros((10,nboot)) for k in ['ap','auroc','event','matched_far']};mass=np.zeros(nboot)
                curve_num=np.zeros((2,5,10));frame_num=np.zeros((2,10));original_case_weight=[];group_cases=[]
                for fold in range(5):
                    cs=[c for c in cases if c['scene']==scene and c['fold']==fold and c['edit']==edit and c['severity']==sev]
                    if not cs:continue
                    own=np.array([idmap[c['source_id']] for c in cs]);unique,countsper=np.unique(own,return_counts=True)
                    multiplicity=np.bincount(own,minlength=len(ids));scaled=count/np.maximum(1,multiplicity[:,None]);m=count[unique].sum(0);mass+=m
                    weights=(1/multiplicity[own]);original_case_weight.extend(weights);group_cases+=cs
                    r=runmap[(scene,fold)];thr=np.array(r['window_thresholds']);ft=np.array(r['frame_thresholds'])
                    ed=np.array([c['edited_max_D'] for c in cs]);orig=np.array([c['original_max_D'] for c in cs]);pos=np.array([c['edited_max_W'] for c in cs]);neg=np.array([c['original_max_W'] for c in cs])
                    y=np.r_[np.ones(len(cs)),np.zeros(len(cs))];owner=np.r_[own,own]
                    for a in range(10):
                        ap,auc=weighted_metrics(y,np.r_[pos[:,a],neg[:,a]],owner,scaled)
                        if bootstrap:
                            np.testing.assert_allclose([ap[0],auc[0]],[average_precision_score(y,np.r_[pos[:,a],neg[:,a]],sample_weight=np.r_[weights,weights]),roc_auc_score(y,np.r_[pos[:,a],neg[:,a]],sample_weight=np.r_[weights,weights])],atol=1e-10)
                        numer['ap'][a]+=np.nan_to_num(ap)*m;numer['auroc'][a]+=np.nan_to_num(auc)*m
                    events=ed>thr[3];fars=orig>thr[3]
                    np.testing.assert_array_equal(events,np.array([[v is not None for v in c['delay_primary']] for c in cs]))
                    # Each source has unit total weight across its eligible onsets.
                    numer['event']+=events.T@scaled[own];numer['matched_far']+=fars.T@scaled[own]
                    curve_num[0]+=np.einsum('i,iqa->qa',weights,ed[:,None,:]>thr[None,:,:]);curve_num[1]+=np.einsum('i,iqa->qa',weights,orig[:,None,:]>thr[None,:,:])
                    frame_num[0]+=weights@(ed>ft);frame_num[1]+=weights@(orig>ft)
                assert mass[0]>0
                values={k:np.divide(v,mass[None,:],out=np.full_like(v,np.nan),where=mass[None,:]>0) for k,v in numer.items()}
                scene_metrics[(edit,sev)]=values
                curve=curve_num/mass[0];frameop=frame_num/mass[0];scene_curves.append(curve);scene_frames.append(frameop)
                # Conditional source-balanced delay; missing detections stay misses.
                weights=np.array(original_case_weight);delays=np.array([[np.nan if v is None else v for v in c['delay_primary']] for c in group_cases]);delaymean=[];delaymedian=[]
                interior=np.array([c['interior_frames']>0 for c in group_cases]);ir=[]
                for a in range(10):
                    mask=np.isfinite(delays[:,a]);w=weights[mask];v=delays[mask,a]
                    delaymean.append(float(np.average(v,weights=w)) if len(v) else None)
                    if len(v):
                        order=np.argsort(v);delaymedian.append(float(v[order][min(np.searchsorted(np.cumsum(w[order]),w.sum()/2),len(v)-1)]))
                    else:delaymedian.append(None)
                    hits=np.array([c['interior_alarm_primary'][a] for c in group_cases if c['interior_frames']>0])
                    ir.append(float(np.average(hits,weights=weights[interior])) if len(hits) else None)
                result_groups.append({'scene':scene,'edit':edit,'severity':sev,'cases':len(group_cases),'eligible_sources':int(mass[0]),
                    **{k:v[:,0].tolist() for k,v in values.items()},'operating_curves_event_matchedfar':curve.tolist(),'frameq99_event_matchedfar':frameop.tolist(),
                    'conditional_delay_mean_frames':delaymean,'conditional_delay_median_frames':delaymedian,'interior_cases':int(interior.sum()),'interior_hit_rate':ir})
        scene_boot[scene]={k:np.mean([v[k] for v in scene_metrics.values()],axis=0) for k in ['ap','auroc','event','matched_far']}
        scene_boot[scene]['grid_far']=grid_boot
        normal_result[scene]['operating_curves_event_matchedfar']=np.mean(scene_curves,axis=0).tolist();normal_result[scene]['frameq99_event_matchedfar']=np.mean(scene_frames,axis=0).tolist()
    macro={k:np.mean([scene_boot[s][k] for s in SCENES],axis=0) for k in scene_boot[SCENES[0]]}
    summary={'stride':stride,'seed':seed,'cases':len(cases),'normal_sources':len(normal),'arms':[f'P{i}' for i in range(10)],'weighted_prevalence':.5,
        'macro':{k:v[:,0].tolist() for k,v in macro.items()},'by_scene':{s:{k:v[:,0].tolist() for k,v in vals.items()} for s,vals in scene_boot.items()},
        'normal':normal_result,'speed_stress':speed_result,'groups':result_groups}
    write_json(base/'summary.json',summary)
    if bootstrap:
        pairs=[(4,0),(4,5),(4,7),(8,1),(4,9)];contrasts=[]
        for a,b in pairs:
            for metric in macro:
                contrasts.append({'contrast':f'P{a}-P{b}','metric':metric+'_gain_pp','macro':interval(100*(macro[metric][a]-macro[metric][b])),
                    'by_scene':{s:interval(100*(scene_boot[s][metric][a]-scene_boot[s][metric][b])) for s in SCENES}})
        write_json(base/'paired_bootstrap.json',{'resamples':N,'seed':20261009,'unit':'paired source-file bootstrap within scene; derivatives remain clustered; conditional on fitted models',
            'contrasts':contrasts,'point_auc_ap_sklearn_checks':len(result_groups)*5*10*2})
    print(f'E9S stride{stride} seed{seed}: {len(cases)} cases, {len(normal)} sources, bootstrap={bootstrap}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['E8B','E9S'],required=True);p.add_argument('--stride',type=int,default=2);p.add_argument('--seed',type=int,default=42);p.add_argument('--bootstrap',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='E8B':e8()
        else:e9(a.stride,a.seed,a.bootstrap)
