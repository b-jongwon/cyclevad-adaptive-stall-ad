"""Unified E11 and E12 summaries, clustered uncertainty and branch contributions."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from bootstrap_followup import weighted_metrics
from summarize_phase_process import interval,draws
from cycle_vad.data import write_json
ROOT=Path(__file__).resolve().parents[1]
SCENES=['R01','R02','R03','R04'];N=2000
EDITS=['freeze','reverse','skip','swap_adjacent_blocks'];SEVERITIES=[.05,.1,.2]
ARMS=['B0','B1','B2','B3','C','P']
def read(p):return json.loads(Path(p).read_text())
def e11_synthetic(seed,bootstrap=False,stage='E11',arms=None,contrasts=None):
    arms=ARMS if arms is None else arms;n_arms=len(arms)
    stride=2
    base=ROOT/'results'/stage/f'seed{seed}';runs=sorted(base.glob('R*/fold*/run.json'));assert len(runs)==20,(stride,seed,len(runs))
    cases=[];normal=[];stress=[];runmap={};normal_scores={}
    for p in runs:
        r=read(p);assert r['status']=='complete' and r['identity_max_error']<=1e-10
        runmap[(r['scene'],r['fold'])]=r
        c=read(p.with_name('cases.json'));assert len(c)==r['edited_cases'];cases+=c
        normal.extend(dict(v,scene=r['scene'],fold=r['fold']) for v in read(p.with_name('normal.json')))
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
        scene_curves=[];scene_frames=[]
        for edit in EDITS:
            for sev in SEVERITIES:
                numer={k:np.zeros((n_arms,nboot)) for k in ['ap','auroc','event','matched_far']};mass=np.zeros(nboot)
                curve_num=np.zeros((2,5,n_arms));frame_num=np.zeros((2,n_arms));original_case_weight=[];group_cases=[]
                for fold in range(5):
                    cs=[c for c in cases if c['scene']==scene and c['fold']==fold and c['edit']==edit and c['severity']==sev]
                    if not cs:continue
                    own=np.array([idmap[c['source_id']] for c in cs]);unique,countsper=np.unique(own,return_counts=True)
                    multiplicity=np.bincount(own,minlength=len(ids));scaled=count/np.maximum(1,multiplicity[:,None]);m=count[unique].sum(0);mass+=m
                    weights=(1/multiplicity[own]);original_case_weight.extend(weights);group_cases+=cs
                    r=runmap[(scene,fold)];thr=np.array(r['window_thresholds']);ft=np.array(r['frame_thresholds'])
                    ed=np.array([c['edited_max_D'] for c in cs]);orig=np.array([c['original_max_D'] for c in cs]);pos=np.array([c['edited_max_W'] for c in cs]);neg=np.array([c['original_max_W'] for c in cs])
                    y=np.r_[np.ones(len(cs)),np.zeros(len(cs))];owner=np.r_[own,own]
                    for a in range(n_arms):
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
                for a in range(n_arms):
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
    summary={'stride':stride,'seed':seed,'cases':len(cases),'normal_sources':len(normal),'arms':arms,'weighted_prevalence':.5,
        'macro':{k:v[:,0].tolist() for k,v in macro.items()},'by_scene':{s:{k:v[:,0].tolist() for k,v in vals.items()} for s,vals in scene_boot.items()},
        'normal':normal_result,'groups':result_groups}
    write_json(base/'synthetic_summary.json',summary)
    if bootstrap:
        pairs=contrasts if contrasts is not None else [(3,2),(3,1),(3,0)];contrasts=[]
        for a,b in pairs:
            for metric in macro:
                contrasts.append({'contrast':f'{arms[a]}-{arms[b]}','metric':metric+'_gain_pp','macro':interval(100*(macro[metric][a]-macro[metric][b])),
                    'by_scene':{s:interval(100*(scene_boot[s][metric][a]-scene_boot[s][metric][b])) for s in SCENES}})
        write_json(base/'synthetic_bootstrap.json',{'resamples':N,'seed':20261009,'unit':'paired source-file bootstrap within scene; derivatives remain clustered; conditional on fitted models',
            'contrasts':contrasts,'point_auc_ap_sklearn_checks':len(result_groups)*5*n_arms*2})
    print(f'{stage} synthetic stride{stride} seed{seed}: {len(cases)} cases, {len(normal)} sources, bootstrap={bootstrap}',flush=True)

def e11_historical(seed,bootstrap=False):
    base=ROOT/'results/E11'/f'seed{seed}';runs=sorted(base.glob('R*/fold*/run.json'));assert len(runs)==20
    metrics=[];videos=[];events=[]
    for p in runs:
        metrics+=read(p.with_name('metrics.json'));videos.extend(dict(v,scene=p.parent.parent.name,fold=int(p.parent.name[4:])) for v in read(p.with_name('per_video.json')))
        events.extend(dict(v,scene=p.parent.parent.name) for v in read(p.with_name('historical_events.json')))
    summary=[]
    for arm in ARMS:
        hist=[m for m in metrics if m['partition']=='historical_test' and m['arm']==arm];assert len(hist)==4
        normal={}
        for sc in SCENES:
            vs=[r for r in videos if r['scene']==sc and r['partition']=='normal_evaluation' and r['arm']==arm]
            assert len(set(r['id'] for r in vs))==len(vs)
            normal[sc]={'videos':len(vs),'frame_fpr':float(np.average([r['fpr'] for r in vs],weights=[r['frames'] for r in vs])),'video_fpr':float(np.mean([r['fpr'] for r in vs]))}
        summary.append({'arm':arm,'historical_macro_ap':float(np.mean([r['ap'] for r in hist])),'historical_macro_auroc':float(np.mean([r['auroc'] for r in hist])),
            'historical_macro_frameq99_fpr':float(np.mean([r['frame_q99']['fpr'] for r in hist])),
            'historical_by_scene':{r['scene']:r for r in hist},'normal_by_scene':normal})
    overlap=[]
    for sc in SCENES:
        es=[e for e in events if e['scene']==sc];hits=np.array([[v is not None for v in e['delay_primary']] for e in es]);a,p=hits[:,1],hits[:,5];union=a|p;full=hits[:,3]
        overlap.append({'scene':sc,'events':len(es),'appearance_only':int((a&~p).sum()),'process_only':int((~a&p).sum()),'both':int((a&p).sum()),'neither':int((~a&~p).sum()),
            'B3_detected':int(full.sum()),'union_detected':int(union.sum()),'B3_added_over_B1':int((full&~a).sum()),'B3_lost_from_B1':int((~full&a).sum()),
            'B3_added_over_B2':int((full&~hits[:,2]).sum()),'B3_lost_from_B2':int((~full&hits[:,2]).sum()),'B3_lost_from_union':int((~full&union).sum())})
    write_json(base/'historical_summary.json',{'seed':seed,'arms':summary,'event_overlap':overlap,'event_scope':'contiguous historical labels, deadline min(D,event length); actual anomaly types unannotated'})
    if bootstrap:
        rng=np.random.default_rng(20261009);boot={}
        for sc in SCENES:
            ds=[]
            for p in sorted((ROOT/'runs/E11'/f'seed{seed}'/sc/'fold0').glob('testing_*.npz')):
                with np.load(p) as d:
                    mask=d['labels']>=0
                    if mask.any():ds.append((d['labels'][mask],d['scores'][mask]))
            y=np.concatenate([d[0] for d in ds]);score=np.concatenate([d[1] for d in ds]);owner=np.concatenate([np.full(len(d[0]),i) for i,d in enumerate(ds)]);count=draws(len(ds),rng)
            boot[sc]={}
            for a,arm in enumerate(ARMS):
                ap,auc=weighted_metrics(y,score[:,a],owner,count);expected=next(r for r in summary if r['arm']==arm)['historical_by_scene'][sc]
                np.testing.assert_allclose([ap[0],auc[0]],[expected['ap'],expected['auroc']],atol=1e-10);boot[sc][arm]={'ap':ap,'auroc':auc}
        contrasts=[]
        for control in ['B0','B1','B2']:
            for metric in ['ap','auroc']:
                vs={sc:100*(boot[sc]['B3'][metric]-boot[sc][control][metric]) for sc in SCENES}
                contrasts.append({'contrast':f'B3-{control}','metric':metric+'_gain_pp','macro':interval(np.mean(list(vs.values()),axis=0)),'by_scene':{sc:interval(v) for sc,v in vs.items()}})
        write_json(base/'historical_bootstrap.json',{'resamples':N,'unit':'paired source-file within scene, seed42 fitted models fixed','contrasts':contrasts})
    print(f'E11 historical seed{seed}: {len(events)} events, {len(summary)} arms',flush=True)


def e12():
    paths=sorted((ROOT/'results/E12').glob('seed*/R*/fold*/run.json'));assert len(paths)==60,len(paths)
    metrics=[];videos=[];models=[]
    for p in paths:
        run=read(p);metrics+=read(p.with_name('metrics.json'))
        videos.extend(dict(v,scene=run['scene'],seed=run['seed'],fold=run['fold']) for v in read(p.with_name('per_video.json')))
        models.extend(dict(m,scene=run['scene'],seed=run['seed'],fold=run['fold']) for m in run['models'])
    summary=[]
    for seed in [42,43,44]:
        for frac in [.25,.5,1.]:
            for arm in ['S0','S1','S2_K4','S2_K8','S3_K4','S3_K8','S4_K4','S4_K8','S5']:
                md=[m for m in models if m['seed']==seed and m['fraction']==frac and m['arm']==arm];ok=[m for m in md if m['status']=='complete']
                for out in ['head_only','appearance_head']:
                    hist=[r for r in metrics if r['seed']==seed and r['fraction']==frac and r['arm']==arm and r['readout']==out and r['partition']=='historical_test']
                    norm={}
                    for sc in SCENES:
                        vs=[v for v in videos if v['seed']==seed and v['fraction']==frac and v['arm']==arm and v['readout']==out and v['partition']=='normal_evaluation' and v['scene']==sc]
                        norm[sc]={'videos':len(vs),'frame_fpr':float(np.average([v['fpr'] for v in vs],weights=[v['frames'] for v in vs])) if vs else None,
                            'video_fpr':float(np.mean([v['fpr'] for v in vs])) if vs else None,'mu_mse':float(np.mean([v['mu_mse'] for v in vs])) if vs else None,
                            'outside_mse':float(np.mean([v['outside_mse'] for v in vs])) if vs else None}
                    summary.append({'seed':seed,'fraction':frac,'arm':arm,'readout':out,'completed_units':len(ok),'unavailable_units':len(md)-len(ok),
                        'historical_scenes':len(hist),'historical_macro_ap':float(np.mean([r['ap'] for r in hist])) if len(hist)==4 else None,
                        'historical_macro_auroc':float(np.mean([r['auroc'] for r in hist])) if len(hist)==4 else None,
                        'historical_by_scene':{r['scene']:r for r in hist},'normal_by_scene':norm,
                        **{k:float(np.mean([m[k] for m in ok])) if ok else None for k in ['model_array_bytes','head_array_bytes','compressed_checkpoint_bytes','actual_total_rank','score_latency_ms_p50','score_latency_ms_p95']}})
    write_json(ROOT/'results/E12/summary.json',summary)
    # Full-data primary mechanism contrasts; same video bootstrap draws across heads.
    rng=np.random.default_rng(20261009);boot={};arms=['S0','S1','S2_K4','S2_K8','S3_K4','S3_K8','S5']
    for sc in SCENES:
        ds=[]
        for p in sorted((ROOT/'runs/E12/seed42'/sc/'fold0').glob('testing_*.npz')):
            with np.load(p) as d:
                mask=d['labels']>=0
                if mask.any():ds.append((d['labels'][mask],{a:d[f'{a}_f100'][mask] for a in arms}))
        y=np.concatenate([d[0] for d in ds]);owner=np.concatenate([np.full(len(d[0]),i) for i,d in enumerate(ds)]);count=draws(len(ds),rng);boot[sc]={}
        for arm in arms:
            for j,out in enumerate(['head_only','appearance_head']):
                score=np.concatenate([d[1][arm][:,j] for d in ds]);ap,auc=weighted_metrics(y,score,owner,count)
                expected=next(r for r in summary if r['seed']==42 and r['fraction']==1 and r['arm']==arm and r['readout']==out)['historical_by_scene'][sc]
                np.testing.assert_allclose([ap[0],auc[0]],[expected['ap'],expected['auroc']],atol=1e-10);boot[sc][(arm,out)]={'ap':ap,'auroc':auc}
    contrasts=[]
    for arm in [a for a in arms if a!='S1']:
        for out in ['head_only','appearance_head']:
            for metric in ['ap','auroc']:
                values={sc:100*(boot[sc][('S1',out)][metric]-boot[sc][(arm,out)][metric]) for sc in SCENES}
                contrasts.append({'contrast':f'S1-{arm}','readout':out,'metric':metric+'_gain_pp','macro':interval(np.mean(list(values.values()),axis=0)),'by_scene':{s:interval(v) for s,v in values.items()}})
    write_json(ROOT/'results/E12/paired_bootstrap.json',{'resamples':N,'seed':20261009,'scope':'full-data head comparison; seed42 fixed models, source-cluster paired','contrasts':contrasts})
    print(f'E12 summary: {len(models)} head attempts, {sum(m["status"]=="complete" for m in models)} completed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['E11','E12'],required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--bootstrap',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='E11':e11_synthetic(a.seed,a.bootstrap);e11_historical(a.seed,a.bootstrap)
        else:e12()
