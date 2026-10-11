"""Seven-arm paired comparisons and descriptive frozen-component diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from cycle_vad.data import write_json
from cycle_vad.phase_experiment import compact
from cycle_vad.paper_diagnostics import ARMS,COMPONENTS
from bootstrap_followup import weighted_metrics
from summarize_phase_process import draws,interval
from summarize_paper_stage1 import e11_synthetic
ROOT=Path(__file__).resolve().parents[1];SCENES=['R01','R02','R03','R04'];PAIRS=[(6,4),(3,6),(3,4)]
def read(p):return json.loads(Path(p).read_text())

def historical(seed,bootstrap):
    base=ROOT/'results/E11B'/f'seed{seed}';runs=sorted(base.glob('R*/fold*/run.json'));assert len(runs)==20
    metrics=[];events=[]
    for p in runs:
        metrics+=read(p.with_name('metrics.json'));events.extend(dict(e,scene=p.parent.parent.name) for e in read(p.with_name('historical_events.json')))
    byscene={sc:[next(r for r in metrics if r['scene']==sc and r['partition']=='historical_test' and r['arm']==a) for a in ARMS] for sc in SCENES}
    result={'seed':seed,'arms':ARMS,'by_scene':byscene,'macro':{k:np.mean([[r[k] for r in byscene[sc]] for sc in SCENES],axis=0).tolist() for k in ['ap','auroc']}}
    result['leave_one_scene_out']={drop:{k:np.mean([[r[k] for r in byscene[sc]] for sc in SCENES if sc!=drop],axis=0).tolist() for k in ['ap','auroc']} for drop in SCENES}
    overlap=[]
    for sc in SCENES:
        es=[e for e in events if e['scene']==sc];hit=np.array([[v is not None for v in e['delay_primary']] for e in es])
        c,p,cp,full=hit[:,4],hit[:,5],hit[:,6],hit[:,3]
        overlap.append({'scene':sc,'events':len(es),'detected':hit.sum(0).tolist(),'C_only':int((c&~p).sum()),'P_only':int((p&~c).sum()),'both_C_P':int((p&c).sum()),'neither_C_P':int((~p&~c).sum()),
                        'CP_lost_from_union':int(((c|p)&~cp).sum()),'CP_added_over_C':int((cp&~c).sum()),'CP_lost_from_C':int((~cp&c).sum()),
                        'Full_added_over_CP':int((full&~cp).sum()),'Full_lost_from_CP':int((~full&cp).sum())})
    result['events']=overlap;write_json(base/'historical_summary.json',result)
    if bootstrap:
        rng=np.random.default_rng(20261009);boot={}
        for sc in SCENES:
            ds=[]
            for f in sorted((ROOT/'runs/E11B'/f'seed{seed}'/sc/'fold0').glob('testing_*.npz')):
                with np.load(f) as d:
                    valid=d['labels']>=0
                    if valid.any():ds.append((d['labels'][valid],d['scores'][valid]))
            y=np.concatenate([v[0] for v in ds]);s=np.concatenate([v[1] for v in ds]);owner=np.concatenate([np.full(len(v[0]),j) for j,v in enumerate(ds)]);counts=draws(len(ds),rng);boot[sc]={}
            for a in range(len(ARMS)):
                ap,auc=weighted_metrics(y,s[:,a],owner,counts);np.testing.assert_allclose([ap[0],auc[0]],[byscene[sc][a]['ap'],byscene[sc][a]['auroc']],atol=1e-10)
                boot[sc][a]={'ap':ap,'auroc':auc}
        contrasts=[]
        for a,b in PAIRS:
            for k in ['ap','auroc']:
                vals={sc:100*(boot[sc][a][k]-boot[sc][b][k]) for sc in SCENES}
                contrasts.append({'contrast':f'{ARMS[a]}-{ARMS[b]}','metric':k+'_gain_pp','role':'secondary' if (a,b)==(3,4) else 'primary',
                                  'macro':interval(np.mean(list(vals.values()),axis=0)),'by_scene':{s:interval(v) for s,v in vals.items()},
                                  'leave_one_scene_out':{drop:interval(np.mean([v for sc,v in vals.items() if sc!=drop],axis=0)) for drop in SCENES}})
        write_json(base/'historical_bootstrap.json',{'resamples':2000,'seed':20261009,'unit':'paired source-file within scene; conditional on fitted seed42 models','contrasts':contrasts})
    # Original six arms must reproduce previously published summary values.
    old=read(ROOT/'results/E11'/f'seed{seed}'/'historical_summary.json')
    np.testing.assert_allclose(result['macro']['ap'][:6],[v['historical_macro_ap'] for v in old['arms']],atol=1e-12)
    print(f'E11B historical seed{seed}: original6 reproduced; {sum(v["events"] for v in overlap)} events',flush=True)

def synthetic_overlap(seed):
    base=ROOT/'results/E11B'/f'seed{seed}';cases=[];periods={}
    for p in base.glob('R*/fold*/run.json'):
        run=read(p);periods[(run['scene'],run['fold'])]=run['W'];cases+=read(p.with_name('cases.json'))
    groups=[]
    for sc in SCENES:
        for ed in ['freeze','reverse','skip','swap_adjacent_blocks']:
            for sev in [.05,.1,.2]:
                cs=[c for c in cases if c['scene']==sc and c['edit']==ed and c['severity']==sev];ids=sorted(set(c['source_id'] for c in cs));samples=[]
                for i in ids:
                    src=[c for c in cs if c['source_id']==i];h=np.array([[v is not None for v in c['delay_primary']] for c in src]);a,b,f=h[:,4],h[:,6],h[:,3]
                    samples.append(np.r_[(b&~a).mean(),(~b&a).mean(),(f&~b).mean(),(~f&b).mean(),np.mean([np.array(c['alarm_frames_W'])/periods[(sc,c['fold'])] for c in src],axis=0)])
                v=np.mean(samples,axis=0);groups.append({'scene':sc,'edit':ed,'severity':sev,'sources':len(ids),'CP_added_over_C':v[0],'CP_lost_from_C':v[1],'Full_added_over_CP':v[2],'Full_lost_from_CP':v[3],'alarm_fraction_W':v[4:].tolist()})
    write_json(base/'synthetic_overlap.json',{'weighting':'source equal within scene/type/severity; groups equal in macro','groups':groups,
        'macro':{k:np.mean([g[k] for g in groups],axis=0).tolist() for k in ['CP_added_over_C','CP_lost_from_C','Full_added_over_CP','Full_lost_from_CP','alarm_fraction_W']}})

def aggregate(records):
    w=np.array([r['frames'] for r in records]);out={'videos':len(records),'frames':int(w.sum())}
    keys=['full_fpr','full_threshold_exceed','own_window_threshold_exceed','exclusive_full_alarm','component_winner_share','branch_winner_share']
    for k in keys:
        arr=np.array([r[k] for r in records]);out[k+'_frame_weighted']=np.average(arr,axis=0,weights=w).tolist();out[k+'_video_equal']=arr.mean(0).tolist()
    # Alarm-conditional attribution weights by alarm frames, not all normal frames.
    selected=[r for r in records if r['branch_winner_given_full_alarm'] is not None]
    out['branch_winner_given_full_alarm']=np.average([r['branch_winner_given_full_alarm'] for r in selected],axis=0,weights=[r['frames']*r['full_fpr'] for r in selected]).tolist() if selected else None
    return out

def mixture_quantiles(arrays,qs):
    a=np.concatenate(arrays);w=np.concatenate([np.full(len(x),1/len(x)) for x in arrays]);out=[]
    for j in range(a.shape[1]):
        order=np.argsort(a[:,j],kind='stable');cum=np.cumsum(w[order]);idx=np.minimum(np.searchsorted(cum,np.array(qs)*cum[-1],side='left'),len(cum)-1);out.append(a[order[idx],j].tolist())
    return out

def diagnostics():
    groups=[];distributions=[]
    for seed in [42,43,44]:
        for sc in SCENES:
            allrows=[]
            for fold in range(5):
                p=ROOT/'results/E15A'/f'seed{seed}'/sc/f'fold{fold}';assert read(p/'run.json')['status']=='complete'
                allrows.extend(dict(v,fold=fold) for v in read(p/'per_video_strata.json'))
            for scope in ['fold0','normal_oof']:
                rs=[r for r in allrows if r['fold']==0] if scope=='fold0' else [r for r in allrows if r['partition']=='normal_evaluation']
                for part,name in sorted(set((r['partition'],r['stratum']) for r in rs)):
                    vs=[r for r in rs if r['partition']==part and r['stratum']==name];assert len({r['id'] for r in vs})==len(vs)
                    groups.append({'seed':seed,'scene':sc,'scope':scope,'partition':part,'stratum':name,**aggregate(vs)})
            for part in ['reference','threshold','normal_evaluation','historical_test']:
                rs=[r for r in allrows if r['fold']==0 and r['partition']==part and r['stratum']=='normal'];raw=[];cal=[]
                for r in rs:
                    _,partition,seq=r['id'].split('/')
                    with np.load(ROOT/'runs/E15A'/f'seed{seed}'/sc/'fold0'/f'{partition}_{seq}.npz') as d:
                        mask=d['labels']==0;raw.append(d['raw'][mask]);cal.append(d['calibrated'][mask])
                branches=[np.column_stack([x[:,:3].max(1),x[:,3:5].max(1),x[:,5:].max(1),x.max(1)]) for x in cal]
                qs=np.linspace(0,1,101).tolist();distributions.append({'seed':seed,'scene':sc,'fold':0,'partition':part,'videos':len(raw),'frames':sum(map(len,raw)),
                    'quantile_levels':qs,'raw_video_equal_quantiles':mixture_quantiles(raw,qs),'calibrated_video_equal_quantiles':mixture_quantiles(cal,qs),
                    'branch_order':['A','C','P','Full'],'branch_video_equal_quantiles':mixture_quantiles(branches,qs)})
    dest=ROOT/'results/E15A';write_json(dest/'summary.json',{'components':COMPONENTS,'groups':groups,'scope':'descriptive, no new calibrator fit; historical all-normal includes normal frames after an anomaly; phase is estimated not GT'})
    compact(dest/'distributions.json',{'weighting':'equal source weight within scene/split; fold0 only; values are mixture quantiles, not averages of video quantiles','distributions':distributions})
    print(f'E15A: {len(groups)} aggregate strata; {len(distributions)} split distributions',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['E11B','E15A'],required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--bootstrap',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='E11B':
            e11_synthetic(a.seed,a.bootstrap,stage='E11B',arms=ARMS,contrasts=PAIRS);historical(a.seed,a.bootstrap);synthetic_overlap(a.seed)
            old=read(ROOT/'results/E11'/f'seed{a.seed}'/'synthetic_summary.json');new=read(ROOT/'results/E11B'/f'seed{a.seed}'/'synthetic_summary.json')
            for k in old['macro']:np.testing.assert_allclose(new['macro'][k][:6],old['macro'][k],atol=1e-12)
        else:diagnostics()
