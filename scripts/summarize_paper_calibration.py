"""Source-clustered E15BC operating results, with fixed-model paired intervals."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from cycle_vad.data import write_json
from cycle_vad.paper_calibration import SCENES,POLICIES,ALPHAS,read
from summarize_phase_process import draws,interval
ROOT=Path(__file__).resolve().parents[1]
EDITS=['freeze','reverse','skip','swap_adjacent_blocks'];SEVERITIES=[.05,.1,.2]

def ratio_boot(numerator,denominator,count):
    num=np.tensordot(numerator,count,axes=(0,0));den=np.tensordot(denominator,count,axes=(0,0))
    while den.ndim<num.ndim:den=den[None,:]
    return np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0)

def aggregate(seed):
    base=ROOT/f'results/E15BC/seed{seed}';paths=sorted(base.glob('R*/fold*/run.json'));assert len(paths)==20
    normal=[];events=[];frames=[];cases=[];ranks=[];runs=[]
    for p in paths:
        r=read(p);assert r['status']=='complete';runs.append(r)
        for name,target in [('normal',normal),('events',events),('frames',frames),('cases',cases),('ranking',ranks)]:
            target.extend(dict(v,scene=r['scene'],fold=r['fold']) for v in read(p.with_name(name+'.json')))
    assert len(cases)==7820 and len(events)==132
    assert len(normal)==222 and len({v['id'] for v in normal})==111
    rng=np.random.default_rng(20261009);boots={};scene_rows={};groups=[]
    for sc in SCENES:
        nr=[v for v in normal if v['scene']==sc];ids=sorted({v['id'] for v in nr});index={i:j for j,i in enumerate(ids)}
        count=draws(len(ids),rng) if seed==42 else np.ones((len(ids),1));B=count.shape[1]
        grid=np.zeros((len(ids),2,3,5));frame=np.zeros_like(grid);length=np.zeros(len(ids))
        for v in nr:
            j=index[v['id']];grid[j,v['g']]=v['grid_far'];frame[j,v['g']]=v['frame_fpr'];length[j]=v['frames']
        boot={'grid_far':ratio_boot(grid,np.ones(len(ids)),count),
              'normal_frame_fpr':ratio_boot(frame*length[:,None,None,None],length,count)}
        metrics_by_group=[]
        for edit in EDITS:
            for severity in SEVERITIES:
                cs=[v for v in cases if v['scene']==sc and v['edit']==edit and v['severity']==severity]
                lookup={(v['id'],v['g']):v for v in cs};source_ids=sorted({v['source_id'] for v in cs});group=[]
                arrays={k:np.zeros((len(ids),2,3,5)) for k in ['synthetic_event','matched_far','alarm_fraction_W','synthetic_added','synthetic_lost']}
                eligible=np.zeros(len(ids));delay_sum=np.zeros((2,3,5));delay_mass=np.zeros((2,3,5))
                for i in source_ids:
                    subset=[v for v in cs if v['source_id']==i and v['g']==0];eligible[index[i]]=1
                    for g in range(2):
                        vs=[lookup[(v['id'],g)] for v in subset];h=np.array([v['hit'] for v in vs],bool);c=h[:,:,0,None]
                        arrays['synthetic_event'][index[i],g]=h.mean(0)
                        arrays['matched_far'][index[i],g]=np.mean([v['matched_hit'] for v in vs],axis=0)
                        arrays['alarm_fraction_W'][index[i],g]=np.mean([v['alarm_fraction_W'] for v in vs],axis=0)
                        arrays['synthetic_added'][index[i],g]=(h&~c).mean(0);arrays['synthetic_lost'][index[i],g]=(~h&c).mean(0)
                        delay=np.array([v['delay'] for v in vs]);delay_sum[g]+=np.where(delay>=0,delay,0).sum(0)/len(vs);delay_mass[g]+=(delay>=0).sum(0)/len(vs)
                gb={k:ratio_boot(a,eligible,count) for k,a in arrays.items()};metrics_by_group.append(gb)
                groups.append({'scene':sc,'edit':edit,'severity':severity,'eligible_sources':len(source_ids),'cases':len(cs)//2,
                               **{k:v[...,0].tolist() for k,v in gb.items()},
                               'conditional_delay_mean_frames':np.divide(delay_sum,delay_mass,out=np.full_like(delay_sum,np.nan),where=delay_mass>0).tolist()})
        for k in metrics_by_group[0]:boot[k]=np.mean([v[k] for v in metrics_by_group],axis=0)
        # Historical counts: one source file owns all of its events and frames.
        hids=[]
        for p in sorted((ROOT/f'runs/E15BC/seed{seed}'/sc/'fold0').glob('testing_*.npz')):
            with np.load(p) as d:
                if np.any(d['labels']>=0):hids.append(f'{sc}/testing/{p.stem.split("_",1)[1]}')
        hindex={i:j for j,i in enumerate(hids)};hc=draws(len(hids),rng) if seed==42 else np.ones((len(hids),1))
        es=[e for e in events if e['scene']==sc];elook={(e['id'],e['g']):e for e in es}
        nums={k:np.zeros((len(hids),2,3,5)) for k in ['historical_event','retained','added','lost','retained_original','added_original','lost_original']}
        denom=np.zeros(len(hids));retained_den=np.zeros_like(nums['retained']);original_den=np.zeros_like(retained_den)
        delays=[]
        for e in [v for v in es if v['g']==0]:
            j=hindex[e['source_id']];denom[j]+=1;h=np.array([elook[(e['id'],g)]['hit'] for g in range(2)],bool)
            c=h[:,:,[0]];original=h[0:1,:,[0]]
            nums['historical_event'][j]+=h
            for name,ref,den in [('','c',retained_den),('_original','original',original_den)]:
                b=c if ref=='c' else original;den[j]+=np.broadcast_to(b,(2,3,5))
                nums['retained'+name][j]+=h&b;nums['added'+name][j]+=h&~b;nums['lost'+name][j]+=~h&b
            delays.append(np.array([elook[(e['id'],g)]['delay'] for g in range(2)]))
        for k,a in nums.items():boot[k]=ratio_boot(a,denom,hc)
        boot['retention_fraction']=np.divide(np.tensordot(nums['retained'],hc,axes=(0,0)),np.tensordot(retained_den,hc,axes=(0,0)),out=np.full((2,3,5,hc.shape[1]),np.nan),where=np.tensordot(retained_den,hc,axes=(0,0))>0)
        strata={}
        for st in ['normal','anomaly','normal_before','normal_after']:
            vals=[v for v in frames if v['scene']==sc and v['stratum']==st];nn=np.zeros((len(hids),2,3,5));dd=np.zeros(len(hids))
            for v in vals:
                j=hindex[v['id']];nn[j,v['g']]=v['alarm_count'];dd[j]=v['frames']
            bb=ratio_boot(nn,dd,hc);boot['historical_'+st+'_alarm']=bb;strata[st]={'frames':int(dd.sum()),'rate':bb[...,0].tolist()}
        scene_rows[sc]={'historical_events':int(denom.sum()),'historical_counts':{k:a.sum(0).astype(int).tolist() for k,a in nums.items()},
                        'normal_sources':len(ids),'normal_windows':sum(v['windows'] for v in nr if v['g']==0),
                        'metrics':{k:v[...,0].tolist() for k,v in boot.items()},'historical_frame_strata':strata,
                        'ranking':[v for v in ranks if v['scene']==sc],
                        'historical_delay_mean_detected_frames':np.nanmean(np.where(np.array(delays)>=0,delays,np.nan),axis=0).tolist()}
        boots[sc]=boot
    macro={k:np.mean([boots[sc][k] for sc in SCENES],axis=0) for k in boots['R01']}
    result={'seed':seed,'axes':['calibration','alpha','policy'],'calibrations':['G0','G1'],'alphas':ALPHAS,'policies':POLICIES,
            'macro':{k:v[...,0].tolist() for k,v in macro.items()},'by_scene':scene_rows,'synthetic_groups':groups,
            'historical_total_counts':{k:np.sum([v['historical_counts'][k] for v in scene_rows.values()],axis=0).tolist() for k in scene_rows['R01']['historical_counts']},
            'historical_macro_ranking':{f'G{g}_{a}':{k:float(np.mean([v[k] for v in ranks if v['g']==g and v['arm']==a])) for k in ['ap','auroc']} for g in range(2) for a in ['C','MAX']},
            'leave_one_scene_out':{drop:{k:np.mean([boots[sc][k][...,0] for sc in SCENES if sc!=drop],axis=0).tolist() for k in macro} for drop in SCENES}}
    # Exact legacy aggregation checks: source-equal, then edit/severity/scene macro.
    old=read(ROOT/f'results/E11B/seed{seed}/synthetic_summary.json')
    for k,legacy in [('synthetic_event','event'),('matched_far','matched_far'),('grid_far','grid_far')]:
        np.testing.assert_allclose(np.array(result['macro'][k])[0,0,:2],np.array(old['macro'][legacy])[[4,6]],atol=1e-12)
    write_json(base/'summary.json',clean(result))
    if seed==42:
        comparisons=[]
        for g in range(2):
            for a in range(3):
                for p in range(5):
                    for scope,(rg,rp) in [('same_G_C',(g,0)),('original_C',(0,0))]:
                        for metric in ['historical_event','grid_far','matched_far','synthetic_event','normal_frame_fpr','historical_normal_alarm']:
                            differences={sc:100*(boots[sc][metric][g,a,p]-boots[sc][metric][rg,a,rp]) for sc in SCENES}
                            comparisons.append({'g':g,'alpha':ALPHAS[a],'policy':POLICIES[p],'reference':scope,'metric':metric,
                                                'macro':interval(np.mean(list(differences.values()),axis=0)),
                                                'by_scene':{sc:interval(v) for sc,v in differences.items()},
                                                'leave_one_scene_out':{drop:interval(np.mean([v for sc,v in differences.items() if sc!=drop],axis=0)) for drop in SCENES}})
        estimates={k:[[[interval(macro[k][g,a,p]*100) for p in range(5)] for a in range(3)] for g in range(2)] for k in ['retained','added','lost','retained_original','added_original','lost_original','retention_fraction']}
        write_json(base/'bootstrap.json',clean({'unit':'paired source file within scene; fixed seed42 models; synthetic derivatives clustered','resamples':2000,'rng_seed':20261009,'comparisons':comparisons,'event_rate_estimates_percent':estimates}))
    print(f'E15BC summarized seed{seed}: 30 policy/alpha/calibration combinations incl diagnostics; 3910 edited cases, 111 normal sources, 66 historical events',flush=True)

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
if __name__=='__main__':
    with threadpool_limits(limits=4):
        for seed in [42,43,44]:aggregate(seed)
