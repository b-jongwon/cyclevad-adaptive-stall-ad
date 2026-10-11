"""Registered score ablations with per-variant normal-only thresholds."""
from __future__ import annotations
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .data import balanced_sample, fingerprint, hold, labels_for, load_cache, write_json
from .metrics import frame_metrics, event_metrics
from .model import ResidualSpace, TailCalibrator, descriptor
from .pipeline import cache_path, code_fingerprint, validate_model_cache

CORE = ['pooled', 'appearance', 'cycle_conditioned', 'appearance_process', 'full']
REMOVALS = ['no_alignment','no_innovation','no_progress','single_lag','no_confidence','no_local','no_pooled','outside_only','discrete_phase']


def compose(q, confidence):
    pooled=np.maximum(q['pooled'],q['pooled_inside'])
    appearance=np.maximum(pooled,q['local'])
    conditional=np.maximum(q['conditional'],q['conditional_inside'])
    conditioned=conditional*confidence
    process=np.maximum.reduce([q[k] for k in ('alignment','innovation','progress')])
    cc=np.maximum(appearance,conditioned)
    full=np.maximum(cc,process)
    scores={'pooled':pooled,'appearance':appearance,'cycle_conditioned':cc,
            'appearance_process':np.maximum(appearance,process),'full':full,
            'no_confidence':np.maximum.reduce([appearance,conditional,process]),
            'no_local':np.maximum.reduce([pooled,conditioned,process]),
            'no_pooled':np.maximum.reduce([q['local'],conditioned,process]),
            'outside_only':np.maximum.reduce([q['pooled'],q['local'],q['conditional']*confidence,process])}
    for removed in ('alignment','innovation','progress'):
        scores['no_'+removed]=np.maximum(cc,np.maximum.reduce([q[k] for k in ('alignment','innovation','progress') if k!=removed]))
    if 'single_progress' in q:
        scores['single_lag']=np.maximum.reduce([cc,q['alignment'],q['innovation'],q['single_progress']])
    if 'discrete' in q:
        discrete=np.maximum(q['discrete'],q['discrete_inside'])*confidence
        scores['discrete_phase']=np.maximum.reduce([appearance,discrete,process])
    return scores


class DiscretePhase:
    """Four hard bins, total rank cap equal to the shared conditional rank cap."""
    def fit(self, model, data):
        bins=[[] for _ in range(4)]
        for d in data:
            z=descriptor(d); angle=model.tracker.predict(z,d['indices'])['angle']
            ids=np.minimum((angle*4).astype(int),3)
            for k in range(4):
                if np.any(ids==k): bins[k].append(z[ids==k])
        seed=model.config.get('seed',42); cap=model.config.get('subspace_rank',64)//4
        self.spaces=[]
        for group in bins:
            if not group: raise ValueError('Empty normal phase bin')
            x=balanced_sample(group,max(4,model.config.get('fit_samples',4096)//4),seed)
            self.spaces.append(ResidualSpace().fit(x,rank=cap,seed=seed))
        self.diagnostics={'phase_bins':4,'rank_cap_per_bin':cap,'actual_ranks':[len(s.basis) for s in self.spaces],
            'storage_bytes':sum(s.mean.nbytes+s.basis.nbytes+s.eigen.nbytes for s in self.spaces),
            'continuous_storage_bytes':model.coef.nbytes+model.conditional.mean.nbytes+model.conditional.basis.nbytes+model.conditional.eigen.nbytes,
            'bins':[s.diagnostics for s in self.spaces]}
        return self

    def score(self,z,angle):
        ids=np.minimum((angle*4).astype(int),3); a=np.empty(len(z)); b=np.empty(len(z))
        for k,s in enumerate(self.spaces):
            mask=ids==k
            if mask.any(): a[mask],b[mask],_=s.score(z[mask])
        return a,b


def evaluate_variants(model, rows, run_root, destination, extended=False):
    started=time.perf_counter(); out=Path(destination); out.mkdir(parents=True,exist_ok=True)
    splits=model.diagnostics['splits']; by_id={r['id']:r for r in rows}
    selected=CORE+REMOVALS if extended else CORE
    load=lambda rid: load_cache(cache_path(run_root,by_id[rid]))
    discrete=None
    if extended:
        discrete=DiscretePhase().fit(model,[load(rid) for rid in splits['fit']])
        write_json(out/'discrete_model.json',discrete.diagnostics)
    raw_dir=Path(run_root)/model.scene/'ablation_raw'; raw_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_digest=hashlib.sha256((Path(run_root)/model.scene/'model.joblib').read_bytes()).hexdigest()
    memo={}
    for part in ('reference','threshold','testing'):
        ids=splits[part] if part!='testing' else [r['id'] for r in rows if r['partition']=='testing']
        for rid in ids:
            d=load(rid); validate_model_cache(model,d)
            provenance=fingerprint({'checkpoint':checkpoint_digest,'feature_signature':str(d['signature']),'raw_schema':1})
            target=raw_dir/(rid.replace('/','_')+'.npz')
            record=None
            if target.exists():
                with np.load(target,allow_pickle=False) as a:
                    if 'provenance' in a.files and str(a['provenance'])==provenance:
                        record={k:a[k] for k in a.files}
            if record is None:
                raw,cycle,_=model.raw(d)
                record={'provenance':np.array(provenance),**raw,'angle':cycle['angle'],'confidence':cycle['confidence'],'indices':d['indices'],'frame_count':d['frame_count']}
            if extended:
                z=descriptor(d)
                record['single_progress']=model.tracker.predict(z,d['indices'],progress_lags=(1,))['progress']
                record['discrete'],record['discrete_inside']=discrete.score(z,record['angle'])
            with target.with_suffix('.tmp').open('wb') as f: np.savez_compressed(f,**record)
            target.with_suffix('.tmp').replace(target)
            memo[rid]=record
    calibrators=dict(model.calibrators)
    if extended:
        for k in ('single_progress','discrete','discrete_inside'):
            calibrators[k]=TailCalibrator().fit(np.concatenate([memo[rid][k] for rid in splits['reference']]))
    streams={rid:compose({k:c.score(record[k]) for k,c in calibrators.items()},record['confidence']) for rid,record in memo.items()}
    thresholds={}
    for variant in selected:
        values=np.concatenate([hold(memo[rid]['indices'],streams[rid][variant],by_id[rid]['frames']) for rid in splits['threshold']])
        thresholds[variant]=float(np.quantile(values,model.config.get('threshold_quantile',.99),method='higher'))
    # A fixed test population for every variant; labels never affect calibration.
    reports=[]; labels=[]; all_scores={v:[] for v in selected}; diagnostics=[]
    for r in rows:
        if r['partition']!='testing': continue
        rid=r['id']; record=memo[rid]; y=labels_for(r); labels.append(y)
        for v in selected:
            score=hold(record['indices'],streams[rid][v],r['frames']); all_scores[v].append(score)
            reports.append({'id':rid,'variant':v,**frame_metrics(y,score,thresholds[v]),**event_metrics(y,score,thresholds[v])})
        diagnostics.append({'id':rid,'mean_confidence':float(record['confidence'].mean()),'conditional_lifts_appearance_fraction':float(np.mean(streams[rid]['cycle_conditioned']>streams[rid]['appearance'])), 'process_lifts_conditioned_fraction':float(np.mean(streams[rid]['full']>streams[rid]['cycle_conditioned']))})
    aggregate={}
    for v in selected:
        rs=[r for r in reports if r['variant']==v]; ne=sum(r['events'] for r in rs); nd=sum(r['detected_events'] for r in rs)
        delays=[d for r in rs for d in r['detected_event_delays_source_frames']]
        aggregate[v]={**frame_metrics(np.concatenate(labels),np.concatenate(all_scores[v]),thresholds[v]),'events':ne,'detected_events':nd,'missed_events':ne-nd,'event_coverage':nd/ne if ne else None,'median_detected_event_delay_frames':float(np.median(delays)) if delays else None}
    write_json(out/'metrics.json',aggregate); write_json(out/'per_sequence.json',reports)
    write_json(out/'diagnostics.json',diagnostics)
    write_json(out/'protocol.json',{'model_seed':model.config['seed'],'split_seed':42,'stride':model.stride,'scene':model.scene,'splits':splits,'thresholds':thresholds,'variants':selected,'code_fingerprint':code_fingerprint(),'encoder_identity':json.loads(model.encoder_identity),'evaluation_seconds':time.perf_counter()-started,'timing_scope':'shared computation for all variants, not standalone inference latency','label_policy':'strict-v1'})
    with (out/'per_sequence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(reports[0])); w.writeheader(); w.writerows(reports)
    # Deterministic qualitative example; do not select by model performance.
    for r in rows:
        if r['partition']!='testing': continue
        y=labels_for(r)
        if not np.any(y==1): continue
        rid=r['id']; rec=memo[rid]
        write_json(out/'timeline.json',{'id':rid,'indices':rec['indices'].tolist(),'labels':y[rec['indices']].tolist(),'angle':rec['angle'].tolist(),'confidence':rec['confidence'].tolist(),'thresholds':thresholds,'scores':{v:streams[rid][v].tolist() for v in CORE}})
        break
    print(f'[ablations] {model.scene}: '+', '.join(f'{v}={aggregate[v]["auroc"]}' for v in selected),flush=True)
    return aggregate
