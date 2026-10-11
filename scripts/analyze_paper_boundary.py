"""E14A: source-balanced initial/interior comparisons from frozen E9S artifacts."""
import json
from pathlib import Path
import numpy as np
from cycle_vad.data import write_json
from cycle_vad.phase_experiment import compact
ROOT=Path(__file__).resolve().parents[1];SCENES=['R01','R02','R03','R04'];EDITS=['freeze','reverse','skip','swap_adjacent_blocks']
def read(p):return json.loads(p.read_text())
def main():
    specs={c['id']:c for c in read(ROOT/'configs/experiments/followup/e9_cases_v2.json')};rows=[];summaries=[]
    for stride,seed in [(2,42),(2,43),(2,44),(1,42)]:
        base=ROOT/'results/E9S'/f'stride{stride}'/f'seed{seed}';local=[]
        for p in sorted(base.glob('R*/fold*/run.json')):
            r=read(p);thr=np.array(r['window_thresholds'])[3];bank={}
            for c in read(p.with_name('cases.json')):
                spec=specs[c['id']];i=c['source_id'];o,L=spec['onset'],spec['length'];W=spec['window']
                if i not in bank:
                    f=ROOT/'runs/E9S'/f'stride{stride}'/f'seed{seed}'/r['scene']/f'fold{r["fold"]}'/f'{i.split("/")[-1]}_normal.npz'
                    with np.load(f) as d:bank[i]=d['scores']
                original=bank[i];ilo=o+32;ihi=min(o+(2*L if c['edit']=='swap_adjacent_blocks' else L),o+W)
                n=ihi-ilo if c['edit']!='skip' and ihi>ilo else 0;assert n==c['interior_frames']
                rec={'id':c['id'],'source_id':i,'scene':r['scene'],'fold':r['fold'],'seed':seed,'stride':stride,'edit':c['edit'],'severity':c['severity'],
                    'initial_frames':min(32,W),'interior_frames':n,'initial_edited_hits':(np.array(c['initial_transient_max'])>thr).astype(int).tolist(),
                    'initial_original_hits':(original[o:min(o+32,o+W)].max(0)>thr).astype(int).tolist(),
                    'interior_edited_hits':c['interior_alarm_primary'],'interior_original_hits':(original[ilo:ihi].max(0)>thr).tolist() if n else None}
                rows.append(rec);local.append(rec)
        groups=[]
        for sc in SCENES:
            for ed in EDITS:
                for sev in [.05,.1,.2]:
                    cs=[c for c in local if c['scene']==sc and c['edit']==ed and c['severity']==sev];g={'scene':sc,'edit':ed,'severity':sev,'cases':len(cs)}
                    for segment in ['initial','interior']:
                        eligible=[c for c in cs if c[f'{segment}_frames']>0];ids=sorted(set(c['source_id'] for c in eligible));g[f'{segment}_cases']=len(eligible);g[f'{segment}_sources']=len(ids)
                        for mode in ['edited','original']:
                            g[f'{segment}_{mode}_rate']=np.mean([np.mean([c[f'{segment}_{mode}_hits'] for c in eligible if c['source_id']==i],axis=0) for i in ids],axis=0).tolist() if ids else None
                    groups.append(g)
        bytype={}
        for ed in EDITS:
            gs=[g for g in groups if g['edit']==ed];v={}
            for segment in ['initial','interior']:
                # Equal severity within scene, then equal scenes with coverage.
                available=[sc for sc in SCENES if any(g['scene']==sc and g[f'{segment}_sources']>0 for g in gs)]
                v[f'{segment}_eligible_cases']=sum(g[f'{segment}_cases'] for g in gs);v[f'{segment}_scenes']=available
                for mode in ['edited','original']:
                    v[f'{segment}_{mode}_rate']=np.mean([np.mean([g[f'{segment}_{mode}_rate'] for g in gs if g['scene']==sc and g[f'{segment}_sources']>0],axis=0) for sc in available],axis=0).tolist() if available else None
            bytype[ed]=v
        summaries.append({'stride':stride,'seed':seed,'cases':len(local),'by_type':bytype,'groups':groups})
    dest=ROOT/'results/E14'
    for stride,seed in [(2,42),(2,43),(2,44),(1,42)]:
        compact(dest/f'stride{stride}'/f'seed{seed}'/'cases.json',[r for r in rows if r['stride']==stride and r['seed']==seed])
    write_json(dest/'summary.json',summaries)
    write_json(dest/'run.json',{'status':'boundary_analysis_complete','scope':'E14A existing boundary/interior results only; new order-preserving jump controls and post-window recovery remain pending',
        'cases':len(rows),'models_refit':False,'warning':'Interior is a different conditional subset/time window. Filter state still carries boundary history; this is not proof of independence from edit discontinuities. Skip has no interior.'})
    print(f'PASS E14A: {len(rows)} case analyses; original windows read from saved normal scores; interior coverage matched manifest.')
if __name__=='__main__':main()
