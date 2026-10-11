"""Validate the planned E8B/E9S grid and audit edit feasibility, without scoring."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from cycle_vad.data import fingerprint, inventory, write_json
from cycle_vad.followup import audit_folds
ROOT=Path(__file__).resolve().parents[1]

def halfup(x):return int(math.floor(x+.5))

def prepare(data_root):
    cfg=json.loads((ROOT/'configs/experiments/followup/phase_process_v2.json').read_text())
    folds=json.loads((ROOT/'configs/experiments/followup/normal_folds.json').read_text())
    rows=inventory(data_root,cfg['scenes']);lookup={r['id']:r for r in rows}
    checks=audit_folds(folds,rows);grid=cfg['E9S'];cases=[];periods=[]
    for scene,entries in folds['scenes'].items():
        for entry in entries:
            fold=entry['fold']
            # Actual FIT recording duration, shared across strides; no test duration is used for T.
            T=float(np.median([lookup[i]['frames'] for i in entry['splits']['fit']]))
            report=json.loads((ROOT/f'results/E8/{scene}/fold{fold}/run.json').read_text())
            legacy=float(np.median([2*math.ceil(lookup[i]['frames']/2) for i in entry['splits']['fit']]))
            assert legacy==report['diagnostics']['cycle']['median_period_source_frames']
            periods.append({'scene':scene,'fold':fold,'T_source_frames':T})
            W,D=halfup(.4*T),halfup(.2*T)
            for i in entry['splits']['normal_evaluation']:
                N=lookup[i]['frames']
                for kind in grid['edits']:
                    for severity in grid['severity_period_fractions']:
                        L=max(2,halfup(severity*T))
                        for onset in grid['onset_period_fractions']:
                            o=halfup(onset*T); reasons=[]
                            count=N+(L if kind=='freeze' else -L if kind=='skip' else 0)
                            if o<grid['minimum_prefix_source_frames']:reasons.append('insufficient_prefix')
                            if kind!='freeze' and o+(2*L if kind=='swap_adjacent_blocks' else L)>N:reasons.append('edit_out_of_range')
                            if o+W>N:reasons.append('original_window_out_of_range')
                            if o+W>count:reasons.append('edited_window_out_of_range')
                            case={'id':f'{i}/f{fold}/{kind}/s{severity}/o{onset}', 'source_id':i,'scene':scene,'fold':fold,
                                  'edit':kind,'severity':severity,'onset_fraction':onset,'source_frames':N,
                                  'T':T,'onset':o,'length':L,'window':W,'deadline':D,'output_frames':count,
                                  'eligible':not reasons,'exclusion_reasons':reasons}
                            cases.append(case)
    assert len(cases)==111*4*3*3 and len({c['id'] for c in cases})==len(cases)
    summary={kind:{'eligible':sum(c['eligible'] for c in cases if c['edit']==kind),
                   'planned':sum(c['edit']==kind for c in cases)} for kind in grid['edits']}
    plan={'status':'planned_not_run','source':'Lengths and locked FIT splits only; no anomaly scores computed',
          'protocol_sha256':fingerprint(cfg),'split_sha256':fingerprint(folds),'fold_audits':len(checks),
          'periods':periods,'source_files':111,'planned_cases':len(cases),'eligible_cases':sum(c['eligible'] for c in cases),
          'by_edit':summary,'by_scene':{s:{'eligible':sum(c['eligible'] for c in cases if c['scene']==s),
             'planned':sum(c['scene']==s for c in cases)} for s in cfg['scenes']},
          'exclusion_reason_counts':dict(Counter(reason for c in cases for reason in c['exclusion_reasons'])),
          'E8B_arms':len(cfg['E8B']['arms']),'E8B_effective_conditions':9,'E9S_arms':len(grid['arms']),
          'notes':['Counts are unique edit cases; seeds/stride do not add independent samples.',
                   'Periods and eligibility must be rechecked before executing with actual dense-cache-derived features.']}
    out=ROOT/'configs/experiments/followup'
    write_json(out/'phase_process_v2_preflight.json',plan)
    (out/'e9_cases_v2.json').write_text('[\n'+',\n'.join(json.dumps(c,separators=(',',':')) for c in cases)+'\n]\n')
    print(json.dumps({k:plan[k] for k in ['status','source_files','planned_cases','eligible_cases','by_edit','by_scene']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True)
    prepare(p.parse_args().data_root)
