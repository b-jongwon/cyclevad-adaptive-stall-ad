"""Attribute R01 fold-0 false alarms to the maximum calibrated score channel."""
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from cycle_vad.data import inventory, labels_for, load_cache, hold, write_json
ROOT=Path(__file__).resolve().parents[1]

def main():
    model=joblib.load(ROOT/'runs/followup/R01/fold0/train0.joblib')
    run=json.loads((ROOT/'results/E8/R01/fold0/run.json').read_text())
    rows=inventory(ROOT.parent/'IPAD_dataset',['R01']);lookup={r['id']:r for r in rows}
    groups={}
    groups['historical_test']=[r['id'] for r in rows if r['partition']=='testing']
    keys=['pooled','pooled_inside','local','conditional_gated','alignment','innovation','progress']
    result=[];max_difference=0.;replayed_alarm_disagreements=0
    for part,ids in groups.items():
        count=np.zeros(len(keys),dtype=int);normal=0;exceeds=np.zeros(len(keys),dtype=int)
        for i in ids:
            row=lookup[i];d=load_cache(ROOT/'runs/stride2_seed42/cache'/f'{i}.npz')
            raw,c,_=model.raw(d)
            scores={k:model.calibrators[k].score(v) for k,v in raw.items()}
            scores['conditional_gated']=np.maximum(scores['conditional'],scores['conditional_inside'])*c['confidence']
            matrix=np.stack([hold(d['indices'],scores[k],row['frames']) for k in keys],axis=1)
            y=labels_for(row);matrix=matrix[y==0];normal+=len(matrix)
            with np.load(ROOT/'runs/followup/R01/fold0/scores'/f'{row["partition"]}_{row["sequence"]}.npz') as saved:
                original=saved['C-00__full'][y==0]
            max_difference=max(max_difference,float(np.max(np.abs(original-matrix.max(1)))))
            np.testing.assert_allclose(original,matrix.max(1),rtol=1e-5,atol=1e-6)
            alarm=original>model.thresholds['combined']
            replayed_alarm_disagreements+=int(np.sum(alarm!=(matrix.max(1)>model.thresholds['combined'])))
            count+=np.bincount(matrix[alarm].argmax(1),minlength=len(keys))
            exceeds+=((matrix>model.thresholds['combined']) & alarm[:,None]).sum(0)
        result.append({'partition':part,'normal_frames':normal,'false_alarm_frames':int(count.sum()),
            'fpr':float(count.sum()/normal),'dominant_channel_false_alarms':dict(zip(keys,map(int,count))),
            'channel_exceedances_within_saved_false_alarms':dict(zip(keys,map(int,exceeds)))})
    write_json(ROOT/'results/E6/R01_channel_diagnosis.json',{'scene':'R01','fold':0,'cell':'C-00','readout':'full',
        'threshold':model.thresholds['combined'],'results':result,
        'interpretation':'Descriptive maximum-channel attribution, not a causal ablation; ties use listed channel order.',
        'channel_order':keys,'alarm_mask_source':'original saved full scores',
        'replayed_max_score_abs_difference':max_difference,
        'replayed_alarm_disagreements_near_threshold':replayed_alarm_disagreements,
        'numerical_note':'Recomputed floating-point channels may cross a tied threshold by roundoff. Published metrics and attribution counts use original saved score alarms.'})
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
