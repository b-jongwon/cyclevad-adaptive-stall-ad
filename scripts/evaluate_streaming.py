"""E17: complete historical replay with batch-one features and causal scores."""
import argparse, time, platform
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits
from benchmark_streaming import ROOT, read, encode, check_sequence
from cycle_vad.features import VisualEncoder
from cycle_vad.streaming import StreamingDetector
from cycle_vad.data import inventory, frame_paths, labels_for, write_json
from cycle_vad.paper_diagnostics import sha
from cycle_vad.metrics import frame_metrics
OUT=ROOT/'results/E17'; LOCAL=ROOT/'runs/E17'
ARMS=['A','C','P','CP','Full']; COLS=[0,4,5,6,3]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rows=[r for r in inventory(ROOT.parent/'IPAD_dataset',['R01','R02','R03','R04']) if r['partition']=='testing']
    config={'seed':42,'fold':0,'sources':[r['id'] for r in rows],'source_frames':sum(r['frames'] for r in rows),'strides':[1,2],'arms':ARMS,'batch_size':1,'encoder':read(ROOT/'configs/experiments/replay_pinned.json')['encoder'],'frozen_checkpoints':'E11 seed42 fold0','thresholds':'E11B normal window q99, no refitting','labels':'strict-v1, attached only after whole video inference','stride2_odd_frames':'previous prediction held; evaluate all source frames','execution':'One current image encoding shared by two independent causal detector states; not a speed benchmark','unknown_frames':'excluded only from metrics; still processed','cadence1_scope':'Frozen stride2 model with dt=1 and source-frame lag adaptation; calibrators unchanged','test_selection':'All historical testing videos; previously inspected development benchmark, not independent confirmation'}
    cfgpath=ROOT/'configs/experiments/followup/online_accuracy_v1.json'
    if cfgpath.exists():assert read(cfgpath)==config
    else:write_json(cfgpath,config)
    assert torch.cuda.is_available()
    with threadpool_limits(limits=4):
        encoder=VisualEncoder(config['encoder']);models={sc:joblib.load(ROOT/f'runs/E11/seed42/{sc}/fold0/model.joblib') for sc in ['R01','R02','R03','R04']}
        write_json(OUT/'environment.json',{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),'encoder_identity':encoder.identity,'runner_sha256':sha(Path(__file__)),'adapter_sha256':sha(ROOT/'src/cycle_vad/streaming.py'),'checkpoint_sha256':{sc:sha(ROOT/f'runs/E11/seed42/{sc}/fold0/model.joblib') for sc in models},'execution':config['execution']})
        for number,row in enumerate(rows,1):
            tag=row['id'].replace('/','_');out=OUT/'videos'/f'{tag}.json';local=LOCAL/f'{tag}.npz'
            if args.resume and out.exists() and local.exists():continue
            saved=read(ROOT/f'results/E11B/seed42/{row["scene"]}/fold0/run.json');thresholds=np.asarray(saved['window_thresholds'][3])[COLS]
            detectors={s:StreamingDetector(models[row['scene']],thresholds[-1],s,'Full') for s in [1,2]}
            records={s:[] for s in [1,2]};scores={s:[] for s in [1,2]};began=time.perf_counter()
            for index,path in enumerate(frame_paths(row['directory'])):
                with Image.open(path) as f:im=f.convert('RGB')
                g,p,_=encode(encoder,im);im.close()
                for stride,detector in detectors.items():
                    result=detector.step(g,p) if index%stride==0 else detector.step()
                    scores[stride].append([result['A'],result['C'],result['P'],max(result['C'],result['P']),result['score']])
                    if result['fresh']:records[stride].append(result)
            elapsed=time.perf_counter()-began
            # No labels or cached features are available to the predictor above.
            labels=labels_for(row);arrays={s:np.asarray(scores[s]) for s in [1,2]}
            parity={str(s):check_sequence(records[s],models[row['scene']],s) for s in [1,2]}
            np.testing.assert_array_equal(arrays[2][1::2],arrays[2][::2][:len(arrays[2][1::2])])
            baseline=ROOT/f'runs/E11B/seed42/{row["scene"]}/fold0/testing_{row["sequence"]}.npz'
            with np.load(baseline) as original:
                np.testing.assert_array_equal(labels,original['labels']);cached=original['scores'][:,COLS]
            LOCAL.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(local,labels=labels,stride1=arrays[1],stride2=arrays[2],cached=cached,thresholds=thresholds)
            result={'source_id':row['id'],'scene':row['scene'],'frames':len(labels),'valid_frames':int((labels>=0).sum()),'unknown_frames':int((labels<0).sum()),'joint_replay_seconds':elapsed,'parity':parity,'scores_sha256':sha(local),'metrics':{key:{arm:frame_metrics(labels,arr[:,j],thresholds[j]) for j,arm in enumerate(ARMS)} for key,arr in [('stride1',arrays[1]),('stride2',arrays[2]),('cached',cached)]}}
            write_json(out,result)
            print(f'{number}/{len(rows)} {row["id"]}: {len(labels)} frames, {elapsed:.1f}s, causal parity passed',flush=True)

if __name__=='__main__':main()
