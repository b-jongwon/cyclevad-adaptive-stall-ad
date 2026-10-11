"""Read-only dataset audit and reproducibility evidence."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
from cycle_vad.data import inventory, labels_for, split_normal, write_json
from cycle_vad.pipeline import environment


def audit(root, output):
    out=Path(output); out.mkdir(parents=True, exist_ok=True)
    rows=inventory(root, ['R01','R02','R03','R04'])
    public=[]; summary={}; splits={}
    for r in rows:
        p={k:v for k,v in r.items() if k not in ('directory','labels')}
        if r['partition']=='testing':
            y=np.load(r['labels'], allow_pickle=False)
            strict=labels_for(r)
            p.update(label_length=len(y), label_sha256=hashlib.sha256(Path(r['labels']).read_bytes()).hexdigest(), valid_frames=int((strict>=0).sum()), unknown_frames=int((strict<0).sum()), anomaly_frames=int((strict==1).sum()))
        public.append(p)
    for scene in ['R01','R02','R03','R04']:
        sr=[r for r in public if r['scene']==scene]
        train=[r for r in sr if r['partition']=='training']; test=[r for r in sr if r['partition']=='testing']
        summary[scene]={'train_videos':len(train),'train_frames':sum(r['frames'] for r in train),'test_videos':len(test),'test_frames':sum(r['frames'] for r in test),'valid_frames':sum(r['valid_frames'] for r in test),'unknown_frames':sum(r['unknown_frames'] for r in test),'anomaly_frames':sum(r['anomaly_frames'] for r in test)}
        splits[scene]=split_normal(train,42)
    write_json(out/'data_audit.json', {'protocol':'strict-v1','scenes':summary,'mismatches':[r for r in public if r.get('unknown_frames',0)],'sequences':public})
    write_json(out/'splits.json', splits)
    env=environment()
    import torch
    assert torch.cuda.is_available(), 'Local CUDA GPU required'
    name=torch.cuda.get_device_name(0)
    assert 'RTX PRO 6000' in name, name
    a=torch.ones((64,64),device='cuda'); b=a@a; torch.cuda.synchronize()
    assert b[0,0].item()==64
    env.update(gpu=name, capability=torch.cuda.get_device_capability(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,cuda=torch.version.cuda,gpu_smoke_test='passed')
    write_json(out/'environment.json',env)
    (out/'requirements-lock.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--output',default='results/E0'); a=p.parse_args(); audit(a.data_root,a.output)
