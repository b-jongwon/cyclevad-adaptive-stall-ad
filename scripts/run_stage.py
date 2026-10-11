"""Run one registered stage. Publication is performed after reviewing artifacts."""
import argparse, copy, json, os, shutil, time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from cycle_vad.data import inventory, write_json
from cycle_vad.pipeline import run, fit_scene, environment
from cycle_vad.ablation import evaluate_variants

SCENES=['R01','R02','R03','R04']


def export_fit(run_root,out,scene):
    dest=Path(out)/scene; dest.mkdir(parents=True,exist_ok=True)
    for name in ['fit_report.json','protocol.json','metrics.json','per_sequence.csv','per_sequence.json']:
        src=Path(run_root)/scene/name
        if src.exists(): shutil.copyfile(src,dest/name)


def mark(stage,started):
    p=Path('results/status.json'); status=json.loads(p.read_text()) if p.exists() else {}
    from datetime import datetime
    from zoneinfo import ZoneInfo
    status[stage]={'status':'completed','completed_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'elapsed_seconds':time.perf_counter()-started}
    write_json(p,status)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',required=True,choices=['E1','E2','E3','E4','E5']); p.add_argument('--data-root',required=True,type=Path); p.add_argument('--threads',type=int,default=4); p.add_argument('--config',type=Path,default=Path('configs/experiments/main.json'))
    args=p.parse_args(); start=time.perf_counter(); cfg=json.loads(args.config.read_text()); main_root=Path('runs/stride2_seed42'); dest=Path('results')/args.stage
    dest.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=args.threads):
        if args.stage=='E1':
            c=copy.deepcopy(cfg); c['scenes']=['R02']
            run(args.data_root,main_root,c)
            export_fit(main_root,dest,'R02')
        elif args.stage=='E2':
            assert (Path('results/E1/R02/metrics.json')).exists()
            c=copy.deepcopy(cfg); c['scenes']=['R01','R03','R04']; run(args.data_root,main_root,c)
            rows=inventory(args.data_root,SCENES)
            for scene in SCENES:
                model=joblib.load(main_root/scene/'model.joblib')
                evaluate_variants(model,[r for r in rows if r['scene']==scene],main_root,dest/scene)
                shutil.copyfile(main_root/scene/'fit_report.json',dest/scene/'fit_report.json')
        elif args.stage=='E3':
            for scene in SCENES:
                src=Path('results/E2')/scene; assert (src/'metrics.json').exists()
                d=dest/scene; d.mkdir(parents=True,exist_ok=True)
                for name in ['metrics.json','per_sequence.json','per_sequence.csv','protocol.json','diagnostics.json']:
                    shutil.copyfile(src/name,d/name)
            write_json(dest/'reuse.json',{'source':'E2','note':'Same frozen predictions; E3 exposes the pre-registered 2x2 factorial comparison including A2. No new fitting or test-based selection.'})
        elif args.stage=='E4':
            rows=inventory(args.data_root,SCENES)
            for scene in SCENES:
                model=joblib.load(main_root/scene/'model.joblib')
                evaluate_variants(model,[r for r in rows if r['scene']==scene],main_root,dest/scene,extended=True)
        elif args.stage=='E5':
            rows=inventory(args.data_root,SCENES)
            for seed in (43,44):
                root=Path(f'runs/stride2_seed{seed}'); root.mkdir(parents=True,exist_ok=True)
                if not (root/'cache').exists(): (root/'cache').symlink_to((main_root/'cache').resolve(),target_is_directory=True)
                c=copy.deepcopy(cfg); c['model']['seed']=seed
                for scene in SCENES:
                    sr=[r for r in rows if r['scene']==scene]; out=dest/f'seed{seed}'/scene
                    if (out/'metrics.json').exists(): continue
                    model=fit_scene(sr,root,c)
                    evaluate_variants(model,sr,root,out)
                    shutil.copyfile(root/scene/'fit_report.json',out/'fit_report.json')
            root=Path('runs/stride1_seed42'); c=copy.deepcopy(cfg); c['stride']=1
            run(args.data_root,root,c,stage='extract')
            for scene in SCENES:
                sr=[r for r in rows if r['scene']==scene]; out=dest/'stride1'/scene
                if (out/'metrics.json').exists(): continue
                model=fit_scene(sr,root,c); evaluate_variants(model,sr,root,out)
                shutil.copyfile(root/scene/'fit_report.json',out/'fit_report.json')
    env=environment()
    if args.stage=='E5':
        import torch
        env.update(gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),memory_scope='PyTorch allocator peak within E5 process; excludes other processes and driver allocations')
    write_json(dest/'environment.json',env); mark(args.stage,start)
    print(f'[{args.stage} COMPLETE] {time.perf_counter()-start:.1f}s',flush=True)

if __name__=='__main__': main()
