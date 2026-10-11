import argparse,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from cycle_vad.data import inventory
from cycle_vad.paper_diagnostics import run
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--folds',nargs='+',type=int,default=list(range(5)));p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    folds=json.loads((ROOT/'configs/experiments/followup/normal_folds.json').read_text());rows=inventory(ROOT.parent/'IPAD_dataset',a.scenes)
    with threadpool_limits(limits=4):
        for seed in a.seeds:
            for fold in a.folds:
                for sc in a.scenes:
                    dest=[ROOT/'results'/stage/f'seed{seed}'/sc/f'fold{fold}'/'run.json' for stage in ['E11B','E15A']]
                    if a.resume and all(f.exists() and json.loads(f.read_text())['status']=='complete' for f in dest):continue
                    run(ROOT,rows,sc,fold,seed,folds['scenes'][sc][fold]['splits'])
