import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from cycle_vad.data import inventory
from cycle_vad.paper_calibration import run_b0,run_bc,read,suffix
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['E15B0','E15BC'],required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--folds',nargs='+',type=int,default=list(range(5)))
    p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    rows=inventory(ROOT.parent/'IPAD_dataset',a.scenes) if a.stage=='E15BC' else None
    with threadpool_limits(limits=4):
        for seed in a.seeds:
            for fold in a.folds:
                for sc in a.scenes:
                    path=ROOT/'results'/a.stage/suffix(seed,sc,fold)/'run.json'
                    if a.resume and path.exists() and read(path)['status']=='complete':continue
                    if a.stage=='E15B0':run_b0(ROOT,seed,sc,fold)
                    else:run_bc(ROOT,rows,seed,sc,fold)
