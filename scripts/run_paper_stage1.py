import argparse,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from cycle_vad.data import inventory
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['E11','E12'],required=True);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--folds',nargs='+',type=int,default=list(range(5)));p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04']);p.add_argument('--resume',action='store_true');p.add_argument('--data-root',type=Path,default=ROOT.parent/'IPAD_dataset');a=p.parse_args()
    rows=inventory(a.data_root,a.scenes);folds=json.loads((ROOT/'configs/experiments/followup/normal_folds.json').read_text())
    if a.stage=='E11':from cycle_vad.paper_experiments import run_e11 as run
    else:from cycle_vad.paper_subspaces import run_e12 as run
    with threadpool_limits(limits=4):
        for seed in a.seeds:
            for fold in a.folds:
                for scene in a.scenes:
                    done=ROOT/'results'/a.stage/f'seed{seed}'/scene/f'fold{fold}'/'run.json'
                    if a.resume and done.exists() and json.loads(done.read_text()).get('status')=='complete':continue
                    run(ROOT,rows,scene,fold,seed,folds['scenes'][scene][fold]['splits'])
if __name__=='__main__':main()
