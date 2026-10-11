"""Execute the frozen E8B or E9S protocol with resumable scene/fold units."""
import argparse,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from cycle_vad.data import inventory
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['E8B','E9S'],required=True)
    p.add_argument('--data-root',type=Path,default=ROOT.parent/'IPAD_dataset')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);p.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04']);p.add_argument('--stride',type=int,choices=[1,2],default=2)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    rows=inventory(a.data_root,a.scenes);folds=json.loads((ROOT/'configs/experiments/followup/normal_folds.json').read_text())
    if a.stage=='E8B':from cycle_vad.phase_experiment import run
    else:from cycle_vad.process_experiment import run
    with threadpool_limits(limits=4):
        for seed in a.seeds:
            for fold in a.folds:
                for scene in a.scenes:
                    suffix=Path(f'seed{seed}') if a.stage=='E8B' else Path(f'stride{a.stride}')/f'seed{seed}'
                    done=ROOT/'results'/a.stage/suffix/scene/f'fold{fold}'/'run.json'
                    if a.resume and done.exists() and json.loads(done.read_text()).get('status')=='complete':continue
                    extra={} if a.stage=='E8B' else {'stride':a.stride}
                    run(ROOT,rows,scene,fold,seed,folds['scenes'][scene][fold]['splits'],**extra)

if __name__=='__main__':main()
