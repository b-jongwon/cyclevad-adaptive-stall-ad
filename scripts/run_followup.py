"""Run preregistered E6/E8 folds using local RTX PRO 6000 feature caches."""
import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from cycle_vad.data import inventory, write_json
from cycle_vad.followup import audit_folds, run_fold

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--scenes', nargs='+', default=['R01', 'R02', 'R03', 'R04'])
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42, choices=[42,43,44])
    args = p.parse_args()
    config = json.loads((ROOT/'configs/experiments/replay_pinned.json').read_text())
    config['model']['seed'] = args.seed
    folds = json.loads((ROOT/'configs/experiments/followup/normal_folds.json').read_text())
    rows = inventory(args.data_root, config['scenes'])
    write_json(ROOT/'results/E6/fold_audit.json', {'checks': audit_folds(folds, rows),
        'normal_files': sum(r['partition'] == 'training' for r in rows),
        'recording_group_independence_verified': False})
    with threadpool_limits(limits=args.cpu_threads):
        for fold in args.folds:
            for scene in args.scenes:
                if args.seed != 42:
                    # Seed robustness freezes seed-42 normal-validation choices;
                    # projection/cache and five-way file assignments also stay fixed.
                    primary = json.loads((ROOT/f'results/E8/{scene}/fold{fold}/run.json').read_text())
                    config['model']['harmonics_candidates'] = [primary['diagnostics']['selected_harmonics']]
                    config['model']['ridge_candidates'] = [primary['diagnostics']['selected_ridge']]
                suffix = Path('.') if args.seed == 42 else Path(f'seed{args.seed}')
                run_fold(scene, folds['scenes'][scene][fold], rows, config,
                         ROOT/'runs/stride2_seed42', ROOT/'runs/followup'/suffix, ROOT/'results/E8'/suffix)

if __name__ == '__main__':
    main()
