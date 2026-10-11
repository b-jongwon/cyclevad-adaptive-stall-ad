"""Preserve completed diagnostic sections when earlier reports are regenerated."""
def diagnostics_fragment(root):
    path=root/'docs/PAPER_DIAGNOSTICS_RESULTS.md'
    if not path.exists():return ''
    prefix=online_accuracy_fragment(root)+streaming_fragment(root)+calibration_fragment(root)
    return prefix+path.read_text().replace('](../results/','](results/').replace('](../configs/','](configs/').replace('](figures/','](docs/figures/').replace('](PAPER_STAGE2_DESIGN.md)','](docs/PAPER_STAGE2_DESIGN.md)').replace('](PAPER_CALIBRATION_RESULTS.md)','](docs/PAPER_CALIBRATION_RESULTS.md)')


def calibration_fragment(root):
    path=root/'docs/PAPER_CALIBRATION_RESULTS.md'
    if not path.exists():return ''
    return path.read_text().replace('](../results/','](results/').replace('](../configs/','](configs/').replace('](figures/','](docs/figures/').replace('](PAPER_NEXT_EXPERIMENTS_20261009.md)','](docs/PAPER_NEXT_EXPERIMENTS_20261009.md)').replace('](R01_AUDIT_GUIDE.md)','](docs/R01_AUDIT_GUIDE.md)')


def streaming_fragment(root):
    path=root/'docs/STREAMING_RESULTS.md'
    if not path.exists():return ''
    return path.read_text().replace('](../results/','](results/').replace('](../configs/','](configs/').replace('](figures/','](docs/figures/')


def online_accuracy_fragment(root):
    path=root/'docs/ONLINE_ACCURACY_RESULTS.md'
    if not path.exists():return ''
    return path.read_text().replace('](../results/','](results/').replace('](../configs/','](configs/').replace('](../scripts/','](scripts/').replace('](figures/','](docs/figures/')
