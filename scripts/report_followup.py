"""Evidence-driven follow-up README section and static research plots."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from cycle_vad.data import write_json
ROOT = Path(__file__).resolve().parents[1]
SCENES = ['R01', 'R02', 'R03', 'R04']
FIG = ROOT/'docs/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'cyclevad-followup-v1','axes.spines.top':False,'axes.spines.right':False})
def read(p): return json.loads((ROOT/p).read_text())
def table(h, rows): return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def fmt(x): return '—' if x is None else f'{100*x:.2f}'
def save(fig, name):
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});
    svg=FIG/f'{name}.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n');fig.savefig(FIG/f'{name}.png',bbox_inches='tight',dpi=150);plt.close(fig)
def scorefile(scene, fold, identity):
    _, part, seq = identity.split('/')
    return ROOT/'runs/followup'/scene/f'fold{fold}'/'scores'/f'{part}_{seq}.npz'

def report_e6():
    records=[];quant=[]
    fig, axes=plt.subplots(2,2,figsize=(12,8))
    parts=['reference','threshold','normal_evaluation','historical_test']
    colors=['#6b7280','#d49b38','#3676b5','#c75a56']
    box, bax=plt.subplots(1,4,figsize=(14,4),sharey=True)
    for scene,ax,bx in zip(SCENES,axes.flat,bax):
        ms=read(f'results/E8/{scene}/fold0/metrics.json'); pv=read(f'results/E8/{scene}/fold0/per_video.json')
        for part,color in zip(parts,colors):
            m=next(x for x in ms if x['partition']==part and x['cell']=='C-00' and x['readout']=='full')
            records.append(m)
            videos=[x for x in pv if x['partition']==part and x['cell']=='C-00' and x['readout']=='full']
            values=[]
            for v in videos:
                with np.load(scorefile(scene,0,v['id'])) as d:
                    values.append(d['C-00__full'][d['labels']==0])
            x=np.sort(np.concatenate(values));ids=np.unique(np.linspace(0,len(x)-1,min(1200,len(x))).astype(int))
            ax.plot(x[ids],(ids+1)/len(x),label=part,color=color,lw=1.8)
            quant.append({'scene':scene,'partition':part,'normal_frames':len(x),'p50_p90_p99':np.quantile(x,[.5,.9,.99]).tolist()})
        ax.axvline(m['threshold'],ls='--',c='black',lw=1,label='normal threshold q99')
        ax.set(title=scene,xlabel='C-00 full score (symlog)',ylabel='Normal-frame ECDF');ax.set_xscale('symlog',linthresh=1);ax.set_xlim(left=0);ax.grid(alpha=.15)
        vals=[[x['fpr'] for x in pv if x['partition']==part and x['cell']=='C-00' and x['readout']=='full' and x['fpr'] is not None] for part in parts]
        bx.boxplot(vals,tick_labels=['Ref','Thr','Eval','Hist'],showfliers=True)
        bx.set(title=scene,ylim=(-.02,1.02));bx.yaxis.set_major_formatter(PercentFormatter(1));bx.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('E6 / Fold 0: calibration transfer on normal frames');fig.tight_layout();save(fig,'E6_normal_ecdf')
    bax[0].set_ylabel('Per-video normal false-positive rate');box.suptitle('E6 / Fold 0: file-level FPR distribution');box.tight_layout();save(box,'E6_video_fpr')
    write_json(ROOT/'results/E6/calibration_summary.json', {'metrics':records,'normal_score_quantiles':quant,
        'status':'complete','diagnostic_scope':'fold0, C-00 full; historical test normal frames are diagnostic only',
        'no_threshold_retuning':True,'annotations_completed':0})
    rows=[]
    for scene in SCENES:
        selected=[next(x for x in records if x['scene']==scene and x['partition']==part) for part in parts]
        rows.append([scene]+[fmt(m['fpr']) for m in selected]+[fmt(selected[2]['video_mean_fpr'])])
    diagnosis=''
    dp=ROOT/'results/E6/R01_channel_diagnosis.json'
    if dp.exists():
        d=json.loads(dp.read_text())['results'][0];counts=d['dominant_channel_false_alarms']
        fraction=(counts['pooled']+counts['pooled_inside'])/d['false_alarm_frames']
        diagnosis=f'R01 historical 정상 오경보 {d["false_alarm_frames"]}프레임 중 **{100*fraction:.2f}%**에서 pooled 외형 분기가 최댓값이었습니다. Confidence로 가중한 조건부 분기가 최댓값인 오경보는 {counts["conditional_gated"]}개였습니다. 이는 점수 귀속 진단이며 촬영 환경 등 원인을 확정하는 인과 분석은 아닙니다. [분기별 진단](results/E6/R01_channel_diagnosis.json)\n\n'
    return ('### E6 — 정상 분할·보정 이동 진단 완료\n\n'
        '정상 111개 영상의 5개 분할(장면×fold 20개)에서 fit/validation/reference/threshold/evaluation 중복이 없고 각 영상이 정확히 한 번 평가됨을 확인했습니다. 파일 간 원본 recording 독립성은 아직 미확인입니다. 40개 영상의 모델 예측을 숨긴 로컬 프레임 뷰어와 접촉시트를 준비했습니다. **실제 cycle 주석은 0개이며 위치 정확도는 미측정**입니다.\n\n'
        +table(['장면','Reference FPR %','Threshold FPR %','Normal eval FPR %','Historical normal FPR %','Eval 영상 평균 FPR %'],rows)
        +'표는 fold 0의 C-00 Full, 각 분할의 정상 프레임 기준입니다. Reference/threshold는 보정에 사용했으므로 일반화 성능이 아닙니다. Historical은 기존 테스트의 정상 라벨 구간입니다. 정상 q99라도 새로운 정상 영상에서 1% FPR을 보장하지 않습니다. 테스트 정상 구간에는 앞선 이상으로 인한 추적 상태 영향도 있을 수 있어, 이 차이를 촬영 환경 변화만의 원인으로 확정하지 않습니다.\n\n'
        +'![정상 점수 분포](docs/figures/E6_normal_ecdf.svg)\n\n![영상별 오경보](docs/figures/E6_video_fpr.svg)\n\n'
        +diagnosis+'[E6 수치](results/E6/calibration_summary.json) · [분할 검사](results/E6/fold_audit.json) · [주석 자료 상태](results/E6/annotation_packet.json)\n\n')

def report_e8(complete):
    all_metrics=[];all_pv=[];runs=[]
    for p in sorted((ROOT/'results/E8').glob('R*/fold*/run.json')):
        runs.append(json.loads(p.read_text()))
        all_metrics.extend(json.loads((p.parent/'metrics.json').read_text()))
        all_pv.extend(json.loads((p.parent/'per_video.json').read_text()))
    primary='appearance_conditional'
    cells=[f'C-{r}{g}' for r in range(3) for g in range(3)]
    hist=[m for m in all_metrics if m['partition']=='historical_test' and m['readout']==primary]
    normal=[m for m in all_pv if m['partition']=='normal_evaluation' and m['readout']==primary]
    summary=[]
    for cell in cells:
        h=[m for m in hist if m['cell']==cell]
        fpr={s:sum(m['fpr']*m['valid_frames'] for m in normal if m['cell']==cell and m['scene']==s)/sum(m['valid_frames'] for m in normal if m['cell']==cell and m['scene']==s) for s in SCENES}
        summary.append({'cell':cell,'historical_macro_ap':float(np.mean([m['ap'] for m in h])),
            'historical_macro_auroc':float(np.mean([m['auroc'] for m in h])), 'normal_oof_fpr':fpr,
            'normal_macro_fpr':float(np.mean(list(fpr.values())))})
    base=summary[0];cand=next(x for x in summary if x['cell']=='C-21')
    decision={'candidate':'C-21','baseline':'C-00','primary_readout':primary,
        'historical_macro_ap_gain_pp':100*(cand['historical_macro_ap']-base['historical_macro_ap']),
        'normal_fpr_increase_pp':{s:100*(cand['normal_oof_fpr'][s]-base['normal_oof_fpr'][s]) for s in SCENES}}
    decision['point_target_met']=decision['historical_macro_ap_gain_pp']>=1 and all(v<=1 for v in decision['normal_fpr_increase_pp'].values())
    write_json(ROOT/'results/E8/summary.json', {'status':'complete_seed42' if complete else 'running',
        'completed_scene_folds':len(runs),'statistical_models':len(runs)*3,'seed':42,
        'rows':summary,'decision':decision,'historical_test_diagnostic_only':True,
        'scope':'primary_seed42','robustness_seeds_43_44':('complete' if all(len(list((ROOT/f'results/E8/seed{s}').glob('R*/fold*/run.json')))==20 for s in (43,44)) else 'pending'),'tracking_accuracy':None,'wall_seconds':sum(x['wall_seconds'] for x in runs)})
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    for ax,key,title in zip(axes,['historical_macro_ap','normal_macro_fpr'],['Historical test: macro AP (%)','Normal evaluation: macro FPR (%)']):
        a=np.array([x[key]*100 for x in summary]).reshape(3,3)
        im=ax.imshow(a,cmap='Blues' if key.endswith('ap') else 'Oranges')
        for r in range(3):
            for g in range(3):ax.text(g,r,f'{a[r,g]:.2f}',ha='center',va='center',color='white' if a[r,g] > a.min()+.55*np.ptp(a) else 'black')
        ax.set_xticks(range(3),['r × agreement','r only','No gate']);ax.set_yticks(range(3),['Legacy weights','r weights','Uniform weights'])
        ax.set(title=title,xlabel='Inference gate',ylabel='Mean-fitting weights');fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('E8 / Primary readout: max(appearance, conditional)');fig.tight_layout();save(fig,'E8_confidence_factorial')
    rows=[[x['cell'],fmt(x['historical_macro_auroc']),fmt(x['historical_macro_ap'])]+[fmt(x['normal_oof_fpr'][s]) for s in SCENES] for x in summary]
    text=('### E8 — Confidence 3×3 실험 '+('seed 42 완료' if complete else '진행 중')+'\n\n'
        +f'{len(runs)}개 장면×fold, 통계 모델 {len(runs)*3}개를 학습하고 각 모델의 추론 게이트 3종과 readout 3종을 평가했습니다. 각 fold의 추적기·θ·pooled/local 분기와 C-00에서 선택한 Fourier 차수·ridge를 고정했습니다. C-00 재학습 및 원래 점수와의 동등성 검사를 모두 통과했습니다.\n\n'
        +'C-행열: 학습 행 0=legacy, 1=위치 신뢰도 r, 2=균등; 추론 열 0=legacy, 1=r, 2=gate 없음. 아래 주지표는 max(appearance, conditional)입니다. 모든 조합과 readout은 각각 정상 q99 임계값을 사용합니다.\n\n'
        +table(['조합','Historical macro AUROC %','Historical macro AP %']+[f'{s} 정상 OOF FPR %' for s in SCENES],rows)
        +'![Confidence factorial](docs/figures/E8_confidence_factorial.svg)\n\n'
        +f'사전에 고정한 C-21의 C-00 대비 historical macro AP 변화는 **{decision["historical_macro_ap_gain_pp"]:+.3f} pp**입니다. AP +1 pp 및 장면별 정상 FPR 증가 ≤1 pp라는 점추정 기준은 **'+('충족' if decision['point_target_met'] else '미충족')+'**입니다. 기존 테스트는 탐색적 진단이며 확증 자료가 아닙니다.\n\n'
        +'[전체 요약](results/E8/summary.json) · 각 장면/fold의 metrics.json, per_video.json, mechanism.json에는 27개 readout과 gate 활성 비율·점수 분포가 포함됩니다. seed 43/44 재검증은 아직 실행하지 않았습니다.\n\n')
    mechanisms=[]
    for scene in SCENES:
        mechanisms+=read(f'results/E8/{scene}/fold0/mechanism.json')
    activation=[]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,target in zip(axs,['normal','anomaly']):
        for j,cell in enumerate(['C-00','C-01','C-21','C-22']):
            values=[]
            for scene in SCENES:
                selected=[m for m in mechanisms if m['scene']==scene and m['partition']=='historical_test' and m['cell']==cell and m['target']==target]
                total=sum(m['frames'] for m in selected)
                value=sum(m['frames']*m['activation_fraction'] for m in selected)/total
                values.append(value)
                activation.append({'scene':scene,'cell':cell,'target':target,'frames':total,'activation_fraction':value,
                    'mean_r':sum(m['frames']*m['mean_r'] for m in selected)/total,
                    'mean_confidence':sum(m['frames']*m['mean_legacy_confidence'] for m in selected)/total})
            ax.bar(np.arange(4)+(j-1.5)*.18,values,.18,label=cell)
        ax.set_xticks(range(4),SCENES);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title('Historical '+target+' frames');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axs[0].set_ylabel('Fraction: gated conditional > appearance');axs[1].legend(frameon=False,ncol=2)
    fig.suptitle('E8 / Does the conditional channel contribute to the maximum?');fig.tight_layout();save(fig,'E8_gate_activation')
    write_json(ROOT/'results/E8/gate_activation.json',activation)
    text+='![조건부 분기 활성 비율](docs/figures/E8_gate_activation.svg)\n\nR01은 일치도 항을 제거해도 r 자체가 낮고 조건부 점수가 외형 점수를 넘는 비율이 0%였습니다(C-00/C-01/C-21, historical 이상 프레임). 낮은 r이 실제 위치 오차인지 반복 외형의 모호성인지 구분하려면 E7 주석이 필요합니다.\n\n'
    readout_rows=[]
    for readout in ('conditional_only','appearance_conditional','full'):
        for cell in ('C-00','C-21','C-22'):
            selected=[m for m in all_metrics if m['partition']=='historical_test' and m['readout']==readout and m['cell']==cell]
            readout_rows.append([readout,cell,fmt(float(np.mean([m['auroc'] for m in selected]))),fmt(float(np.mean([m['ap'] for m in selected])))])
    text+=table(['Readout','조합','Historical macro AUROC %','Historical macro AP %'],readout_rows)
    control_ci=ROOT/'results/E8/paired_bootstrap_C22.json'
    if complete and control_ci.exists():
        cc=json.loads(control_ci.read_text());control=next(x for x in summary if x['cell']=='C-22')
        gain=100*(control['historical_macro_ap']-base['historical_macro_ap'])
        text+=f'사전 대조군 C-22(균등 학습·gate 없음)의 주 readout AP 변화는 **{gain:+.3f} pp**, 파일 bootstrap 95% 구간은 [{cc["macro_ap_gain_pp_ci95"][0]:+.3f}, {cc["macro_ap_gain_pp_ci95"][1]:+.3f}] pp입니다. C-21을 사후 교체하지 않았으며, gate 제거를 별도 가설로 검토할 근거입니다. 두 구간은 다중 비교 보정 없는 탐색적 구간입니다.\n\n'
    ci_path=ROOT/'results/E8/paired_bootstrap.json'
    if complete and ci_path.exists():
        ci=json.loads(ci_path.read_text())
        text+=f'C-21−C-00 historical macro AP의 파일 단위 paired bootstrap 95% 구간: **[{ci["macro_ap_gain_pp_ci95"][0]:+.3f}, {ci["macro_ap_gain_pp_ci95"][1]:+.3f}] pp** (2,000회). 영상 원본 그룹 독립성은 미검증입니다. [불확실성 수치](results/E8/paired_bootstrap.json)\n\n'
    return text

def report_robustness():
    output=[]
    for seed in (42,43,44):
        root=ROOT/'results/E8'/('' if seed==42 else f'seed{seed}')
        if len(list(root.glob('R*/fold*/run.json')))!=20:
            return ''
        metrics=[];videos=[]
        for p in root.glob('R*/fold*/metrics.json'):metrics+=json.loads(p.read_text())
        for p in root.glob('R*/fold*/per_video.json'):videos+=json.loads(p.read_text())
        for cell in ('C-00','C-21','C-22'):
            h=[m for m in metrics if m['partition']=='historical_test' and m['cell']==cell and m['readout']=='appearance_conditional']
            n=[m for m in videos if m['partition']=='normal_evaluation' and m['cell']==cell and m['readout']=='appearance_conditional']
            output.append({'seed':seed,'cell':cell,'historical_macro_ap':float(np.mean([x['ap'] for x in h])),
                'normal_oof_fpr':{scene:sum(x['valid_frames']*x['fpr'] for x in n if x['scene']==scene)/sum(x['valid_frames'] for x in n if x['scene']==scene) for scene in SCENES}})
    write_json(ROOT/'results/E8/robustness.json',{'status':'complete','seeds':[42,43,44],
        'scene_folds':60,'statistical_models':180,'readouts_per_scene_fold':27,'rows':output,
        'fourier_ridge_frozen_from_seed42':True,'features_and_splits_fixed':True,
        'scope':'Seed sensitivity on the same data; not three independent datasets.'})
    rows=[]
    for seed in (42,43,44):
        by={x['cell']:x for x in output if x['seed']==seed}
        rows.append([seed]+[fmt(by[c]['historical_macro_ap']) for c in ('C-00','C-21','C-22')]
            +[f"{100*(by[c]['historical_macro_ap']-by['C-00']['historical_macro_ap']):+.3f}" for c in ('C-21','C-22')])
    return ('### E8 — seed 43·44 고정 설정 재검증 완료\n\n'
        +'총 3 seeds × 4 scenes × 5 folds에서 통계 모델 180개, 조합별 readout 1,620개를 평가했습니다. Seed 42의 각 fold에서 선택한 Fourier 차수·ridge, 데이터 분할과 특징 투영을 고정했습니다. 표는 주 readout의 historical macro AP입니다.\n\n'
        +table(['Seed','C-00 AP %','C-21 AP %','C-22 AP %','C-21 ΔAP pp','C-22 ΔAP pp'],rows)
        +'같은 자료에서의 seed 민감도이며 독립 데이터 재현을 의미하지 않습니다. [재검증 수치](results/E8/robustness.json)\n\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--complete-e8',action='store_true');p.add_argument('--e6-only',action='store_true');args=p.parse_args()
    text='## 후속 실험 E6–E10\n\n[사전 고정 실험 명세](docs/FOLLOWUP_EXPERIMENTS.md) · [후속 검사 로그](results/followup_tests.txt)\n\n'
    text+=report_e6()
    text+=('E8: Confidence 3×3 × readout 3종의 5-fold 실험 진행 중입니다.\n\n' if args.e6_only else report_e8(args.complete_e8))
    robust=report_robustness() if args.complete_e8 else ''
    if robust:
        text=text.replace('seed 43/44 재검증은 아직 실행하지 않았습니다.', 'seed 43/44 재검증 결과는 아래에 정리했습니다.')
        text+=robust
    text+='재실행: `PYTHONPATH=src python scripts/run_followup.py --data-root /path/to/IPAD_dataset` 후 `--seed 43`, `--seed 44`로 반복합니다. `scripts/prepare_annotation_packet.py`는 로컬 원본 경로에서 주석 뷰어를 생성합니다. `scripts/bootstrap_followup.py`와 `--candidate C-22`로 paired CI를 계산하고, `scripts/check_followup.py`로 저장된 점수와 수치를 검증합니다. [정합성 검사](results/followup_validation.txt)\n\n'
    text+='E7: T0/T1/T2 인과적 예측을 로컬에 저장했으며 [독립적인 실제 cycle/anchor 주석](docs/ANNOTATION_GUIDE.md)이 필요합니다. E9S: 합성 편집 검증은 E7 주석과 독립적으로 진행하며 아래 실행 상태를 따릅니다. 실제 유형별 E9R은 주석이 필요합니다. E10: 신규 독립 촬영 자료가 필요합니다. 미실행 항목을 완료로 표시하지 않습니다.\n\n'
    if (ROOT/'docs/PHASE_PROCESS_EXPERIMENTS.md').exists():
        text+='### 다음 실험 계획\n\n[위치별 정상 기준과 진행 이상 검증 계획](docs/PHASE_PROCESS_EXPERIMENTS.md)을 고정했습니다. E8B는 위치 조건 5개 비교군, E9S는 진행 점수 10개 비교군을 사용합니다. 편집 후보 3,996개 중 3,910개가 길이 검사를 통과했습니다. 실행 상태와 완료 수치는 아래 E8B / E9S 절을 따릅니다.\n\n'
    (ROOT/'docs/FOLLOWUP_RESULTS.md').write_text(text.replace('](docs/', '](').replace('](results/', '](../results/'))
    path=ROOT/'README.md';old=path.read_text()
    start=old.index('## 후속 실험');end=old.index('## 실험 상태',start)
    extra=''
    if (ROOT/'docs/PHASE_PROCESS_RESULTS.md').exists():
        extra=(ROOT/'docs/PHASE_PROCESS_RESULTS.md').read_text().replace('](../results/', '](results/').replace('](figures/', '](docs/figures/').replace('](PHASE_PROCESS_EXPERIMENTS.md)', '](docs/PHASE_PROCESS_EXPERIMENTS.md)')
    if (ROOT/'docs/PAPER_STAGE1_RESULTS.md').exists():
        extra+=(ROOT/'docs/PAPER_STAGE1_RESULTS.md').read_text().replace('](../results/', '](results/').replace('](../configs/', '](configs/').replace('](figures/', '](docs/figures/').replace('](PAPER_STAGE1_PROTOCOL.md)', '](docs/PAPER_STAGE1_PROTOCOL.md)')
    from report_fragments import diagnostics_fragment
    path.write_text(old[:start]+text+extra+diagnostics_fragment(ROOT)+old[end:])
    prior_status=read('results/followup_status.json') if (ROOT/'results/followup_status.json').exists() else {}
    write_json(ROOT/'results/followup_status.json',{**prior_status,'E6':'complete','E7':'predictions_ready_independent_annotations_required',
        'E8':('complete' if robust else 'seed42_complete_robustness_pending') if args.complete_e8 else 'running',
        'E8B':prior_status.get('E8B','planned_not_run'),'E9':prior_status.get('E9','synthetic_stage_planned_not_run'),'E9S':prior_status.get('E9S','planned_not_run'),
        'E9R':'independent_process_annotations_required','E10':'new_independent_recordings_required'})

if __name__=='__main__':main()
