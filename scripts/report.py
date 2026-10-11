"""Rebuild README and static research figures from completed-stage evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
SCENES=['R01','R02','R03','R04']
CORE=['pooled','appearance','cycle_conditioned','appearance_process','full']
LABELS={'pooled':'Pooled PCA','appearance':'Appearance (A0)','cycle_conditioned':'Cycle-conditioned (A1)','appearance_process':'Appearance + process (A2)','full':'Full (A3)'}
COLORS=['#6b7280','#b38c36','#cd7555','#688656','#4277ae']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#24303d','text.color':'#24303d','svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})
FIG=ROOT/'docs/figures'; FIG.mkdir(parents=True,exist_ok=True)

def read(p): return json.loads((ROOT/p).read_text())
def save(fig,name):
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight')
    plt.close(fig)
def fmt(x): return '—' if x is None else f'{x*100:.2f}'
def table(headers,rows): return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def metrics(stage): return {s:read(f'results/{stage}/{s}/metrics.json') for s in SCENES}
def macro(ms,v,k): return float(np.mean([ms[s][v][k] for s in SCENES]))
def scoretable(ms,vs):
    return table(['모델']+SCENES+['Macro'],[[LABELS.get(v,v)]+[f'{fmt(ms[s][v]["auroc"])} / {fmt(ms[s][v]["ap"])}' for s in SCENES]+[f'{fmt(macro(ms,v,"auroc"))} / {fmt(macro(ms,v,"ap"))}'] for v in vs])
def bars(ms,vs,name):
    fig,axes=plt.subplots(1,2,figsize=(13,5.2)); x=np.arange(5); width=.8/len(vs)
    for ax,key,title in zip(axes,['auroc','ap'],['Frame AUROC','Frame average precision']):
        for j,v in enumerate(vs):
            vals=[ms[s][v][key] for s in SCENES]+[macro(ms,v,key)]
            ax.bar(x-.4+width/2+j*width,vals,width,label=LABELS[v],color=COLORS[CORE.index(v)])
        ax.set_xticks(x,SCENES+['Macro']); ax.set_ylim(0,1); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(title); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.subplots_adjust(top=.77,bottom=.17); fig.text(.02,.015,'IPAD R01–R04 · strict-v1 label mask · seed 42 · stride 2 · Macro: unweighted scene mean',fontsize=10)
    save(fig,name)

def main():
    status=read('results/status.json') if (ROOT/'results/status.json').exists() else {}
    lines=['# CycleVAD — IPAD R01–R04 experiments\n',
    '공정의 연속 진행 위치를 추적해 조건부 외형 이상과 진행 흐름의 이상을 결합하는 subspace AD 실험입니다.\n',
    '로컬 **NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB**에서 특징을 추출하고 CPU에서 통계 모델을 학습합니다. 수치는 실제 결과 JSON에서 자동 생성합니다.\n',
    '## 실험 상태\n',table(['단계','내용','상태'],[[e,d,status.get(e,{}).get('status','pending')] for e,d in [('E0','환경·데이터·테스트'),('E1','R02 재현'),('E2','R01–R04 확장'),('E3','핵심 2×2 ablation'),('E4','세부 제거·이산 phase 대조'),('E5','3 seeds·stride·사례 분석')]]),
    '## 방법과 평가 규칙\n',
    '```mermaid\nflowchart LR\n V[Video] --> F[Frozen DINOv2 features]\n F --> T[Causal cycle tracker]\n F --> A[Pooled PCA + local memory]\n T --> C[Fourier mean + shared residual PCA]\n T --> P[Alignment / innovation / progress]\n A --> S[Normal-reference calibration]\n C --> S\n P --> S\n S --> Q[Variant-specific normal q99 threshold]\n```\n',
    '- DINOv2-base / 336px letterbox / layer -1,-3 / 6×6 patches / FP16 / primary stride 2.\n- Fit, validation, reference, threshold 영상을 분리합니다. 테스트 라벨로 설정·임계값을 고르지 않습니다.\n- 실제 cycle 경계가 없는 `weak_recording_alignment`입니다. Cycle 위치 정확도는 미측정입니다.\n- 원본 recording group 정보가 없어 파일 간 그룹 독립성은 입증하지 못했습니다.\n- 주지표: frame AUROC/AP. 표의 값은 %이며, `AUROC / AP` 순서입니다. Macro는 네 장면의 단순 평균입니다.\n- FPR/Recall/Event coverage는 정상 holdout q99 임계값 기준입니다. 지연은 탐지된 이벤트에 한정한 원본 프레임 수입니다.\n',
    '[구현 방법](docs/METHOD.md) · [전체 사전 실험 규칙](docs/EXPERIMENT_PROTOCOL.md) · [환경 및 패키지 버전](results/E0/environment.json) · [고정 데이터 분할](results/E0/splits.json)\n']
    if (ROOT/'docs/FOLLOWUP_RESULTS.md').exists():
        followup=(ROOT/'docs/FOLLOWUP_RESULTS.md').read_text().replace('](../results/', '](results/').replace('](figures/', '](docs/figures/').replace('](FOLLOWUP_EXPERIMENTS.md)', '](docs/FOLLOWUP_EXPERIMENTS.md)').replace('](ANNOTATION_GUIDE.md)', '](docs/ANNOTATION_GUIDE.md)').replace('](PHASE_PROCESS_EXPERIMENTS.md)', '](docs/PHASE_PROCESS_EXPERIMENTS.md)').replace('](PHASE_PROCESS_RESULTS.md)', '](docs/PHASE_PROCESS_RESULTS.md)')
        lines.insert(3,followup)
    elif (ROOT/'docs/FOLLOWUP_EXPERIMENTS.md').exists():
        lines.insert(3,'## 후속 실험 계획 — 아직 미실행\n\n[구체적인 E6–E10 실험 명세](docs/FOLLOWUP_EXPERIMENTS.md): 위치 추적 대조군 3개, Confidence 학습·추론 3×3 조합, 진행 이상 대조군 6개를 정의했습니다. 정상 5-fold 분할과 40개 영상의 주석 대상 목록을 고정했으며, 실제 주석과 후속 실험 결과는 아직 없습니다.\n')
    if 'E4' in status:
        m=metrics('E2'); ab=metrics('E4')
        lines.insert(3,'## 초기 실험 E0–E5 관찰\n\n'+f'- Seed 42, stride 2의 Full Macro AUROC는 **{fmt(macro(m,"full","auroc"))}%**, Appearance는 **{fmt(macro(m,"appearance","auroc"))}%**입니다.\n'+f'- 진행 점수의 추가 효과(A2−A0)는 Macro AUROC **{(macro(m,"appearance_process","auroc")-macro(m,"appearance","auroc"))*100:+.3f} pp**입니다. 조건부 외형의 추가 효과는 현재 설정에서 거의 없습니다.\n'+f'- Confidence 가중치를 제거하면 Full 대비 Macro AUROC가 **{(macro(ab,"no_confidence","auroc")-macro(ab,"full","auroc"))*100:+.3f} pp** 변합니다. 이는 신뢰도 가중 방식의 재검토 근거이며, 테스트 결과로 최적 모델을 확정한 것은 아닙니다.\n'+f'- R01 Full의 정상 프레임 오경보율은 **{fmt(m["R01"]["full"]["fpr"])}%**입니다. 정상 holdout q99 임계값이 테스트 정상 프레임에 잘 일반화되지 않아, 높은 Recall을 단독으로 해석하면 안 됩니다.\n'+'- R02에서 나타난 큰 향상이 R01·R04에서는 재현되지 않습니다. 장면별 결과와 음의 효과도 함께 공개합니다.\n')
    if (ROOT/'docs/PHASE_PROCESS_RESULTS.md').exists():
        phase=(ROOT/'docs/PHASE_PROCESS_RESULTS.md').read_text().replace('](../results/', '](results/').replace('](figures/', '](docs/figures/').replace('](PHASE_PROCESS_EXPERIMENTS.md)', '](docs/PHASE_PROCESS_EXPERIMENTS.md)').replace('](PHASE_PROCESS_RESULTS.md)', '](docs/PHASE_PROCESS_RESULTS.md)')
        idx=next(i for i,v in enumerate(lines) if v.startswith('## 실험 상태'))
        lines.insert(idx,phase)
    if (ROOT/'docs/PAPER_STAGE1_RESULTS.md').exists():
        paper=(ROOT/'docs/PAPER_STAGE1_RESULTS.md').read_text().replace('](../results/', '](results/').replace('](../configs/', '](configs/').replace('](figures/', '](docs/figures/').replace('](PAPER_STAGE1_PROTOCOL.md)', '](docs/PAPER_STAGE1_PROTOCOL.md)')
        idx=next(i for i,v in enumerate(lines) if v.startswith('## 실험 상태'))
        lines.insert(idx,paper)
    from report_fragments import diagnostics_fragment
    diagnostic=diagnostics_fragment(ROOT)
    if diagnostic:
        idx=next(i for i,v in enumerate(lines) if v.startswith('## 실험 상태'))
        lines.insert(idx,diagnostic)
    if (ROOT/'results/E0/data_audit.json').exists():
        a=read('results/E0/data_audit.json'); lines+=['## E0 — 데이터와 재현 기반\n',table(['장면','학습 영상','학습 프레임','테스트 영상','테스트 프레임','유효 평가','제외 프레임'],[[s]+[a['scenes'][s][k] for k in ['train_videos','train_frames','test_videos','test_frames','valid_frames','unknown_frames']] for s in SCENES])]
        lines+=['R02 테스트 12/13/14의 라벨 길이가 각각 1프레임씩 다릅니다. 원본을 수정하지 않고 세 영상 전체 1,912프레임을 평가에서 제외합니다. 모든 모델에 같은 `strict-v1` 마스크를 적용하며, 전체 공식 benchmark와 동일한 평가라고 주장하지 않습니다.\n', '[검사 결과](results/E0/data_audit.json) · [테스트 로그](results/E0/test_output.txt)\n']
        fig,ax=plt.subplots(figsize=(8,3.5)); x=np.arange(4); valid=[a['scenes'][s]['valid_frames'] for s in SCENES]; unknown=[a['scenes'][s]['unknown_frames'] for s in SCENES]
        ax.bar(x,valid,color='#4277ae',label='Evaluated'); ax.bar(x,unknown,bottom=valid,color='#cd7555',label='Excluded: label mismatch'); ax.set_xticks(x,SCENES); ax.set_ylabel('Test frames'); ax.set_title('Evaluation coverage by scene',pad=52); ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.23),ncol=2); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); save(fig,'E0_coverage'); lines+=['![평가 대상 프레임](docs/figures/E0_coverage.svg)\n']
    if 'E1' in status:
        m=read('results/E1/R02/metrics.json'); old={'appearance':(.796916,.726953),'cycle_conditioned':(.796962,.727088),'combined':(.916316,.874450)}
        comparison=[]
        for v,(u,a) in old.items(): comparison.append([v,f'{u*100:.4f} / {a*100:.4f}',f'{m[v]["auroc"]*100:.4f} / {m[v]["ap"]*100:.4f}',f'{(m[v]["auroc"]-u)*100:+.4f}'])
        lines+=['## E1 — R02 로컬 재실행\n',table(['분기','기존 노트북 AUROC/AP','로컬 AUROC/AP','ΔAUROC (pp)'],comparison),'기존 값은 노트북에 표시된 반올림 수치입니다. Colab의 전체 패키지 버전·원본 캐시가 없어 비트 단위 재현을 주장하지 않습니다. [실행 결과](results/E1/R02/metrics.json)\n']
    if 'E2' in status:
        ms=metrics('E2'); vs=['pooled','appearance','cycle_conditioned','full']; bars(ms,vs,'E2_scenes')
        lines+=['## E2 — R01–R04 확장\n',scoretable(ms,vs),'![장면별 성능](docs/figures/E2_scenes.svg)\n',table(['장면','Full FPR','Full Recall','이벤트 탐지/전체','탐지 이벤트 지연 중앙값'],[[s,fmt(ms[s]['full']['fpr']),fmt(ms[s]['full']['recall']),f'{ms[s]["full"]["detected_events"]}/{ms[s]["full"]["events"]}',ms[s]['full']['median_detected_event_delay_frames']] for s in SCENES])]
    if 'E3' in status:
        ms=metrics('E3'); lines+=['## E3 — 핵심 요소 2×2 ablation\n',scoretable(ms,CORE[1:]),'A0: 외형, A1: 외형+조건부 외형, A2: 외형+진행, A3: 모두 결합. 같은 특징·분할·추적기를 사용하며 A2는 E2의 동결된 예측을 재사용합니다.\n']
        effects={k:{'conditional_only':macro(ms,'cycle_conditioned',k)-macro(ms,'appearance',k),'process_only':macro(ms,'appearance_process',k)-macro(ms,'appearance',k),'conditional_on_process':macro(ms,'full',k)-macro(ms,'appearance_process',k)} for k in ['auroc','ap']}
        (ROOT/'results/E3/effects.json').write_text(json.dumps(effects,indent=2))
        lines+=[table(['효과 (Macro pp)','AUROC','AP'],[[label,f'{effects["auroc"][key]*100:+.3f}',f'{effects["ap"][key]*100:+.3f}'] for key,label in [('conditional_only','조건부 외형 추가: A1−A0'),('process_only','진행 추가: A2−A0'),('conditional_on_process','진행 위에 조건부 외형 추가: A3−A2')]])]
        fig,ax=plt.subplots(figsize=(8,4)); keys=list(effects['auroc']); labels=['Conditional: A1 - A0','Process: A2 - A0','Conditional on process: A3 - A2']; y=np.arange(3)
        for j,k in enumerate(['auroc','ap']): ax.barh(y+(j-.5)*.3,[effects[k][x]*100 for x in keys],height=.3,color=['#4277ae','#b38c36'][j],label=k.upper())
        ax.axvline(0,color='#24303d',lw=.8); ax.set_yticks(y,labels); ax.set_xlabel('Macro change (percentage points)'); ax.legend(frameon=False); ax.set_title('Core component contribution · seed 42, stride 2'); save(fig,'E3_effects'); lines+=['![핵심 요소 효과](docs/figures/E3_effects.svg)\n']
    if 'E4' in status:
        ms=metrics('E4'); vs=[v for v in ms['R01'] if v not in CORE]; lines+=['## E4 — 세부 ablation\n',scoretable(ms,['full']+vs)]
        fig,axes=plt.subplots(1,2,figsize=(13,6));
        for ax,k in zip(axes,['auroc','ap']):
            values=np.array([[(ms[s][v][k]-ms[s]['full'][k])*100 for s in SCENES]+[(macro(ms,v,k)-macro(ms,'full',k))*100] for v in vs]); lim=max(.1,float(np.abs(values).max())); im=ax.imshow(values,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto'); ax.set_xticks(range(5),SCENES+['Macro']); ax.set_yticks(range(len(vs)),vs); ax.set_title(f'{k.upper()} change vs Full (pp)')
            for i in range(len(vs)):
                for j in range(5): ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(values[i,j])>.6*lim else '#24303d')
            fig.colorbar(im,ax=ax,shrink=.8)
        fig.tight_layout(); save(fig,'E4_ablation'); lines+=['![세부 ablation](docs/figures/E4_ablation.svg)\n','양수는 해당 변형이 Full보다 높다는 뜻입니다. `no_local`은 local-memory 점수만 제거하며 지역 descriptor는 유지합니다. 이산 phase는 자체 구현한 4-bin 대조군입니다.\n']
        lines+=[table(['장면','이산 phase 실제 rank','연속 조건부 bytes','이산 조건부 bytes'],[[s,str(read(f'results/E4/{s}/discrete_model.json')['actual_ranks']),read(f'results/E4/{s}/discrete_model.json')['continuous_storage_bytes'],read(f'results/E4/{s}/discrete_model.json')['storage_bytes']] for s in SCENES])]
    if 'E4' in status:
        diagnostics=[]
        for scene in SCENES:
            ds=read(f'results/E4/{scene}/diagnostics.json')
            diagnostics.append([scene,f"{np.mean([r['mean_confidence'] for r in ds]):.3f}",f"{np.mean([r['conditional_lifts_appearance_fraction'] for r in ds])*100:.2f}",f"{np.mean([r['process_lifts_conditioned_fraction'] for r in ds])*100:.2f}"])
        lines+=['### 분기 활성화 진단\n',table(['장면','평균 confidence','조건부 점수가 외형을 초과 (%)','진행 점수가 A1을 초과 (%)'],diagnostics),'각 테스트 영상의 sampled-frame 평균을 구한 뒤 영상 간 단순 평균했습니다. 라벨 불일치 영상도 이 라벨 비의존 진단에는 포함합니다. Confidence는 위치 정확도의 확률이 아닙니다.\n']
    if 'E5' in status:
        base=metrics('E2'); seeds=[base,metrics('E5/seed43'),metrics('E5/seed44')]; stride=metrics('E5/stride1'); rows=[]; means=[]; stds=[]
        for v in CORE:
            vals={k:[macro(ms,v,k) for ms in seeds] for k in ['auroc','ap']}; means.append(np.mean(vals['auroc'])); stds.append(np.std(vals['auroc'],ddof=1)); rows.append([LABELS[v]]+[f'{np.mean(vals[k])*100:.2f} ± {np.std(vals[k],ddof=1)*100:.2f}' for k in ['auroc','ap']])
        summary={v:{k:{'seeds':[42,43,44],'macro_values':[macro(ms,v,k) for ms in seeds],'mean':float(np.mean([macro(ms,v,k) for ms in seeds])),'sample_sd':float(np.std([macro(ms,v,k) for ms in seeds],ddof=1))} for k in ['auroc','ap','fpr','recall','event_coverage']} for v in CORE}
        (ROOT/'results/E5/stability_summary.json').write_text(json.dumps(summary,indent=2))
        (ROOT/'results/E5/stride_summary.json').write_text(json.dumps({v:{k:{'stride2':macro(base,v,k),'stride1':macro(stride,v,k)} for k in ['auroc','ap','fpr','recall','event_coverage']} for v in CORE},indent=2))
        lines+=['## E5 — 안정성·stride·사례 분석\n',table(['모델','Macro AUROC 평균 ± SD','Macro AP 평균 ± SD'],rows),'Seed 42/43/44의 모델 무작위성만 변경했습니다. 정상 데이터 분할과 encoder projection은 고정입니다. 표준편차는 신뢰구간이 아니며 데이터 분할 불확실성은 포함하지 않습니다.\n']
        fig,ax=plt.subplots(figsize=(9,4)); ax.barh(np.arange(5),means,xerr=stds,color=COLORS,capsize=4); ax.set_yticks(range(5),[LABELS[v] for v in CORE]); ax.set_xlim(0,1); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_title('Macro AUROC · mean ± sample SD across 3 model seeds'); save(fig,'E5_seeds'); lines+=['![Seed 안정성](docs/figures/E5_seeds.svg)\n',table(['모델','Stride 2 Macro AUROC/AP','Stride 1 Macro AUROC/AP'],[[LABELS[v],f'{fmt(macro(base,v,"auroc"))} / {fmt(macro(base,v,"ap"))}',f'{fmt(macro(stride,v,"auroc"))} / {fmt(macro(stride,v,"ap"))}'] for v in ['pooled','full']])]
        lines+=['Stride 1은 descriptor 차분과 진행 점수의 **샘플 단위 lag를 그대로 유지**하므로 실제 원본 프레임 기준 시간 범위도 줄어듭니다. 순수한 샘플링 밀도 효과만 분리한 실험은 아닙니다.\n']
        fig,axes=plt.subplots(4,1,figsize=(12,10))
        for ax,s in zip(axes,SCENES):
            t=read(f'results/E2/{s}/timeline.json'); x=np.array(t['indices']);
            for v,c in [('appearance','#b38c36'),('full','#4277ae')]: ax.plot(x,t['scores'][v],color=c,label=LABELS[v])
            ax.axhline(t['thresholds']['full'],ls='--',color='#24303d',label='Full threshold'); ax.fill_between(x,0,1,where=np.array(t['labels'])==1,transform=ax.get_xaxis_transform(),color='#cd7555',alpha=.15,label='Anomaly label'); ax.set_title(t['id'],loc='left'); ax.set_ylabel('Calibrated score')
        axes[0].legend(ncol=4,fontsize=9,frameon=False); axes[-1].set_xlabel('Source frame index'); fig.tight_layout(); save(fig,'E5_timelines'); lines+=['![탐지 타임라인](docs/figures/E5_timelines.svg)\n','각 장면에서 이름순으로 처음 나타나는 유효 이상 영상을 표시했습니다. 성능이 좋은 사례를 골라내지 않았습니다. 각 패널은 장면별 보정 점수와 독립적인 y축 범위를 사용하므로 점수 크기를 장면 간 직접 비교하지 않습니다.\n']
    if 'E5' in status:
        failures=[]
        for scene in SCENES:
            rs=[r for r in read(f'results/E2/{scene}/per_sequence.json') if r['variant']=='full' and r['valid_frames']>0]
            fp=[r for r in rs if r['fpr'] is not None]
            fn=[r for r in rs if r['recall'] is not None]
            for kind,chosen in [('highest FPR',max(fp,key=lambda r:r['fpr'])),('lowest recall',min(fn,key=lambda r:r['recall']))]:
                failures.append([scene,kind,chosen['id'],fmt(chosen['fpr']),fmt(chosen['recall']),f"{chosen['detected_events']}/{chosen['events']}"])
        lines+=['### 오류 사례 점검\n',table(['장면','선정 기준','영상','FPR','Recall','탐지 이벤트'],failures),'각 장면에서 Full의 FPR 최대 영상과 Recall 최소 이상 영상을 진단 목적으로 선정했습니다. 대표 표본이 아니며 이상 유형의 원인을 자동 확정하지 않습니다.\n']
    if status:
        lines+=['## 실행 비용\n',table(['단계','실측 wall time (초)'],[[s,f'{v["elapsed_seconds"]:.1f}'] for s,v in status.items()]),'E0 시간은 테스트 실행만 포함합니다. E1 이후 시간은 해당 단계의 특징 추출·학습·공유 점수 계산을 포함하며 업로드/환경 설치 시간은 제외합니다. 개별 ablation의 독립 추론 latency로 해석하지 않습니다.\n']
    if 'E5' in status:
        env=read('results/E5/environment.json')
        lines+=[f"E5 PyTorch allocator peak: allocated {env['peak_allocated_bytes']/2**30:.3f} GiB, reserved {env['peak_reserved_bytes']/2**30:.3f} GiB. 드라이버와 다른 프로세스 메모리는 포함하지 않습니다.\n"]
    lines+=['## 재현\n','```bash\npython3 -m venv .venv\n.venv/bin/pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128\n.venv/bin/pip install -r requirements-local.txt\nexport PYTHONPATH=src\nexport HF_HOME="$PWD/.cache/huggingface"\nexport MPLCONFIGDIR="$PWD/.cache/matplotlib"\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n.venv/bin/python scripts/audit.py --data-root /path/to/IPAD_dataset\n.venv/bin/python -m pytest -q\n# E1 → E2 → E3 → E4 → E5 순서로 실행\n.venv/bin/python scripts/run_stage.py --stage E1 --config configs/experiments/replay_pinned.json --data-root /path/to/IPAD_dataset\n.venv/bin/python scripts/report.py\n```\n','사전학습 checkpoint revision은 [고정 재현 설정](configs/experiments/replay_pinned.json)에 기록했습니다. 실제 실행 환경은 [lock 파일](results/E0/requirements-lock.txt)을 참고하세요. E0 완료 표시는 검사·테스트 통과 후 기록합니다. 원본 프레임, 특징 캐시, 모델 체크포인트는 Git에 포함하지 않습니다. 각 단계의 JSON/CSV, 설정, 코드 fingerprint와 그래프를 공개합니다. [최종 22개 테스트](results/final_tests.txt)와 [결과 정합성 검사](results/final_validation.txt)도 확인할 수 있습니다.\n','## 한계\n','실제 cycle 경계, 원본 recording 그룹, pixel localization GT가 없는 상태입니다. R02 라벨 불일치 영상은 제외했고 알려진 이상 유형별 주석이 없어 정지/역행/생략별 실데이터 탐지 성능을 별도로 주장하지 않습니다. 테스트 비교 결과로 최적 모델을 자동 선택하지 않습니다.\n']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('README and figures regenerated for:',', '.join(status))

if __name__=='__main__': main()
