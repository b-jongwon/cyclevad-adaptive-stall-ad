"""Paper-stage results generated from checked numerical artifacts."""
import argparse,json
from report_fragments import diagnostics_fragment
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SCENES=['R01','R02','R03','R04']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cyclevad-paper-stage1','figure.facecolor':'white'})
def read(p):return json.loads((ROOT/p).read_text())
def pct(v):return '—' if v is None else f'{100*v:.2f}'
def table(h,rows):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def save(fig,name):
    d=ROOT/'docs/figures';fig.savefig(d/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});fig.savefig(d/f'{name}.png',bbox_inches='tight',dpi=150);plt.close(fig)
def ci(v):return f'{v["estimate"]:+.2f} [{v["ci95"][0]:+.2f}, {v["ci95"][1]:+.2f}]'
def e14():
    allrows=read('results/E14/summary.json');s=allrows[0];rows=[]
    for ed,r in s['by_type'].items():
        for part in ['initial','interior']:
            a=r[f'{part}_edited_rate'];b=r[f'{part}_original_rate'];rows.append([ed,part,r[f'{part}_eligible_cases']]+[pct(a[i])+' / '+pct(b[i]) if a is not None else '해당 없음' for i in [4,7,9]])
    text='### E14A — 기존 합성 편집의 경계·내부 반응 분석 완료\n\nE9S 15,640개 편집 평가에서 최초 32프레임과 유효한 편집 내부를 분리하고, 각 구간과 정확히 같은 위치·길이의 원본 정상 창을 비교했습니다. 아래는 seed 42, stride 2입니다. 값은 **편집 경보율 / 원본 경보율 %**입니다.\n\n'
    text+=table(['유형','구간','적격 편집 수','P4 진행 결합','P7 다중 차분','P9 관측 기반'],rows)
    text+='![경계 및 내부 반응](docs/figures/E14_boundary.svg)\n\n역행·구간 교환은 유효 내부에서도 P4의 편집–원본 경보율 차이가 남았습니다. 정지는 P4의 내부 반응도 정상 원본과 비슷하여, 실패를 편집 시작점만의 문제로 설명하기 어렵습니다. 내부 평가에는 짧은 편집이 제외되므로 처음 32프레임과 다른 조건부 표본이며 창 길이도 다릅니다. **내부 구간에도 경계에서 바뀐 추적 상태가 남으므로 편집 흔적과 독립적인 순서 이해를 입증한 결과는 아닙니다.** Skip에는 지속 내부 구간이 없습니다.\n\n'
    text+='원본 영상 동일 가중 → severity 동일 가중 → 적격 scene 동일 가중으로 집계했습니다. 이 분석은 점추정이며 신규 학습은 없습니다. 세 seed와 stride 1의 coverage·수치는 [상세 결과](results/E14/summary.json)에 있습니다. 새로운 변화량 대응 대조군과 편집 후 회복 실험은 E14 후속 항목으로 남깁니다.\n\n'
    fig,axes=plt.subplots(1,2,figsize=(11,4.2));colors=['#176e72','#6975b4','#ae557d']
    for ax,part in zip(axes,['initial','interior']):
        edits=[e for e,r in s['by_type'].items() if r[f'{part}_edited_rate'] is not None];x=np.arange(len(edits))
        for j,a in enumerate([4,7,9]):
            vals=[100*(s['by_type'][ed][f'{part}_edited_rate'][a]-s['by_type'][ed][f'{part}_original_rate'][a]) for ed in edits]
            ax.bar(x+(j-1)*.23,vals,.23,label=f'P{a}',color=colors[j])
        ax.axhline(0,color='#555',lw=.8);ax.set_xticks(x,[e.replace('swap_adjacent_blocks','swap') for e in edits]);ax.set_ylim(-10,80);ax.set_ylabel('Edited minus matched normal alarm (pp)');ax.set_title('First 32 frames' if part=='initial' else 'Eligible edited interior after 32 frames');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,ncol=3);fig.tight_layout();save(fig,'E14_boundary');return text

def e11():
    hs=[read(f'results/E11/seed{seed}/historical_summary.json') for seed in [42,43,44]];ss=[read(f'results/E11/seed{seed}/synthetic_summary.json') for seed in [42,43,44]]
    h,s=hs[0],ss[0];hb=read('results/E11/seed42/historical_bootstrap.json');sb=read('results/E11/seed42/synthetic_bootstrap.json');names=['A','A+C','A+P','A+C+P','C only','P only']
    text='### E11 — 동일 파이프라인 통합 ablation 완료\n\n60개 scene×fold×seed에서 같은 PhysicalTracker·외형 모델을 공유하고, 새 조건부 평균·잔차 PCA와 모든 점수의 정상 reference 보정을 구성했습니다. C는 균등 학습·gate 없음, P는 alignment·innovation·progress입니다. E8B와 E9S의 저장 점수를 사후 합친 결과가 아닙니다.\n\n'
    text+=table(['모델','Historical AP %','AUROC %','합성 event %','Matched 정상 FAR %','Grid 정상 FAR %'],[[f'{r["arm"]} {names[j]}',pct(r['historical_macro_ap']),pct(r['historical_macro_auroc'])]+[pct(s['macro'][k][j]) for k in ['event','matched_far','grid_far']] for j,r in enumerate(h['arms'])])
    text+='주 운영점은 모델별 정상 window-q99이며 목표 보정 규칙이 같아도 평가 FAR은 같지 않습니다. Historical은 이전 개발에 사용한 테스트 fold 0이고 이상 유형은 미주석입니다. 합성은 정상 OOF 111개 원본의 고정 3,910개 편집/seed입니다. 주 event deadline은 합성 0.2T, historical min(이벤트 길이,0.2T)입니다. AUROC/AP는 historical 프레임 단위, 합성 점수 순위는 별도 JSON에 window 단위로 공개합니다.\n\n'
    rows=[]
    for control in ['B2','B1']:
        a=next(r for r in hb['contrasts'] if r['contrast']==f'B3-{control}' and r['metric']=='ap_gain_pp');b=next(r for r in sb['contrasts'] if r['contrast']==f'B3-{control}' and r['metric']=='event_gain_pp')
        rows.append([f'B3−{control}',ci(a['macro']),ci(b['macro'])])
    text+=table(['주 비교','Historical AP 차이 pp [95% CI]','합성 event 차이 pp [95% CI]'],rows)
    text+='CI는 seed 42의 고정 모델에 대해 장면별 원본 영상 paired bootstrap 2,000회로 계산했습니다. C를 A+P에 추가한 AP 개선 구간은 양수지만, P를 A+C에 추가한 AP 개선 구간은 0을 포함합니다. 합성에서는 P 추가 효과가 크고 C 추가 효과가 작습니다. 높은 평균 성능만으로 상보성이나 동일 실제 오경보 조건의 우위를 확정하지 않습니다.\n\n'
    text+=table(['Seed','B1 AP %','B2 AP %','B3 AP %','B3 합성 event %','B3 matched FAR %'],[[v['seed']]+[pct(v['arms'][i]['historical_macro_ap']) for i in [1,2,3]]+[pct(w['macro'][k][3]) for k in ['event','matched_far']] for v,w in zip(hs,ss)])
    text+='![통합 모델과 증분 효과](docs/figures/E11_integration.svg)\n\n'
    text+=table(['장면','이벤트 수','B1만','P만','둘 다','둘 다 실패','Full 탐지','B1 대비 추가/손실','B2 대비 추가/손실','합집합 대비 Full 손실'],[[v['scene'],v['events'],v['appearance_only'],v['process_only'],v['both'],v['neither'],v['B3_detected'],f'{v["B3_added_over_B1"]}/{v["B3_lost_from_B1"]}',f'{v["B3_added_over_B2"]}/{v["B3_lost_from_B2"]}',v['B3_lost_from_union']] for v in h['event_overlap']])
    text+='표는 historical 이벤트의 실제 경보 결과입니다. B1과 P는 각자의 정상 임계값을 사용합니다. 결합 임계값을 다시 보정하므로 점수의 max 관계가 탐지 집합의 포함 관계를 보장하지 않습니다. 실제 유형별 상보성은 독립 주석 후 확인해야 합니다.\n\n'
    text+='조건부 점수 계산은 float64로 수행했습니다. float32 행렬 연산의 batch 길이에 따른 반올림이 경험적 보정 점수에 증폭되는 것을 발견하여 모든 E11 분할을 다시 실행했고, 편집 이전 점수·추적 상태 불변성을 검사했습니다. [아티팩트 검증](results/E11/validation.txt) · [Historical 수치](results/E11/seed42/historical_summary.json) · [합성 수치](results/E11/seed42/synthetic_summary.json) · [Historical CI](results/E11/seed42/historical_bootstrap.json) · [합성 CI](results/E11/seed42/synthetic_bootstrap.json)\n\n'
    fig,axes=plt.subplots(1,2,figsize=(11,4.3));x=np.arange(6)
    for j,v in enumerate(hs):axes[0].plot(x,[r['historical_macro_ap']*100 for r in v['arms']],'-o',label=f'Seed {v["seed"]}')
    axes[0].set_xticks(x,['B0','B1','B2','B3','C','P']);axes[0].set_ylabel('Historical macro AP (%)');axes[0].set_title('Unified appearance and process scores');axes[0].legend(frameon=False);axes[0].grid(axis='y',alpha=.15)
    for j,a in enumerate([0,1,2,3,5]):
        curves=np.mean([np.array(s['normal'][sc]['operating_curves_event_matchedfar']) for sc in SCENES],axis=0)*100
        axes[1].plot(curves[1,:,a],curves[0,:,a],'-o',ms=3,label=s['arms'][a]);axes[1].scatter(curves[1,3,a],curves[0,3,a],s=65)
    axes[1].set_xlabel('Matched normal window alarm (%)');axes[1].set_ylabel('Synthetic event hit (%)');axes[1].set_title('Fixed normal calibration quantiles; large = q99');axes[1].set_ylim(0,100);axes[1].set_xlim(left=0);axes[1].legend(frameon=False,ncol=3);axes[1].grid(alpha=.15);fig.tight_layout();save(fig,'E11_integration');return text

def e12():
    rows=read('results/E12/summary.json');boot=read('results/E12/paired_bootstrap.json')
    arms=['S0','S1','S2_K4','S2_K8','S3_K4','S3_K8','S4_K4','S4_K8','S5']
    def row(a,f=1.,seed=42,out='head_only'):return next(r for r in rows if r['arm']==a and r['fraction']==f and r['seed']==seed and r['readout']==out)
    desc=['global mean/shared','continuous/shared','4-bin mean/shared','8-bin mean/shared','continuous/4 PCA','continuous/8 PCA','4-bin mean/4 PCA','8-bin mean/8 PCA','continuous/norm only']
    text='### E12 — 평균 함수와 공유 subspace 분리 비교 완료\n\n60개 scene×fold×seed에서 9개 구조×3개 FIT 비율을 평가했습니다. S1은 연속 평균+공유 PCA이며, S3은 같은 연속 평균에 phase별 PCA를 적용해 공유 여부만 바꿉니다. 모든 PCA 조건은 해당 분할·부분집합의 동일한 총 rank를 사용하고 variance 조기 절단을 하지 않았습니다. 아래는 seed 42, FIT 100%입니다.\n\n'
    text+=table(['구조','Head AP %','A+Head AP %','Head AUROC %','모델 배열 KiB','보정 포함 KiB','완료 분할'],[[f'{a} {desc[j]}',pct(row(a)['historical_macro_ap']),pct(row(a,out='appearance_head')['historical_macro_ap']),pct(row(a)['historical_macro_auroc']),f'{row(a)["model_array_bytes"]/1024:.1f}',f'{row(a)["head_array_bytes"]/1024:.1f}',f'{row(a)["completed_units"]}/20'] for j,a in enumerate(arms)])
    text+='AP/AUROC는 historical fold 0의 장면 평균입니다. 저장량은 20개 scene/fold 평균이며 추적기·공통 A·encoder는 제외합니다. 같은 총 rank라도 phase별 평균·고유값 및 μ 계수의 저장량이 달라 **동일 bytes 비교가 아닙니다**. 실제 저장량–성능 관계를 보고합니다.\n\n'
    selected=[r for r in boot['contrasts'] if r['readout']=='head_only' and r['metric']=='ap_gain_pp']
    text+=table(['비교','Head AP 차이 pp [95% CI]'],[[r['contrast'],ci(r['macro'])] for r in selected])
    text+='![공유 subspace 구조와 표본 효율](docs/figures/E12_subspaces.svg)\n\n'
    text+=table(['FIT 비율','S0 AP %','S1 AP %','S3_K4 AP %','S3_K8 AP %','S5 AP %'],[[f'{int(f*100)}%']+[pct(row(a,f)['historical_macro_ap']) for a in ['S0','S1','S3_K4','S3_K8','S5']] for f in [.25,.5,1.]])
    text+='부분집합은 원본 FIT 영상의 고정 해시 순서로 중첩 구성했습니다. 추적기·A와 평균 함수의 하이퍼파라미터는 full FIT/normal validation에서 고정했으므로 **외형 head의 표본 효율 실험이며 전체 시스템 few-shot 성능이 아닙니다**. 각 조건의 유지 rank, phase 표본·영상 수, 정상 μ MSE/외부 잔차, 정상 FPR, seed 43/44와 readout별 결과를 공개합니다. Head 단독 batch1 지연은 공유 호스트의 참고 실측이며 encoder/추적기를 포함한 실시간 지연이 아닙니다.\n\n'
    text+='1,620개 head가 모두 완료되었고 unavailable 조건은 없었습니다. Full FIT의 모든 PCA 조건은 총 rank 64였습니다. 다만 seed 44 / R03 / fold 1의 FIT 25%에서는 한 phase bin에 표본이 3개뿐이어서, 사전 규칙에 따라 **모든 PCA 비교군의 총 rank를 함께 8로 낮췄습니다**. 나머지 부분집합은 총 rank 64입니다. 이 데이터와 총 rank 예산에서는 연속 평균+공유 PCA가 전역 평균, phase별 PCA, norm-only 대조군보다 좋은 AP를 보였습니다. 이는 공유 잔차 공간의 선택을 지지하지만, phase별 PCA에 더 큰 rank나 별도 조율을 허용한 경우까지 우월함을 입증하지는 않습니다.\n\n'
    text+='[전체 구조·비율·seed 수치](results/E12/summary.json) · [paired CI](results/E12/paired_bootstrap.json) · [저장 점수 재검증](results/E12/validation.txt)\n\n'
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));colors=plt.get_cmap('tab10').colors
    for j,a in enumerate(arms):
        r=row(a);axes[0].scatter(r['head_array_bytes']/1024,r['historical_macro_ap']*100,s=45,color=colors[j])
        offset={'S0':(-23,-16),'S2_K4':(-40,23),'S2_K8':(8,12)}.get(a,(4,4))
        axes[0].annotate(a,(r['head_array_bytes']/1024,r['historical_macro_ap']*100),xytext=offset,textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','color':'#888','lw':.6} if a.startswith('S2') else None)
    axes[0].set_xlabel('Head arrays incl. calibration (KiB)');axes[0].set_ylabel('Historical macro AP (%)');axes[0].set_title('Measured storage and accuracy, seed 42');axes[0].margins(x=.18,y=.15);axes[0].grid(alpha=.15)
    for a in ['S0','S1','S3_K4','S3_K8','S5']:axes[1].plot([25,50,100],[row(a,f)['historical_macro_ap']*100 for f in [.25,.5,1.]],'-o',label=a,color=colors[arms.index(a)])
    axes[1].set_xticks([25,50,100]);axes[1].set_xlabel('FIT source videos used by head (%)');axes[1].set_ylabel('Historical macro AP (%)');axes[1].set_title('Fixed full-data tracker; head data efficiency');axes[1].legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.18));axes[1].grid(alpha=.15);fig.tight_layout();save(fig,'E12_subspaces');return text

def root_links(text):return text.replace('](../results/','](results/').replace('](figures/','](docs/figures/').replace('](PAPER_STAGE1_PROTOCOL.md)','](docs/PAPER_STAGE1_PROTOCOL.md)')
def main():
    p=argparse.ArgumentParser();p.add_argument('--stages',nargs='+',choices=['E11','E12','E14'],default=['E14']);a=p.parse_args()
    text='## 논문 통합 실험 E11·E12·E14\n\n[고정 실행 규칙](docs/PAPER_STAGE1_PROTOCOL.md) · [실행 설정](configs/experiments/followup/paper_stage1_v1.json) · [코드 검사](results/paper_stage1_tests.txt)\n\n'
    for stage,fn in [('E11',e11),('E12',e12),('E14',e14)]:
        if stage in a.stages:text+=fn()
    text+='미완료 후속 항목: E7 실제 위치·유형 주석, E13 시간 모델·정지 보완, E15 보정 개선, 새로운 E14 대응 편집·회복 검증, 개선 후 최종 E11, 외부 baseline, E10 독립 촬영, end-to-end 온라인 지연. 이 단계의 historical/합성 결과는 탐색적 근거이며 이 항목들을 대신하지 않습니다.\n\n'
    doc=text.replace('](docs/','](').replace('](results/','](../results/').replace('](configs/','](../configs/')
    (ROOT/'docs/PAPER_STAGE1_RESULTS.md').write_text(doc)
    path=ROOT/'README.md';old=path.read_text();end=old.index('## 실험 상태');start=old.find('## 논문 통합 실험 E11·E12·E14');start=end if start<0 else start
    path.write_text(old[:start]+text+diagnostics_fragment(ROOT)+old[end:])
    status=read('results/followup_status.json')
    for stage in a.stages:status[stage]='boundary_analysis_complete_controls_pending' if stage=='E14' else 'complete'
    for stage in ['E11','E12','E13','E15']:status.setdefault(stage,'planned_not_run')
    (ROOT/'results/followup_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('Published report sections:',a.stages)
if __name__=='__main__':main()
