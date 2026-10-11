"""Publish source-backed E8B/E9S tables and static research figures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SCENES=['R01','R02','R03','R04'];OFF=[f'M4_{i}' for i in range(1101,1106)]
COLORS=['#798996','#be8545','#7088c2','#aa74ad','#176e72','#de805e','#938968','#6268ad','#32898a','#aa4d72']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','figure.facecolor':'white','svg.hashsalt':'cyclevad-phase-process-v2'})
def read(p):return json.loads((ROOT/p).read_text())
def pct(v):return f'{100*v:.2f}'
def ci(v):return f'{v["estimate"]:+.2f} [{v["ci95"][0]:+.2f}, {v["ci95"][1]:+.2f}]'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def save(fig,name):
    folder=ROOT/'docs/figures';fig.savefig(folder/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});fig.savefig(folder/f'{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)
def e8():
    data=read('results/E8B/summary.json');boot=read('results/E8B/paired_bootstrap.json')
    def row(arm,seed=42,out='appearance_conditional'):return next(r for r in data if r['arm']==arm and r['seed']==seed and r['readout']==out)
    arms=['M0','M1','M2','M3']+OFF
    text='### E8B — 위치별 정상 기준 검증 완료\n\n60개 장면×fold×seed, 640개 비교 모델을 평가했습니다. Confidence 가중치와 추론 게이트를 제거하고 진행 점수를 제외했습니다. M0=θ 없는 pooled PCA, M1=첫 관측 위치+정상 FIT 시간, M2=관측별 위치, M3=누적 추적 위치(C-22), M4=영상별 고정 무작위 위치 이동입니다. M4의 다섯 offset은 별도 학습·보정한 대조군이며 독립 영상으로 세지 않았습니다.\n\n'
    text+=table(['조건','Historical AP %','AUROC %']+[s+' 정상 FPR %' for s in SCENES],[[a,pct(row(a)['historical_macro_ap']),pct(row(a)['historical_macro_auroc'])]+[pct(row(a)['normal_by_scene'][s]['frame_fpr']) for s in SCENES] for a in arms])
    text+='표는 seed 42, 외형+조건부 점수입니다. Historical은 fold 0의 기존 테스트, 정상 FPR은 5-fold OOF 전체 정상 프레임 기준입니다. 라벨 길이 불일치 R02 테스트 12/13/14는 전체 제외했습니다. M0의 결합 점수는 외형 점수와 정확히 같습니다. M3의 균등 μ 재학습 동등성을 확인했습니다.\n\n'
    text+='![위치 조건 비교](docs/figures/E8B_phase_comparison.svg)\n\n'
    primary=[r for r in boot['historical'] if r['readout']=='appearance_conditional' and r['metric']=='ap_gain_pp']
    text+=table(['사전 지정 비교','Macro AP 차이 pp [95% CI]'],[[r['contrast'],ci(r['macro'])] for r in primary])
    text+='신뢰구간은 seed 42 모델을 고정한 장면별 원본 영상 단위 paired bootstrap 2,000회입니다. 파일 간 원 촬영의 독립성은 미확인이고 다중 비교 보정은 하지 않았습니다. 현재 관측을 이용한 μ의 정상 재구성 오차는 위치 GT 정확도나 미래 예측 정확도가 아닙니다.\n\n'
    text+=table(['Seed','M0 AP %','M3 AP %','M3−M0 pp','M3−M4 평균 pp'],[[s,pct(row('M0',s)['historical_macro_ap']),pct(row('M3',s)['historical_macro_ap']),f'{100*(row("M3",s)["historical_macro_ap"]-row("M0",s)["historical_macro_ap"]):+.2f}',f'{100*(row("M3",s)["historical_macro_ap"]-np.mean([row(a,s)["historical_macro_ap"] for a in OFF])):+.2f}'] for s in [42,43,44]])
    normdiff=[100*(row('M3')['normal_by_scene'][s]['frame_fpr']-row('M0')['normal_by_scene'][s]['frame_fpr']) for s in SCENES]
    main=next(r for r in primary if r['contrast']=='M3-M0')['macro'];scramble=next(r for r in primary if r['contrast']=='M3-M4_mean')['macro']
    point=main['estimate']>=1 and max(normdiff)<=1
    text+=f'사전 점추정 기준(AP +1 pp, 각 장면 정상 FPR 증가 ≤1 pp)은 **{"충족" if point else "미충족"}**입니다. 최대 장면 FPR 증가는 {max(normdiff):+.2f} pp입니다. M3−M0 및 M3−M4 CI의 0 포함 여부는 각각 **{"포함" if main["ci95"][0]<=0<=main["ci95"][1] else "미포함"} / {"포함" if scramble["ci95"][0]<=0<=scramble["ci95"][1] else "미포함"}**입니다. 이 결과는 위치 조건의 추가 가치를 탐색적으로 뒷받침하지만, 기존 테스트를 사용했으므로 독립 확증 결과는 아닙니다.\n\n'
    sensitivity=['M0_fixedrank','M3_fixedrank','M1_retuned','M2_retuned','M3_retuned']
    text+=table(['민감도 조건 (seed 42)','외형+조건부 AP %','조건부 단독 AP %'],[[a,pct(row(a)['historical_macro_ap']),pct(row(a,out='conditional_only')['historical_macro_ap'])] for a in sensitivity])
    text+='고정 rank에서는 두 모델의 축 수를 동일하게 맞췄고, retuned에서는 각 조건에 동일한 정상 validation 9개 조합 탐색을 적용했습니다. 주 비교와 분리하여 보고합니다. 개별 장면의 μ MSE, subspace 외부 MSE, 유지 rank, 영상 평균 FPR과 historical recall·활성화는 아래 JSON에 공개했습니다.\n\n'
    text+='[E8B 요약](results/E8B/summary.json) · [paired CI](results/E8B/paired_bootstrap.json) · [점수 재검증](results/E8B/validation.txt) · [분할별 원자료](results/E8B/)\n\n'
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1.15,1]})
    for j,s in enumerate([42,43,44]):axs[0].plot(np.arange(9),[100*row(a,s)['historical_macro_ap'] for a in arms],marker='o',label=f'Seed {s}',alpha=.85)
    axs[0].set_xticks(np.arange(9),['M0','M1','M2','M3']+[f'M4\n{i}' for i in range(1,6)]);axs[0].set_ylabel('Historical macro AP (%)');axs[0].set_title('Phase controls, same appearance channels');axs[0].legend(frameon=False);axs[0].grid(axis='y',alpha=.2)
    for j,r in enumerate(primary):
        m=r['macro'];axs[1].errorbar(m['estimate'],j,xerr=[[m['estimate']-m['ci95'][0]],[m['ci95'][1]-m['estimate']]],fmt='o',color='#176e72',capsize=4)
    axs[1].axvline(0,color='#89949a',lw=1);axs[1].set_yticks(range(4),[r['contrast'] for r in primary]);axs[1].set_xlabel('AP difference (percentage points)');axs[1].set_title('Paired source-file 95% intervals, seed 42');axs[1].invert_yaxis();fig.tight_layout();save(fig,'E8B_phase_comparison')
    return text

def e9():
    paths=[f'results/E9S/stride2/seed{s}/summary.json' for s in [42,43,44]]+['results/E9S/stride1/seed42/summary.json'];allruns=[read(p) for p in paths];s=allruns[0];b=read('results/E9S/stride2/seed42/paired_bootstrap.json')
    names=['Appearance','+ alignment','+ innovation','+ progress','+ all process','+ AR(1)','+ difference 2','+ difference 2/8/32','+ innovation/progress','+ observation-only process']
    text='### E9S — 합성 진행 이상 검증 완료\n\n정상 OOF 영상 111개에 정지·역행·생략·인접 구간 교환을 적용했습니다. 길이 조건을 통과한 3,910개 편집을 seed 42/43/44, stride 2에서 반복하고 seed 42는 stride 1에서도 확인했습니다(총 15,640개 편집 평가, 80개 모델 분할). 제외한 86개 후보도 사전 목록에 남겼습니다. 실제 산업 고장 유형이나 위치 GT 검증을 의미하지 않습니다.\n\n'
    text+='원본 프레임 순서를 편집한 뒤 출력 시간을 샘플링하고 추적 상태·차분·AR 예측을 처음부터 다시 계산했습니다. 모델은 원본 인덱스와 편집 위치를 입력받지 않습니다. 인과적 descriptor lag=4, 차분/진행 lag=2·8·32 원본 프레임으로 두 stride를 맞췄습니다. FIT 영상 길이 중앙값 T는 실제 cycle 주석이 아닌 약한 시간 기준입니다.\n\n'
    text+=table(['모델','Window AUROC %','AP %','Event hit %','Matched normal FAR %','Grid normal FAR %'],[[f'P{i} {names[i]}']+[pct(s['macro'][k][i]) for k in ['auroc','ap','event','matched_far','grid_far']] for i in range(10)])
    text+='주 운영점은 별도 정상 threshold 영상의 0.2T window maximum q99입니다. Event는 편집 후 0.2T 이내 탐지, AUROC/AP는 동일 길이 0.4T 원본/편집 window maximum 비교입니다. AP의 가중 양성 비율은 50%입니다. 정상 matched FAR은 편집 위치와 같은 원본 창, grid FAR은 정상 영상 전체의 고정 창입니다. q99는 평가 정상 영상에서 1% FAR을 보장하지 않습니다.\n\n'
    text+='각 scene/fold/type/severity 안에서 원본 영상마다 같은 총 가중치를 주고, fold AUROC/AP를 적격 영상 수로 평균한 뒤 severity/type/scene을 동일 비중으로 평균했습니다. 서로 다른 fold의 점수를 합쳐 AUROC를 계산하지 않았습니다.\n\n'
    text+='![진행 이상 탐지 및 유형 비교](docs/figures/E9S_detection.svg)\n\n'
    type_rows=[]
    for edit in ['freeze','reverse','skip','swap_adjacent_blocks']:
        group=[r for r in s['groups'] if r['edit']==edit]
        type_rows.append([edit]+[pct(np.mean([r['event'][i] for r in group])) for i in [4,5,7,9]])
    text+=table(['편집 유형','P4 event %','P5 event %','P7 event %','P9 event %'],type_rows)
    text+=f'**정지는 뚜렷한 예외입니다.** P4의 정지 탐지율은 {type_rows[0][1]}%로, 양방향 다중 차분 P7의 {type_rows[0][3]}%보다 낮습니다. 평균 향상은 주로 역행·생략·구간 교환에서 나왔으며, 세 가지 진행 이상을 모두 잘 잡는다고 주장하면 안 됩니다.\n\n'
    contrasts=[r for r in b['contrasts'] if r['metric']=='event_gain_pp']
    text+=table(['비교','Event 차이 pp [95% CI]'],[[r['contrast'],ci(r['macro'])] for r in contrasts])
    text+='원본 영상으로 묶은 paired bootstrap 2,000회이며 모델은 고정했습니다. 모든 편집·원본 창은 같은 cluster에 남깁니다. 전체 AUROC/AP·정상 FAR 구간도 JSON으로 공개합니다.\n\n'
    text+=table(['장면','P4 event %','P4−P7 event pp','P4 matched FAR %','P4 grid FAR %'],[[scene,pct(s['by_scene'][scene]['event'][4]),f'{100*(s["by_scene"][scene]["event"][4]-s["by_scene"][scene]["event"][7]):+.2f}',pct(s['by_scene'][scene]['matched_far'][4]),pct(s['by_scene'][scene]['grid_far'][4])] for scene in SCENES])
    text+='**진행 이상 탐지의 추가 가치는 관찰되지만, 사전 오경보 기준은 미충족입니다.** P4의 정상 matched/grid FAR이 여러 장면에서 5%를 넘습니다. P4−P9 이벤트 차이의 CI는 0을 포함하므로, 누적 위치 추적이 관측 기반 진행 점수보다 운영점 탐지율을 높였다고 확정할 수 없습니다. P1도 시간 차분 descriptor를 쓰므로 순수 정적 외형 비교군이 아닙니다. 이 결과로 주장할 수 있는 범위는 “위치·진행 일관성 점수의 합성 시간 교란 탐지 효과”이며, 실제 정상 오경보 보정과 실제 공정 주석 검증이 남아 있습니다.\n\n'
    text+='![정상 오경보와 탐지율](docs/figures/E9S_operating_curves.svg)\n\n주 q99는 큰 점으로 표시했습니다. 곡선은 사전 고정 q90/q95/q97.5/q99/q99.5의 별도 보정 결과이며 평가 데이터를 보고 임계값을 선택하지 않았습니다. 두 축은 장면별 macro matched FAR/event입니다.\n\n'
    text+=table(['Stride / seed','P4 AUROC %','P4 event %','P4−P5 event pp','P4−P7 event pp','P4−P9 event pp'],[[f'{r["stride"]} / {r["seed"]}',pct(r['macro']['auroc'][4]),pct(r['macro']['event'][4])]+[f'{100*(r["macro"]["event"][4]-r["macro"]["event"][i]):+.2f}' for i in [5,7,9]] for r in allruns])
    text+='Stride 비교는 원본 시간 lag와 transition 분산·restart hazard를 맞춘 민감도 실험입니다. 관측 밀도와 정상 학습 통계는 여전히 달라집니다. seed 반복은 독립 데이터 반복이 아닙니다.\n\n'
    text+=table(['Speed stress (seed 42, stride 2)','P0 grid alarm %','P4 grid alarm %','P7 grid alarm %','P9 grid alarm %'],[[v]+[pct(np.mean([s['speed_stress'][sc][str(v)]['window_alarm'][i] for sc in SCENES])) for i in [0,4,7,9]] for v in [.8,1.,1.2]])
    text+='속도 변형은 실제 정상 허용 범위가 확인되지 않은 stress test입니다. 실행 중 Identity replay 최대 오차는 모든 분할에서 0이었습니다. 저장 모델을 다시 읽어 320개 편집을 재계산했을 때 최대 점수 차이는 9.93×10⁻⁶이었고, 검사한 탐지 시점은 모두 같았습니다. 최초 32프레임 경계 반응과 편집 내부 반응을 분리하여 저장했습니다. skip에는 지속되는 내부 구간 GT를 만들지 않았으며, 내부 구간이 없는 짧은 편집은 해당 지표에서 제외했습니다. 탐지 지연은 탐지된 사례에 조건부이므로 miss 비율(1−event)과 함께 해석해야 합니다. 유형·강도별 지연과 내부 구간 coverage는 각 요약에 있습니다.\n\n'
    frameop=np.mean([np.array(s['normal'][sc]['frameq99_event_matchedfar']) for sc in SCENES],axis=0)
    text+=table(['보조 frame-q99 운영점','P0','P4','P5','P7','P9'],[[label]+[pct(frameop[j,i]) for i in [0,4,5,7,9]] for j,label in enumerate(['Event hit %','Matched normal FAR %'])])
    text+='보조 frame-q99는 주 window-q99와 별도이며 서로 같은 오경보 예산으로 해석하지 않습니다.\n\n'
    text+='[Seed 42 상세](results/E9S/stride2/seed42/summary.json) · [Seed 43](results/E9S/stride2/seed43/summary.json) · [Seed 44](results/E9S/stride2/seed44/summary.json) · [Stride 1](results/E9S/stride1/seed42/summary.json) · [paired CI](results/E9S/stride2/seed42/paired_bootstrap.json) · [아티팩트 재검증](results/E9S/validation.txt)\n\n'
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(10)
    axs[0].bar(x-.18,np.array(s['macro']['auroc'])*100,.36,label='Window AUROC',color='#176e72');axs[0].bar(x+.18,np.array(s['macro']['event'])*100,.36,label='Event hit',color='#d38956');axs[0].set_xticks(x,s['arms']);axs[0].set_ylim(0,100);axs[0].set_ylabel('%');axs[0].set_title('Primary operating point: seed 42, stride 2');axs[0].legend(frameon=False);axs[0].grid(axis='y',alpha=.2);axs[0].set_axisbelow(True)
    edits=['freeze','reverse','skip','swap_adjacent_blocks'];selected=[0,4,5,7,9];mat=np.array([[np.mean([r['event'][a] for r in s['groups'] if r['edit']==ed])*100 for ed in edits] for a in selected])
    im=axs[1].imshow(mat,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');axs[1].set_xticks(range(4),['Freeze','Reverse','Skip','Swap']);axs[1].set_yticks(range(5),[f'P{i}' for i in selected]);axs[1].set_title('Event hit by edit type (%)')
    for y in range(5):
        for j in range(4):axs[1].text(j,y,f'{mat[y,j]:.1f}',ha='center',va='center',color='white' if mat[y,j]>55 else '#142c38')
    fig.colorbar(im,ax=axs[1],shrink=.8);fig.tight_layout();save(fig,'E9S_detection')
    fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
    xmax=5*np.ceil(max(np.array(s['normal'][sc]['operating_curves_event_matchedfar'])[1,:, [0,4,5,7,9]].max()*100 for sc in SCENES)/5)+1
    for ax,scene in zip(axes.flat,SCENES):
        curve=np.array(s['normal'][scene]['operating_curves_event_matchedfar'])*100
        for a in [0,4,5,7,9]:
            ax.plot(curve[1,:,a],curve[0,:,a],'-o',ms=3,color=COLORS[a],label=f'P{a}');ax.scatter(curve[1,3,a],curve[0,3,a],s=70,color=COLORS[a],edgecolors='white',zorder=4)
        ax.axvline(5,ls='--',color='#89949a',lw=1);ax.set_title(scene);ax.set_xlim(0,xmax);ax.set_ylim(0,100);ax.grid(alpha=.15);ax.set_xlabel('Matched normal window alarm (%)');ax.set_ylabel('Event hit (%)')
    axes[0,0].legend(frameon=False,ncol=3);fig.suptitle('Fixed normal-calibration quantiles; large markers = primary q99');fig.tight_layout();save(fig,'E9S_operating_curves')
    return text

def main():
    p=argparse.ArgumentParser();p.add_argument('--include-e9',action='store_true');a=p.parse_args()
    text='## 위치·진행 검증 E8B / E9S\n\n[고정 실험 명세](docs/PHASE_PROCESS_EXPERIMENTS.md) · [35개 코드 검사](results/phase_process_tests.txt)\n\n'+e8()
    if a.include_e9:text+=e9()
    else:text+='E9S는 전체 seed·stride 실행과 결과 검증을 진행 중입니다. 완료 수치와 오경보 분석은 다음 단계 커밋으로 공개합니다.\n\n'
    text+='재실행: `PYTHONPATH=src python scripts/run_phase_process.py --stage E8B --resume`, `--stage E9S --resume`, `--stage E9S --stride 1 --seeds 42 --resume`. 이어 `scripts/summarize_phase_process.py --stage E8B` 및 E9S의 각 `--seed`/`--stride`를 실행합니다. 주 E9S는 `--bootstrap`을 붙입니다. `scripts/check_phase_process.py --stage E8B`/`E9S`로 검증하고 `scripts/report_phase_process.py --include-e9`로 문서를 재생성합니다. 특징 추출은 로컬 RTX PRO 6000의 고정 캐시를 재사용했고 새 통계 모델은 CPU에서 학습했습니다.\n\n'
    (ROOT/'docs/PHASE_PROCESS_RESULTS.md').write_text(text.replace('](docs/', '](').replace('](results/', '](../results/'))
    path=ROOT/'README.md';old=path.read_text();start=old.find('## 위치·진행 검증 E8B / E9S');end=old.index('## 실험 상태')
    if start<0:start=end
    extra=''
    if (ROOT/'docs/PAPER_STAGE1_RESULTS.md').exists():
        extra=(ROOT/'docs/PAPER_STAGE1_RESULTS.md').read_text().replace('](../results/', '](results/').replace('](../configs/', '](configs/').replace('](figures/', '](docs/figures/').replace('](PAPER_STAGE1_PROTOCOL.md)', '](docs/PAPER_STAGE1_PROTOCOL.md)')
    from report_fragments import diagnostics_fragment
    old=old[:start]+text+extra+diagnostics_fragment(ROOT)+old[end:]
    old=old.replace('모델 실험은 아직 미실행입니다.','실행 결과는 아래 E8B / E9S 절에 정리했습니다.').replace('E9S: 합성 편집 검증은 E7 주석과 독립적으로 실행할 계획입니다.','E9S: 합성 편집 검증은 E7 주석과 독립적으로 진행하며 아래 실행 결과를 따릅니다.')
    old=old.replace('## 주요 관찰','## 초기 실험 E0–E5 관찰')
    path.write_text(old)
    status=read('results/followup_status.json');status.update(E8B='complete',E9S='complete' if a.include_e9 else 'running',E9='synthetic_complete_real_annotations_required' if a.include_e9 else 'synthetic_running');(ROOT/'results/followup_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('Reports and figures generated:', 'E8B + E9S' if a.include_e9 else 'E8B')
if __name__=='__main__':main()
