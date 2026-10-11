"""Publish checked E11B/E15A tables and scientific figures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SCENES=['R01','R02','R03','R04'];LABELS=['A','A+C','A+P','Full','C','P','C+P']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cyclevad-diagnostics','figure.facecolor':'white'})
def read(p):return json.loads((ROOT/p).read_text())
def pct(x):return '—' if x is None else f'{100*x:.2f}'
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def ci(v):return f'{v["estimate"]:+.2f} [{v["ci95"][0]:+.2f}, {v["ci95"][1]:+.2f}]'
def save(fig,name):
    fig.savefig(ROOT/f'docs/figures/{name}.svg',bbox_inches='tight',metadata={'Date':None});fig.savefig(ROOT/f'docs/figures/{name}.png',bbox_inches='tight',dpi=150);plt.close(fig)
def e11b():
    hs=[read(f'results/E11B/seed{s}/historical_summary.json') for s in [42,43,44]];ss=[read(f'results/E11B/seed{s}/synthetic_summary.json') for s in [42,43,44]]
    h,s=hs[0],ss[0];hb=read('results/E11B/seed42/historical_bootstrap.json');sb=read('results/E11B/seed42/synthetic_bootstrap.json')
    text='### E11B 최소 구성 비교 완료\n\nE11의 고정 checkpoint와 원래 여섯 분기를 유지하고 CP=max(C,P)를 추가했습니다. R01–R04×5-fold×3-seed 60개 단위, 합성 11,730개 편집을 재계산했습니다. 새 CP 임계값은 정상 threshold 영상에서만 구했습니다. 아래는 seed 42입니다.\n\n'
    text+=table(['구성','Historical AP %','AUROC %','합성 event %','Matched 정상 FAR %','Grid 정상 FAR %'],[[LABELS[j],pct(h['macro']['ap'][j]),pct(h['macro']['auroc'][j])]+[pct(s['macro'][k][j]) for k in ['event','matched_far','grid_far']] for j in [0,4,5,1,2,6,3]])
    text+='AP는 historical 프레임 순위 성능이며, 합성 event는 기존 D=0.2T deadline의 탐지율입니다. T는 FIT recording 길이 척도입니다. 동일 q99 보정이 동일 실제 FAR을 보장하지 않습니다. 기존 데이터에서의 탐색적 비교이며 독립 검증이 아닙니다.\n\n'
    rows=[]
    for contrast in ['CP-C','B3-CP','B3-C']:
        a=next(v for v in hb['contrasts'] if v['contrast']==contrast and v['metric']=='ap_gain_pp');b=next(v for v in sb['contrasts'] if v['contrast']==contrast and v['metric']=='event_gain_pp')
        rows.append([contrast,ci(a['macro']),ci(a['leave_one_scene_out']['R02']),ci(b['macro'])])
    text+=table(['비교','AP 차이 pp [95% CI]','R02 제외 AP 차이 pp [95% CI]','합성 event 차이 pp [95% CI]'],rows)
    text+='CP−C와 Full−CP가 사전 주 비교이고 Full−C는 보조 비교입니다. CI는 seed 42 고정 모델의 원본 영상 paired bootstrap 2,000회입니다. 네 가지 장면 제외 분석은 민감도 분석이며 미관측 장면 일반화가 아닙니다.\n\n'
    text+=table(['장면','C AP %','CP AP %','Full AP %','C 이벤트','CP 이벤트','Full 이벤트','CP의 C 대비 추가/손실','Full의 CP 대비 추가/손실'],[[sc]+[pct(h['by_scene'][sc][j]['ap']) for j in [4,6,3]]+[f'{r["detected"][j]}/{r["events"]}' for j in [4,6,3]]+[f'{r["CP_added_over_C"]}/{r["CP_lost_from_C"]}',f'{r["Full_added_over_CP"]}/{r["Full_lost_from_CP"]}'] for sc,r in zip(SCENES,h['events'])])
    text+='이벤트 수는 연속 양성 구간이며 실제 이상 유형은 아직 미주석입니다. 서로 다른 정상 임계값 때문에 CP도 C의 경보를 잃을 수 있습니다. 새 임계값의 합성 지연·경보 프레임 수는 기존 지연 요약에서 추정하지 않고 재생성한 전체 W-window 점수로 계산했습니다.\n\n'
    text+=table(['Seed','C AP %','CP AP %','Full AP %','CP 합성 event %','CP matched FAR %'],[[v['seed']]+[pct(v['macro']['ap'][j]) for j in [4,6,3]]+[pct(w['macro'][k][6]) for k in ['event','matched_far']] for v,w in zip(hs,ss)])
    text+='![최소 구성의 효과와 운영점](docs/figures/E11B_minimal.svg)\n\n'
    text+='Seed 42에서 C는 33/66개, CP와 Full은 각각 29/66개 이벤트를 탐지했습니다. CP는 C 대비 3개를 추가하고 7개를 잃었으며, Full은 CP 대비 이벤트 추가·손실이 없었습니다. CP−C의 평균 AP 개선 CI는 0을 포함하고 R02를 제외한 점추정은 음수입니다. 따라서 진행 신호의 광범위한 상보성을 확정할 수 없습니다. A의 추가 필요성도 현 결과로 강하게 지지되지 않지만, Full−CP의 전체 장면 AP CI가 0을 포함하므로 CP의 통계적 우월성을 확정하지 않습니다.\n\n'
    text+='[검증 로그](results/E11B/validation.txt) · [Historical 결과와 장면 제외 분석](results/E11B/seed42/historical_summary.json) · [paired CI](results/E11B/seed42/historical_bootstrap.json) · [합성 운영 곡선](results/E11B/seed42/synthetic_summary.json) · [합성 추가·손실과 경보 지속 비율](results/E11B/seed42/synthetic_overlap.json)\n\n'
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4));x=np.arange(4)
    for a,color in [(4,'#4978a8'),(6,'#167773'),(3,'#b35c4a')]:
        axes[0].plot(x,[100*h['by_scene'][sc][a]['ap'] for sc in SCENES],'-o',label=LABELS[a],color=color)
        curve=np.mean([np.array(s['normal'][sc]['operating_curves_event_matchedfar']) for sc in SCENES],axis=0)*100
        axes[1].plot(curve[1,:,a],curve[0,:,a],'-o',ms=3,label=LABELS[a],color=color);axes[1].scatter(curve[1,3,a],curve[0,3,a],s=70,color=color)
    axes[0].set_xticks(x,SCENES);axes[0].set_ylabel('Historical frame AP (%)');axes[0].set_title('Conditional appearance, process and full model');axes[0].legend(frameon=False)
    axes[1].set_xlabel('Matched normal window alarm (%)');axes[1].set_ylabel('Synthetic event hit (%)');axes[1].set_title('Fixed calibration quantiles; large marker = q99');axes[1].set_ylim(0,100);axes[1].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.15)
    fig.tight_layout();save(fig,'E11B_minimal');return text

def e15a():
    summary=read('results/E15A/summary.json');groups=summary['groups'];distributions=read('results/E15A/distributions.json')['distributions']
    def rec(sc,scope,part,name='normal'):return next((v for v in groups if (v['seed'],v['scene'],v['scope'],v['partition'],v['stratum'])==(42,sc,scope,part,name)),None)
    text='### E15A 고정 모델 오경보 진단 완료\n\n같은 60개 단위에서 8개 raw/component 점수를 다시 계산했습니다. 아래는 seed 42이며, 정상 OOF는 5개 fold에서 원본마다 한 번, historical은 fold 0 모델의 라벨 0 프레임입니다. **보정이나 임계값을 개선한 실험은 아직 아닙니다.**\n\n'
    rows=[]
    for sc in SCENES:
        o=rec(sc,'normal_oof','normal_evaluation');h=rec(sc,'fold0','historical_test');win=h['branch_winner_given_full_alarm'] or [None]*3
        rows.append([sc,o['videos'],pct(o['full_fpr_frame_weighted']),h['videos'],pct(h['full_fpr_frame_weighted'])]+[pct(v) for v in win])
    text+=table(['장면','OOF 영상','OOF 정상 frame FPR %','Historical 정상 포함 영상','Historical 정상 frame FPR %','경보 중 A winner %','C winner %','P winner %'],rows)
    text+='Winner는 경보 프레임에서 가장 큰 분기 점수의 비율이며 동률은 균등 배분했습니다. 경보가 없는 R02의 조건부 winner는 정의되지 않습니다. 이는 상관된 분기 사이의 인과적 책임 또는 해당 분기를 제거한 효과가 아닙니다. 실제 제거 효과는 E11B로 확인합니다. Component의 자체 q99 초과율과 Full의 공통 임계값 초과율도 별도로 공개합니다.\n\n'
    text+='R01 historical 정상 프레임에서 A outside와 C outside의 Full 임계값 초과율은 각각 45.29%, 45.62%입니다. P가 winner인 비율은 해당 정상 경보의 1.79%입니다. 따라서 P만 제거하거나 정지 탐지만 추가하는 것으로 R01의 문제를 해결했다고 볼 수 없습니다. 분포 차이는 확인되었지만 물리적 원인은 원본 영상 검토가 필요합니다.\n\n'
    rows=[]
    for sc in SCENES:
        for name,label in [('normal_no_prior_anomaly','첫 양성 이전/양성 없음'),('normal_after_first_anomaly','첫 양성 이후 정상')]:
            r=rec(sc,'fold0','historical_test',name)
            rows.append([sc,label,r['videos'] if r else 0,r['frames'] if r else 0,pct(r['full_fpr_frame_weighted']) if r else '해당 없음'])
    text+=table(['장면','정상 구간','영상 수','정상 프레임','Full frame FPR %'],rows)
    text+='이 구분은 관측된 binary 라벨에 따른 진단입니다. 원인을 조명·배경이나 tracker 잔류로 확정하지 않습니다. 미확인 라벨 구간은 정상으로 바꾸지 않았고, 동일 영상이 두 정상 구간에 포함될 수 있습니다.\n\n'
    text+='![분할별 정상 점수 분포](docs/figures/E15A_distributions.svg)\n\n곡선은 fold 0에서 각 원본 영상의 가중치를 같게 둔 Full 정상 점수의 분위수입니다. 정상 window-q99 임계값과 프레임 점수 분포는 다른 통계량입니다. Historical 정상 프레임의 높은 값이 사건 이전에도 나타나는지 함께 확인해야 합니다.\n\n'
    rows=[]
    for sc in SCENES:
        row=[sc]
        for name in ['normal_startup','normal_steady']+[f'normal_phase{k}' for k in range(4)]:
            r=rec(sc,'normal_oof','normal_evaluation',name);row.append(pct(r['full_fpr_frame_weighted']) if r else '해당 없음')
        rows.append(row)
    text+=table(['장면','초기32 FPR %','이후 FPR %','phase0','phase1','phase2','phase3'],rows)
    text+='Phase는 모델의 추정 좌표이며 실제 단계 정답이 아닙니다. 빈 bin은 실패 0%로 대체하지 않습니다. 초기/phase별 차이도 서로 다른 조건부 표본에 대한 기술 통계입니다.\n\n'
    text+='![공통 Full 임계값의 component 초과율](docs/figures/E15A_components.svg)\n\n이 그림의 각 칸은 해당 component가 Full 임계값을 넘은 정상 프레임 비율입니다. 여러 component가 동시에 넘을 수 있어 합계는 Full FPR과 같지 않습니다. Exclusive exceedance와 tied winner를 원 수치에 따로 기록했습니다.\n\n'
    text+='[검증 로그](results/E15A/validation.txt) · [원본별 집계와 전체 strata](results/E15A/summary.json) · [raw/calibrated 분위수](results/E15A/distributions.json)\n\n'
    fig,axes=plt.subplots(2,2,figsize=(11.5,7));parts=['reference','threshold','normal_evaluation','historical_test'];names=['Reference','Threshold','Normal OOF (fold0)','Historical normal']
    for ax,sc in zip(axes.flat,SCENES):
        for part,label in zip(parts,names):
            d=next(v for v in distributions if v['seed']==42 and v['scene']==sc and v['partition']==part)
            ax.plot(100*np.array(d['quantile_levels']),np.log1p(d['branch_video_equal_quantiles'][3]),label=label)
        threshold=read(f'results/E11B/seed42/{sc}/fold0/run.json')['window_thresholds'][3][3]
        ax.axhline(np.log1p(threshold),ls='--',color='#555',lw=1,label='Full window-q99 threshold');ax.set_title(sc);ax.set_xlabel('Video-equal normal quantile (%)');ax.set_ylabel('log(1 + Full score)');ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=(0,.1,1,1));save(fig,'E15A_distributions')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));values=[]
    for scope,part in [('normal_oof','normal_evaluation'),('fold0','historical_test')]:values.append(100*np.array([rec(sc,scope,part)['full_threshold_exceed_frame_weighted'] for sc in SCENES]))
    vmax=max(v.max() for v in values)
    for ax,v,title in zip(axes,values,['Held-out normal, all 5 folds','Historical normal frames, fold0']):
        im=ax.imshow(v,aspect='auto',vmin=0,vmax=vmax,cmap='YlOrRd');ax.set_xticks(range(8),['A out','A in','A local','C out','C in','P align','P innov','P prog'],rotation=45,ha='right');ax.set_yticks(range(4),SCENES);ax.set_title(title)
        for i in range(4):
            for j in range(8):ax.text(j,i,f'{v[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if v[i,j]>vmax*.6 else 'black')
    fig.tight_layout(rect=(0,0,.93,1));cax=fig.add_axes([.94,.3,.015,.5]);fig.colorbar(im,cax=cax,label='Full-threshold exceedance (%)');save(fig,'E15A_components');return text

def main():
    p=argparse.ArgumentParser();p.add_argument('--include-e15a',action='store_true');a=p.parse_args()
    text='## 최소 구성과 오경보 진단 E11B·E15A\n\n[후속 설계](docs/PAPER_STAGE2_DESIGN.md) · [고정 실행 설정](configs/experiments/followup/e11b_e15a_v1.json) · [코드 검사](results/paper_diagnostics_tests.txt)\n\n'+e11b()
    if a.include_e15a:text+=e15a()
    else:text+='E15A는 결과 검증·집계 중입니다.\n\n'
    text+='로컬 RTX PRO 6000의 고정 CUDA 특징을 재사용했고 통계 점수 재계산은 CPU에서 수행했습니다. 이 단계에는 학습·보정 변경이 없습니다. E15B/C의 후속 결과는 [최신 보정 실험](docs/PAPER_CALIBRATION_RESULTS.md)을 참고하세요. E13B, E14B, E12B, E7B 독립 주석, 외부 baseline, 카메라·코덱까지 포함한 streaming 및 독립 촬영 검증은 미완료입니다.\n\n'
    doc=text.replace('](docs/','](').replace('](results/','](../results/').replace('](configs/','](../configs/');(ROOT/'docs/PAPER_DIAGNOSTICS_RESULTS.md').write_text(doc)
    p=ROOT/'README.md';old=p.read_text();end=old.index('## 실험 상태');start=old.find('## 최소 구성과 오경보 진단 E11B·E15A');start=end if start<0 else start;p.write_text(old[:start]+text+old[end:])
    status=read('results/followup_status.json');status['E11B']='complete';status['E15A']='complete' if a.include_e15a else 'analysis_in_progress';status['E15']=status.get('E15') if (ROOT/'results/E15BC/validation.txt').exists() else ('diagnosis_complete_calibration_pending' if a.include_e15a else 'diagnosis_in_progress');(ROOT/'results/followup_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('Reported E11B'+(' + E15A' if a.include_e15a else ''))
if __name__=='__main__':main()
