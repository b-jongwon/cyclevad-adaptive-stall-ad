"""Publish E15 calibration experiments with inspectable tables and figures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycle_vad.paper_calibration import SCENES,POLICIES,read,deserialize_thresholds
from cycle_vad.data import write_json
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cyclevad-calibration','figure.facecolor':'white'})
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def save(fig,name):
    fig.savefig(ROOT/f'docs/figures/{name}.svg',bbox_inches='tight',metadata={'Date':None})
    fig.savefig(ROOT/f'docs/figures/{name}.png',bbox_inches='tight',dpi=140);plt.close(fig)
def pct(x):return f'{100*x:.2f}'
def ci(v):return f'{v["estimate"]:+.2f} [{v["ci95"][0]:+.2f}, {v["ci95"][1]:+.2f}]'
def b0():
    result=[];lost=[];allruns=[]
    for seed in [42,43,44]:
        paths=sorted((ROOT/f'results/E15B0/seed{seed}').glob('R*/fold*/run.json'));assert len(paths)==20
        allruns += [read(p) for p in paths]
        for sc in SCENES:
            subset=[p for p in paths if p.parent.parent.name==sc];normal=[];events=[]
            for p in subset:normal+=read(p.with_name('normal.json'));events+=read(p.with_name('events.json'))
            assert len({v['id'] for v in normal})==len(normal)
            hit=np.array([v['hit'] for v in events])[:,0,:].astype(bool)
            item={'seed':seed,'scene':sc,'events':len(events),'detected':hit.sum(0).tolist(),
                  'added_over_C':(hit&~hit[:,0,None]).sum(0).tolist(),'lost_from_C':(~hit&hit[:,0,None]).sum(0).tolist(),
                  'normal_sources':len(normal),'grid_far':np.mean([v['grid_far'] for v in normal],axis=0).tolist(),
                  'overlap':{k:float(np.mean([v['overlap_q99'][k] for v in normal])) for k in normal[0]['overlap_q99']}}
            result.append(item)
            if seed==42:
                for e,h in zip(events,hit):
                    if h[0] and not h[1]:lost.append({'scene':sc,**e})
    collapsed=sum(all(j['q95_equals_q995'] for j in r['ecdf']) for r in allruns)
    write_json(ROOT/'results/E15B0/summary.json',{'policies':POLICIES,'rows':result,'q95_q995_all_three_equal_units':collapsed,'units':len(allruns),'seed42_lost_events':lost})
    r=[v for v in result if v['seed']==42];total=np.sum([v['detected'] for v in r],axis=0);far=np.mean([v['grid_far'] for v in r],axis=0)[0]
    text='### E15B0 경보 손실·보정 해상도 진단 완료\n\n고정 E11B 점수로 60개 단위를 검사했습니다. 아래는 seed 42, α=1%입니다. ORfree는 C/P 각자의 q99 임계값을 유지하는 진단이며 전체 1% 예산 방법이 아닙니다.\n\n'
    text+=table(['정책','실제 이벤트 / 66','C 대비 추가','C 대비 손실','정상 OOF grid FAR %'],[[POLICIES[j],int(total[j]),sum(v['added_over_C'][j] for v in r),sum(v['lost_from_C'][j] for v in r),pct(far[j])] for j in [0,1,4]])
    text+=table(['장면','C / CP / ORfree 탐지','C grid FAR %','ORfree grid FAR %','P만 추가한 정상 창 %'],[[v['scene'],' / '.join(str(v['detected'][j]) for j in [0,1,4]),pct(v['grid_far'][0][0]),pct(v['grid_far'][0][4]),pct(v['overlap']['P_only'])] for v in r])
    text+=f'원 임계값 OR는 C의 탐지를 보존하지만 추가 정상 경보를 발생시킵니다. 60개 단위 중 **{collapsed}개**에서 C/P/CP 모두 q95=q99.5입니다. 임계값 보정 영상은 3–4개, 겹치는 창은 21–33개이므로 높은 분위수의 구분 능력이 제한됩니다. 창 수를 독립 표본 수로 해석하지 않습니다. 원본 하나 제외 임계값과 ECDF jump도 각 run에 공개했습니다.\n\n'
    text+='![경보와 정상 비용](docs/figures/E15B0_tradeoff.svg)\n\n![C가 탐지했으나 CP가 놓친 7개 이벤트](docs/figures/E15B0_lost_events.svg)\n\n회색 영역은 이벤트 onset부터 탐지 deadline까지입니다. 패널별 점수 범위는 다릅니다.\n\n[원본 수치](results/E15B0/summary.json) · [재현 검증](results/E15B0/validation.txt)\n\n'
    fig,ax=plt.subplots(1,2,figsize=(10.5,4));names=['C','joint max','ORfree'];js=[0,1,4];colors=['#3873a1','#e09b38','#555555']
    ax[0].bar(names,total[js],color=colors);ax[0].set_ylim(0,66);ax[0].set_ylabel('Detected historical events / 66')
    ax[1].bar(names,100*far[js],color=colors);ax[1].set_ylabel('Held-out normal grid-window FAR (%)');ax[1].set_ylim(0,max(15,100*far[js].max()*1.2))
    for a,vs in zip(ax,[total[js],100*far[js]]):
        for k,v in enumerate(vs):a.text(k,v+.5,f'{v:.2f}' if a==ax[1] else str(int(v)),ha='center')
    fig.suptitle('Original branch thresholds: event preservation and normal alarm cost');fig.tight_layout();save(fig,'E15B0_tradeoff')
    fig,axes=plt.subplots(4,2,figsize=(12,12));axes=axes.ravel()
    for ax,e in zip(axes,lost):
        _,part,seq=e['source_id'].split('/');rel=Path('seed42')/e['scene']/'fold0';r0=read(ROOT/'results/E15B0'/rel/'run.json');th=deserialize_thresholds(r0['thresholds'])
        with np.load(ROOT/'runs/E11B'/rel/f'{part}_{seq}.npz') as d:s=d['scores'][:,[4,5]]
        lo=max(0,e['onset']-20);hi=min(len(s),e['onset']+e['deadline_frames']+20);xx=np.arange(lo,hi)-e['onset']
        ax.plot(xx,s[lo:hi,0],color='#3873a1',label='C');ax.plot(xx,s[lo:hi,1],color='#e09b38',label='P')
        ax.axhline(th[0,0,0],ls='--',color='#3873a1',label='C threshold');ax.axhline(th[0,1,0],ls=':',color='#555',label='joint threshold')
        ax.axvspan(0,e['deadline_frames'],alpha=.08,color='#555');ax.set_title(e['id']);ax.set_xlabel('Source frames from event onset');ax.set_ylabel('Calibrated score')
    for ax in axes[len(lost):]:ax.axis('off')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',frameon=False);fig.tight_layout(rect=(0,.035,1,1));save(fig,'E15B0_lost_events')
    return text

def bc():
    summaries=[read(ROOT/f'results/E15BC/seed{s}/summary.json') for s in [42,43,44]];s=summaries[0]
    boot=read(ROOT/'results/E15BC/seed42/bootstrap.json');runs=[read(p) for p in (ROOT/'results/E15BC').glob('seed*/R*/fold*/run.json')]
    counts=np.array(s['historical_total_counts']['historical_event']);added=np.array(s['historical_total_counts']['added']);lost=np.array(s['historical_total_counts']['lost'])
    m={k:np.array(v) for k,v in s['macro'].items()}
    text='### E15B/C 정상 보정·결합 정책 비교 완료\n\nR01–R04×5-fold×3-seed의 60개 단위, 고정 편집 3,910개를 seed별로 재생하여 총 11,730회 평가했습니다. seed 반복을 독립 편집 표본으로 세지 않습니다. G0는 기존 보정, G1은 reference 영상별 동일 가중 보정입니다. 모델·raw score·tail 외삽은 고정했습니다. 8개 주 조건과 ORfree 진단 2개를 각각 α=1%,5%,10%에서 평가했습니다. 아래는 seed 42, 명목 α=1%입니다. **동일 명목 예산은 동일 실제 FAR을 뜻하지 않습니다.**\n\n'
    rows=[]
    for g in range(2):
        for p,name in enumerate(POLICIES):
            rows.append([f'G{g}',name,int(counts[g,0,p]),f'{added[g,0,p]} / {lost[g,0,p]}']+[pct(m[k][g,0,p]) for k in ['grid_far','synthetic_event','matched_far']])
    text+=table(['보정','정책','실제 이벤트 / 66','동일 G의 C 대비 추가 / 손실','정상 grid FAR %','합성 event %','Matched FAR %'],rows)
    collapsed=np.array([[sum(r['thresholds'][g][a][2]==r['thresholds'][g][a][3]==r['thresholds'][g][a][4] for r in runs) for a in range(3)] for g in range(2)])
    text+='**주 운영점에서 OR75=OR50=ORfree가 모든 60개 실행, 두 보정 방식에서 동일했습니다.** 높은 분위수가 같은 관측값에 걸렸기 때문입니다. 75:25나 50:50 예산 배분의 우수성을 보여준 결과가 아닙니다. OR는 3개 실제 이벤트를 추가하고 손실 7개를 복구하지만, C 대비 정상 grid FAR 증가를 동반합니다.\n\n'
    text+=table(['보정','α=1%','α=5%','α=10%'],[[f'G{g}']+[f'{int(x)}/60' for x in collapsed[g]] for g in range(2)])
    text+='위 표는 OR75·OR50·ORfree의 **임계값 쌍이 모두 같은 실행 수**입니다. 중첩 창을 독립 표본으로 세지 않습니다.\n\n'
    rows=[]
    for g in range(2):
        for p in [1,2]:
            row=[f'G{g}/{POLICIES[p]} − G{g}/C']
            for metric in ['historical_event','grid_far','synthetic_event']:
                c=next(v for v in boot['comparisons'] if (v['g'],v['alpha'],v['policy'],v['reference'],v['metric'])==(g,.01,POLICIES[p],'same_G_C',metric))
                row.append(ci(c['macro']))
            rows.append(row)
    text+=table(['비교','실제 event recall Δ pp [95% CI]','Grid FAR Δ pp [95% CI]','합성 event Δ pp [95% CI]'],rows)
    text+='CI는 seed 42 고정 모델에 조건부인 원본 영상 paired bootstrap 2,000회입니다. 실제 recall은 장면별 비율의 macro이며 66개 이벤트를 합친 micro와 다릅니다. 양성 이벤트가 없는 bootstrap 표본 1개는 macro recall이 정의되지 않아 제외되어 유효 1,999회입니다. FAR와 합성 지표는 유효 2,000회입니다. 유지·추가·손실의 비율 CI, 모든 장면 제외 분석도 공개했습니다.\n\n'
    text+=table(['Seed','G0 C / MAX / OR75 이벤트','G1 C / MAX / OR75 이벤트','G0 OR75 grid FAR %','G1 OR75 grid FAR %'],[[v['seed'],' / '.join(str(np.array(v['historical_total_counts']['historical_event'])[0,0,p]) for p in [0,1,2]),' / '.join(str(np.array(v['historical_total_counts']['historical_event'])[1,0,p]) for p in [0,1,2]),pct(np.array(v['macro']['grid_far'])[0,0,2]),pct(np.array(v['macro']['grid_far'])[1,0,2])] for v in summaries])
    text+='![명목 예산별 합성 탐지와 정상 경보](docs/figures/E15BC_operating.svg)\n\n원은 α=1%, 나머지 두 점은 사전 지정한 5%,10%입니다. 선은 세 운영점을 연결한 것이며 연속 ROC나 평가 정상 라벨로 FAR을 일치시킨 곡선이 아닙니다. 중복 점은 겹쳐 보입니다.\n\n'
    text+='G1은 G0 대비 실사 AP 변화가 매우 작았습니다. Seed 42에서 C '+pct(s['historical_macro_ranking']['G0_C']['ap'])+'→'+pct(s['historical_macro_ranking']['G1_C']['ap'])+'%, MAX '+pct(s['historical_macro_ranking']['G0_MAX']['ap'])+'→'+pct(s['historical_macro_ranking']['G1_MAX']['ap'])+'%였습니다. 이는 통계적 우월성 주장이 아니며, 실제 탐지 이벤트 수는 같고 정상 grid FAR은 감소하지 않았습니다. OR 정책에는 인위적인 순위 점수를 부여하지 않았습니다.\n\n'
    text+=table(['장면','G0 C 정상 frame FPR %','G1 C 정상 frame FPR %','G0 MAX 정상 frame FPR %','G1 MAX 정상 frame FPR %'],[[sc]+[pct(np.array(s['by_scene'][sc]['metrics']['historical_normal_alarm'])[g,0,p]) for g,p in [(0,0),(1,0),(0,1),(1,1)]] for sc in SCENES])
    text+='이 표는 **historical 라벨 0 프레임**에서 각 정책의 자체 window 임계값을 적용한 결과입니다. OOF grid FAR 및 E15A의 Full 공통 임계값 component 초과율과 다릅니다. R01은 G1에서도 개선되지 않았습니다. 보정 reference의 영상 길이 편향만으로 분포 차이를 설명할 수 없습니다. 실제 원인 판정에는 원본·라벨 검토가 필요합니다.\n\n'
    text+='판정: **정상 오경보를 통제하면서 C의 탐지를 보존하고 P의 이득을 추가한다는 목표는 미달성입니다.** 이번 비교에서 주 방법을 교체하지 않습니다. 낮은 FAR의 세밀한 예산 탐색은 중단하고 독립 정상 보정 자료, R01 검토와 E7B 실제 단계 주석을 우선합니다. P의 합성 시간 교란 탐지 효과와 실제 경보의 비용을 구분해 논문에 제시합니다.\n\n'
    text+='[전체 운영점·장면별 수치](results/E15BC/seed42/summary.json) · [원본 paired CI](results/E15BC/seed42/bootstrap.json) · [전체 검증](results/E15BC/validation.txt) · [실행 기록](results/paper_calibration_execution.json)\n\n'
    text+='로컬 RTX PRO 6000에서 생성된 고정 CUDA 특징을 재사용했습니다. 이번 통계 재생은 CPU에서 수행했고, 60개 단위 실행 시간 합계는 '+f'{sum(r["wall_seconds"] for r in runs):.1f}'+'초입니다. 이는 특징 추출·집계·검증 시간을 제외하므로 end-to-end 실시간 성능이 아닙니다. 독립 인간 주석과 신규 촬영, 외부 baseline은 미완료입니다.\n\n'
    fig,axes=plt.subplots(1,2,figsize=(11,4.6));colors=['#3873a1','#db942e','#b85a38','#555555'];styles=['-','--',':','-.']
    for g,ax in enumerate(axes):
        for p in range(4):
            x=100*m['matched_far'][g,:,p];y=100*m['synthetic_event'][g,:,p]
            ax.plot(x,y,styles[p],color=colors[p],label=POLICIES[p],marker='s',markersize=4)
            ax.scatter(x[0],y[0],s=70,facecolor='white',edgecolor=colors[p],linewidth=1.5,zorder=4)
        ax.set_title(['G0: pooled reference','G1: video-equal reference'][g]);ax.set_xlabel('Matched normal window FAR (%)');ax.set_ylabel('Synthetic event detection (%)');ax.set_xlim(0,30);ax.set_ylim(0,100);ax.grid(alpha=.15);ax.legend(frameon=False)
    fig.tight_layout();save(fig,'E15BC_operating')
    write_json(ROOT/'results/paper_calibration_execution.json',{'status':'complete','units':60,'edited_cases':11730,'primary_conditions':8,'diagnostic_conditions':2,'nominal_alphas':[.01,.05,.1],'unit_wall_seconds_sum':sum(r['wall_seconds'] for r in runs),'feature_hardware':'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition','scoring_device':'CPU; frozen CUDA feature cache reused','timing_excludes':['feature extraction','summary bootstrap','validation','reporting'],'baseline_commit':'4e5808bd14bf4df8f73dd5807e8ac03e46671047','B0_commit':'eed4e1b8b061c071dca2bfebe467e990a8d5c34e'})
    packet=read(ROOT/'results/E15BC/r01_audit_packet.json')
    text+=f'R01 원본 검토용으로 점수·모델 출력·선정 이유를 숨긴 로컬 문맥 구간 {packet["clips"]}개를 준비했습니다(균등 {packet["uniform_clips"]}, 점수 기반 탐색 {packet["score_enriched_clips"]}). 두 묶음은 발생률 추정에 합치지 않습니다. **독립 인간 검토 완료는 0개**이며, 이 자료로 원인을 확정하지 않았습니다. [검토 안내](docs/R01_AUDIT_GUIDE.md) · [준비 상태](results/E15BC/r01_audit_packet.json).\n\n'
    return text

def publish(text):
    title='## 보정과 경보 결합 E15B0·E15B/C\n\n'
    header='[사전 설계](docs/PAPER_NEXT_EXPERIMENTS_20261009.md) · [고정 설정](configs/experiments/followup/e15bc_v1.json) · [코드 검사](results/paper_calibration_tests.txt)\n\n'
    block=title+header+text
    (ROOT/'docs/PAPER_CALIBRATION_RESULTS.md').write_text(block.replace('](docs/','](').replace('](results/','](../results/').replace('](configs/','](../configs/'))
    p=ROOT/'README.md';old=p.read_text();start=old.find(title)
    if start>=0:
        end=old.find('\n## ',start+len(title));end=len(old) if end<0 else end
        old=old[:start]+old[end+1:]
    position=old.index('## 최소 구성과 오경보 진단');p.write_text(old[:position]+block+old[position:])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--include-bc',action='store_true');a=p.parse_args();text=b0()
    if a.include_bc:text+=bc()
    else:text+='E15B/C는 실행·검증 중입니다. 아직 결합 개선 결과를 확정하지 않았습니다.\n\n'
    publish(text)
