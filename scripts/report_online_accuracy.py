"""Summarize full-video causal replay; pair source files, never resample frames."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycle_vad.data import write_json
from cycle_vad.metrics import frame_metrics
from cycle_vad.paper_diagnostics import sha
from bootstrap_followup import weighted_metrics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/E17';LOCAL=ROOT/'runs/E17'
ARMS=['A','C','P','CP','Full'];SCENES=['R01','R02','R03','R04'];KEYS=['cached','stride2','stride1']
def read(p):return json.loads(p.read_text())
def main():
    cfg=read(ROOT/'configs/experiments/followup/online_accuracy_v1.json')
    metadata=[read(OUT/'videos'/f'{s.replace("/","_")}.json') for s in cfg['sources']]
    assert len(metadata)==66 and sum(v['frames'] for v in metadata)==33462
    assert sum(v['valid_frames'] for v in metadata)==31550 and sum(v['unknown_frames'] for v in metadata)==1912
    arrays={m['source_id']:{k:v.copy() for k,v in np.load(LOCAL/f'{m["source_id"].replace("/","_")}.npz').items()} for m in metadata}
    for m in metadata:
        v=arrays[m['source_id']]
        assert sha(LOCAL/f'{m["source_id"].replace("/","_")}.npz')==m['scores_sha256']
        assert len(v['labels'])==m['frames']
        for key in KEYS:assert v[key].shape==(m['frames'],len(ARMS))
        np.testing.assert_array_equal(v['stride2'][1::2],v['stride2'][::2][:len(v['stride2'][1::2])])
        np.testing.assert_array_equal(v['stride1'][::2,0],v['stride2'][::2,0])
        for stride in [1,2]:assert m['parity'][str(stride)]['frames']==(m['frames']+stride-1)//stride
    scene={};bootstrap={};rng=np.random.default_rng(20261009);samples={k:{a:{} for a in ARMS} for k in KEYS}
    old=read(ROOT/'results/E11B/seed42/historical_summary.json')['by_scene'];columns=[0,4,5,6,3]
    for sc in SCENES:
        ids=[m['source_id'] for m in metadata if m['scene']==sc];vs=[arrays[i] for i in ids]
        y=np.concatenate([v['labels'] for v in vs]);scene[sc]={}
        for key in KEYS:
            scores=np.concatenate([v[key] for v in vs]);assert np.isfinite(scores).all()
            scene[sc][key]={a:frame_metrics(y,scores[:,j],vs[0]['thresholds'][j]) for j,a in enumerate(ARMS)}
        for j,a in enumerate(ARMS):
            for metric in ['auroc','ap','fpr','recall']:
                np.testing.assert_allclose(scene[sc]['cached'][a][metric],old[sc][columns[j]][metric],atol=1e-12,rtol=1e-12)
        valid=[v for v in vs if np.any(v['labels']>=0)];y=np.concatenate([v['labels'][v['labels']>=0] for v in valid])
        owner=np.concatenate([np.full((v['labels']>=0).sum(),i) for i,v in enumerate(valid)])
        counts=np.column_stack([np.ones(len(valid),dtype=int),rng.multinomial(len(valid),np.full(len(valid),1/len(valid)),size=2000).T])
        for key in KEYS:
            scores=np.concatenate([v[key][v['labels']>=0] for v in valid])
            for j,a in enumerate(ARMS):
                ap,auc=weighted_metrics(y,scores[:,j],owner,counts)
                np.testing.assert_allclose([ap[0],auc[0]],[average_precision_score(y,scores[:,j]),roc_auc_score(y,scores[:,j])],atol=1e-12)
                samples[key][a][sc]={'ap':ap[1:],'auroc':auc[1:]}
    macro={k:{a:{metric:float(np.mean([scene[sc][k][a][metric] for sc in SCENES])) for metric in ['auroc','ap','fpr','recall']} for a in ARMS} for k in KEYS}
    for hi,lo in [('stride1','stride2'),('stride2','cached')]:
        name=f'{hi}_minus_{lo}';bootstrap[name]={}
        for a in ARMS:
            bootstrap[name][a]={}
            for metric in ['auroc','ap']:
                delta=100*np.mean([samples[hi][a][sc][metric]-samples[lo][a][sc][metric] for sc in SCENES],axis=0);finite=np.isfinite(delta)
                bootstrap[name][a][metric]={'delta_pp':100*(macro[hi][a][metric]-macro[lo][a][metric]),'ci95_pp':np.quantile(delta[finite],[.025,.975]).tolist(),'finite_resamples':int(finite.sum())}
    summary={'protocol':cfg,'videos':66,'valid_videos':63,'source_frames':33462,'valid_frames':31550,'unknown_frames':1912,'scene':scene,'macro':macro,'bootstrap':{'resamples':2000,'seed':20261009,'unit':'source_file','paired':True,'stratified_by':'scene','uncertainty':'Fixed checkpoint only; recording independence unverified; no retraining uncertainty','contrasts':bootstrap},'parity':{k:max(v['parity'][s][k] for v in metadata for s in ['1','2']) for k in ['max_trace_error','max_C_error','max_P_error']},'baseline_metrics_reproduced':True}
    write_json(OUT/'summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'CycleVAD-E17'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');xs=np.arange(4)
    for ax,metric,label in zip(axes,['auroc','ap'],['AUROC (%)','Average precision (%)']):
        for key,color,marker,name in [('cached','#9399A3','x','Cached stride 2'),('stride2','#2B6CB0','o','Online stride 2'),('stride1','#B47714','s','Online stride 1')]:
            ax.plot(xs,[100*scene[s][key]['CP'][metric] for s in SCENES],marker=marker,color=color,label=name,linewidth=1.6)
        ax.set_xticks(xs,SCENES);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_ylim(40,100)
    axes[0].legend(loc='lower left',fontsize=9);fig.suptitle('E17 | C + P accuracy on complete historical videos\nFrozen seed 42 models; current/past observations only',fontsize=13)
    fig.savefig(ROOT/'docs/figures/E17_online_accuracy.svg',metadata={'Date':None});fig.savefig(ROOT/'docs/figures/E17_online_accuracy.png',dpi=160);plt.close(fig)
    lines=['## 온라인 탐지 정확도 — E17 완료','',
    'AUROC는 온라인 이상 점수의 **탐지 정확도**를 평가하고, FPS·지연은 처리 속도를 평가합니다. 각 점수는 현재·과거 관측만으로 생성하며, AUROC/AP 계산에 정답을 사후 사용하는 것은 미래 프레임 참조가 아닙니다.','',
    '**R01–R04 전체 테스트 66개 영상·33,462프레임**을 JPEG부터 batch 1로 순차 처리했습니다. strict-v1에서 라벨 길이가 맞지 않는 R02의 3개 영상·1,912프레임은 처리하되 지표에서는 제외했습니다. 평가 분모는 63개 영상·31,550프레임입니다. Seed 42/fold 0의 E11 모델·보정값·E11B 임계값을 고정했으며 재학습·테스트 라벨 기반 튜닝은 없습니다.','',
    'Stride 1은 매 프레임 새 점수를 생성하고, stride 2는 짝수 프레임에서만 갱신하며 홀수 프레임에 직전 점수를 유지합니다. **두 경우 모두 전체 유효 프레임에서 평가**했습니다. Stride 1은 기존 stride 2 모델에 시간 간격과 source-frame lag만 맞춘 실험이며 별도 cadence 보정은 하지 않았습니다.','',
    '| 모델 | 기존 캐시 S2 AUROC / AP % | 온라인 S2 AUROC / AP % | 온라인 S1 AUROC / AP % |','| --- | --- | --- | --- |']
    for a in ARMS:lines.append('| '+a+' | '+' | '.join(f'{100*macro[k][a]["auroc"]:.2f} / {100*macro[k][a]["ap"]:.2f}' for k in KEYS)+' |')
    lines+=['','표는 장면별 pooled-frame 지표를 계산한 뒤 네 장면을 동일 가중한 Macro 평균입니다. 기존 캐시 결과도 인과적 추정이며, 온라인 S2와의 차이는 실행 경로·특징 추출 배치 등의 수치 차이를 포함합니다. 캐시 특징은 온라인 추론에 사용하지 않았습니다.','',
    '| 장면 | 온라인 S1 C AUROC / AP % | 온라인 S1 C+P AUROC / AP % | 온라인 S2 C+P AUROC / AP % |','| --- | --- | --- | --- |']
    for sc in SCENES:
        vals=[scene[sc][k][a] for k,a in [('stride1','C'),('stride1','CP'),('stride2','CP')]]
        lines.append('| '+sc+' | '+' | '.join(f'{v["auroc"]*100:.2f} / {v["ap"]*100:.2f}' for v in vals)+' |')
    lines+=['','![온라인 정확도](figures/E17_online_accuracy.svg)','','| C+P 비교 | Macro AUROC 변화 pp [95% CI] | Macro AP 변화 pp [95% CI] |','| --- | --- | --- |']
    for contrast in bootstrap:
        vals=bootstrap[contrast]['CP'];lines.append('| '+contrast+' | '+' | '.join(f'{vals[m]["delta_pp"]:+.3f} [{vals[m]["ci95_pp"][0]:+.3f}, {vals[m]["ci95_pp"][1]:+.3f}]' for m in ['auroc','ap'])+' |')
    lines+=['','CI는 장면별 source-file paired bootstrap 2,000회입니다. 재학습 불확실성과 원본 recording 간 종속성은 반영하지 않습니다. 같은 개발 테스트를 재사용하므로 독립 데이터에서의 일반화 검증은 아닙니다.','',
    f'현재 고정 모델에서 온라인 stride 2는 기존 결과를 거의 재현했습니다. 반면 stride 1 전환 시 C+P의 Macro AUROC는 {bootstrap["stride1_minus_stride2"]["CP"]["auroc"]["delta_pp"]:+.2f} pp, AP는 {bootstrap["stride1_minus_stride2"]["CP"]["ap"]["delta_pp"]:+.2f} pp 변했습니다. 판별 빈도를 높였다고 정확도가 개선되는 것은 아닙니다. 논문에서 기존 stride 2 결과와 매 프레임 stride 1 결과를 구분해야 하며, 이 비교로 재학습한 stride 1 모델의 성능까지 판단할 수는 없습니다.','',
    '**경보 성능은 AUROC와 다릅니다.** 고정된 정상 window-q99 임계값에서 C+P의 historical 정상 프레임 FPR / 이상 프레임 Recall Macro 평균은 '+', '.join(f'{k}: {macro[k]["CP"]["fpr"]*100:.2f}% / {macro[k]["CP"]["recall"]*100:.2f}%' for k in ['stride2','stride1'])+'입니다. 높은 AUROC가 낮은 오경보나 충분한 경보 Recall을 보장하지 않으며, E15의 보정 문제는 별도 과제로 남습니다.','',
    'E16의 별도 속도 측정에서 C+P stride 1은 **87.36 FPS**, 새 점수 지연 p95는 **12.77 ms**였습니다. E17은 전체 영상 정확도 평가이며, 현재 프레임의 encoder 출력만 두 개의 독립 추적 상태에 공유했습니다. E17 실행 시간은 두 설정 동시 평가와 검증을 포함하므로 단일 파이프라인 FPS로 사용하지 않습니다. E16의 카메라·영상 코덱·전송 지연 제외 조건도 유지됩니다.','',
    '모든 영상·두 cadence에서 새로 추출한 동일 특징에 대한 배치 참조와 온라인 tracker/C/P 점수 일치를 확인했습니다. 영상 사이에서만 상태를 초기화하며, 영상 내부에서 이상 라벨에 따라 초기화하지 않습니다. 홀수 프레임 hold 일치, 전체 프레임 수·unknown 제외, 기존 E11B 지표 재현도 검사했습니다.','',
    '[실험 명세](../configs/experiments/followup/online_accuracy_v1.json) · [전체 지표·CI·검증](../results/E17/summary.json) · [실행 환경](../results/E17/environment.json) · [검사 로그](../results/E17/validation.txt) · [재현 코드](../scripts/evaluate_streaming.py)','']
    lines+=['재현: 기존 E11/E11B seed 42 checkpoint와 IPAD 원본 JPEG가 있는 환경에서 실행합니다. 모델 경로와 encoder revision은 위 명세에 고정되어 있습니다.','',
            '```bash','PYTHONPATH=src python scripts/evaluate_streaming.py --resume',
            'MPLCONFIGDIR=/tmp/cyclevad-mpl PYTHONPATH=src python scripts/report_online_accuracy.py','```','']
    doc='\n'.join(lines)+'\n';(ROOT/'docs/ONLINE_ACCURACY_RESULTS.md').write_text(doc)
    fragment=doc.replace('](../results/','](results/').replace('](../configs/','](configs/').replace('](../scripts/','](scripts/').replace('](figures/','](docs/figures/')
    path=ROOT/'README.md';text=path.read_text();start=text.find('## 온라인 탐지 정확도 — E17 완료');anchor='## 온라인 스트리밍'
    # Locate E16 by its actual heading; insert immediately before it.
    heading=next(x for x in text.splitlines() if x.startswith('## ') and 'E16' in x)
    if start>=0:text=text[:start]+text[text.index(heading,start):]
    text=text.replace(heading,fragment+'\n'+heading,1);path.write_text(text)
    for path in [ROOT/'docs/STREAMING_RESULTS.md',ROOT/'README.md']:
        text=path.read_text().replace('Stride 1의 이상 탐지 정확도·임계값 보정은 재검증하지 않았습니다.', 'E16에서는 Stride 1 정확도를 미평가했으며, 후속 E17에서 전체 영상의 온라인 AUROC/AP를 평가했습니다. 임계값 재보정은 하지 않았습니다.')
        path.write_text(text)
    path=ROOT/'results/followup_status.json';status=read(path);status['E17']='complete_full_historical_causal_accuracy_frozen_seed42';write_json(path,status)
    print(json.dumps({'macro':macro,'CP_contrasts':{k:v['CP'] for k,v in bootstrap.items()},'parity':summary['parity']},indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
