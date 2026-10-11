"""Validate and report actual batch-one JPEG streaming throughput and latency."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycle_vad.data import write_json
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/E16'
def read(p):return json.loads(Path(p).read_text())
def stats(v):
    v=np.asarray(v,float);return {'mean':float(v.mean()),'p50':float(np.quantile(v,.5)),'p95':float(np.quantile(v,.95)),'p99':float(np.quantile(v,.99)),'max':float(v.max())}
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def f(v):return f'{v:.2f}'
def main():
    env=read(BASE/'environment.json');cfg=read(ROOT/'configs/experiments/followup/streaming_v1.json');paths=sorted((BASE/'runs').glob('*.json'))
    runs=[read(p) for p in paths];main=[r for r in runs if r['repeat'] in [0,1,2] and r['target_rate'] is None];paced=[r for r in runs if r['target_rate'] is not None];parity=[r for r in runs if r['repeat']==99]
    assert len(main)==12 and len(paced)==4 and len(parity)==2
    assert env['instrumented_encoder_matches_original_batch1'] and env['batch_size']==1 and not env['prefetch']
    dimensions=sorted({tuple(d) for r in main for d in r['source_dimensions']})
    rows=[];by_scene=[]
    for readout in ['CP','Full']:
        for stride in [1,2]:
            group=sorted([r for r in main if r['readout']==readout and r['stride']==stride],key=lambda r:r['repeat']);assert len(group)==3
            raw=[]
            for r in group:
                records=read(BASE/'timings'/f'{r["id"]}.json');assert len(records)==r['input_frames']==2048
                assert sum(v['fresh'] for v in records)==r['fresh_updates']==2048//stride
                assert set(v['source_id'] for v in records)==set(cfg['files'])
                for source in cfg['files']:
                    v=[x for x in records if x['source_id']==source];assert [x['index'] for x in v]==list(range(128))
                    assert [x['fresh'] for x in v]==[i%stride==0 for i in range(128)]
                    for k in range(1,len(v)):
                        if not v[k]['fresh']:assert v[k]['score']==v[k-1]['score'] and v[k]['theta']==v[k-1]['theta'] and v[k]['alarm']==v[k-1]['alarm']
                np.testing.assert_allclose(r['input_fps'],len(records)/r['wall_seconds']);raw+=records
            fresh=[v for v in raw if v['fresh']]
            stages={k:stats([v['stage_ms'][j] for v in fresh]) for j,k in enumerate(['decode','preprocess','h2d','gpu_features','d2h','detector'])}
            row={'readout':readout,'stride':stride,'input_frames':len(raw),'fresh_updates':len(fresh),'input_fps':len(raw)/sum(r['wall_seconds'] for r in group),'fresh_fps':len(fresh)/sum(r['wall_seconds'] for r in group),'run_input_fps':[r['input_fps'] for r in group],'fresh_latency_ms':stats([v['service_ms'] for v in fresh]),'stages_ms':stages,'peak_gpu_allocated_MiB':max(r['gpu_peak_allocated_MiB'] for r in group),'peak_gpu_reserved_MiB':max(r['gpu_peak_reserved_MiB'] for r in group),'process_lifetime_peak_rss_MiB':max(r['process_peak_rss_MiB'] for r in group)}
            rows.append(row)
            for sc in ['R01','R02','R03','R04']:
                records=[v for v in raw if v['source_id'].startswith(sc+'/')];fs=[v for v in records if v['fresh']]
                seconds=sum(v['wall_seconds'] for r in group for v in r['files'] if v['source_id'].startswith(sc+'/'))
                by_scene.append({'scene':sc,'readout':readout,'stride':stride,'input_fps':len(records)/seconds,'fresh_fps':len(fs)/seconds,'fresh_latency_ms':stats([v['service_ms'] for v in fs])})
    for r in parity:
        assert len(r['validation'])==4
        assert sum(v['frames'] for v in r['validation'])==512//r['stride']
        assert all(v['max_trace_error']<1e-10 and v['max_C_error']<1e-9 and v['max_P_error']<1e-9 and v['alarm_agreement']==1 for v in r['validation'])
    for r in paced:
        raw=read(BASE/'timings'/f'{r["id"]}.json');assert len(raw)==512
        assert min(v['queue_wait_ms'] for v in raw)>=0
        assert all(v['arrival_to_output_ms']>=v['service_ms'] for v in raw)
        np.testing.assert_allclose(np.mean([v['missed_period'] for v in raw]),r['deadline_miss_fraction'])
    # Same-source repeatability: new input images are encoded afresh in every run.
    repeatability=[]
    for stride in [1,2]:
        for readout in ['CP','Full']:
            traces=[read(BASE/'timings'/f'{readout}_stride{stride}_repeat{r}.json') for r in range(3)]
            assert all([(v['source_id'],v['index']) for v in traces[0]]==[(v['source_id'],v['index']) for v in t] for t in traces[1:])
            score=np.array([[v['score'] for v in t] for t in traces]);alarm=np.array([[v['alarm'] for v in t] for t in traces]);delta=float(abs(score-score[0]).max())
            repeatability.append({'readout':readout,'stride':stride,'max_score_difference':delta,'alarms_equal':bool(np.all(alarm==alarm[0]))})
            assert delta<1e-9 and np.all(alarm==alarm[0])
    result={'status':'complete','aggregate':rows,'by_scene':by_scene,'paced':paced,'repeatability':repeatability,'parity':parity,'environment':env,'source_dimensions':dimensions,'metric_scope':'JPEG file read/decode through alarm; synchronous serial batch1; steady state after 100 warmup frames; no future images or feature cache; fresh and held decisions separated'}
    write_json(BASE/'summary.json',result)
    (BASE/'validation.txt').write_text('PASS 12 throughput runs: 24,576 input frames, batch1, fixed 16 sources, 3 repeats, current/held score semantics.\nPASS 4 paced runs: 2,048 input frames at configured 30/60 FPS; arrival/service/wait and deadline counts checked.\nPASS 768 fresh observations across 8 source/cadence sequences match the reference causal tracker and conditional/process heads; encoder equals original batch1 path.\nPASS Repeat scores and alarms identical; maximum tracker/replay tolerances checked; no cached feature throughput reported.\n')
    text='## 실제 입력 스트리밍 성능 E16\n\n[실행 설정](configs/experiments/followup/streaming_v1.json) · [전체 수치](results/E16/summary.json) · [검증](results/E16/validation.txt) · [테스트](results/streaming_tests.txt)\n\n'
    text+='로컬 **'+env['gpu']+'**, 고정 DINOv2-base 336×336, FP16 AMP, batch 1로 데이터셋 JPEG(입력 '+', '.join(f'{w}×{h}' for w,h in dimensions)+')를 매번 다시 읽었습니다. JPEG 읽기·디코딩→전처리→GPU 전송→특징 추출→CPU 추적·C/P 또는 Full→임계값 판정을 포함합니다. 미래 이미지 prefetch, 특징 캐시, 동적 batch, torch.compile은 사용하지 않았습니다. 모델은 seed 42/fold 0 고정이며 CPU는 AMD Ryzen 5 5600X 6-Core Processor이며 연산은 4 threads입니다.\n\n'
    text+='정상 evaluation 2개와 historical 2개씩 총 16개 영상에서 각각 처음 128프레임을 재생했습니다. 설정당 2,048 입력 프레임×3회, 전체 24,576 입력 프레임을 측정했습니다. 별도 FIT 영상 100프레임으로 warmup했고 녹화 전환마다 상태를 초기화했습니다. 파일·프레임 목록은 설정에 공개했습니다.\n\n'
    text+=table(['구성','Stride','입력 처리 FPS','새 점수 / 초','새 판정 p50 ms','p95 ms','p99 ms','입력 FPS 3회 범위'],[[v['readout'],v['stride'],f(v['input_fps']),f(v['fresh_fps'])]+[f(v['fresh_latency_ms'][k]) for k in ['p50','p95','p99']]+[f'{min(v["run_input_fps"]):.2f}–{max(v["run_input_fps"]):.2f}'] for v in rows])
    text+='FPS는 총 프레임 수÷반복 실행 총 wall time입니다. 지연 분위수는 새 점수를 계산한 프레임만 모았습니다. Stride 2의 중간 프레임은 실제 JPEG를 읽되 특징 추출 없이 직전 판정을 유지하므로, 입력 FPS를 매 프레임 새 추론 FPS로 해석하면 안 됩니다. 단일 스트림이며 여러 카메라의 총 처리량이 아닙니다.\n\n'
    text+='**Stride 2가 기존 평가 설정입니다. Stride 1은 같은 가중치에서 source-time 전이 간격과 과거 lag를 1프레임 간격으로 맞춘 속도 실험입니다. Stride 1의 이상 탐지 정확도·임계값 보정은 재검증하지 않았습니다.** 저장된 D-window-q99 임계값은 정상 보정에 사용한 통계량이며, 실행 시 미래 D프레임을 모아 기다리는 창이 아닙니다. 현재 프레임 점수가 임계값을 넘으면 즉시 출력합니다.\n\n'
    text+='![새 판정 지연과 처리량](docs/figures/E16_runtime.svg)\n\n'
    text+=table(['구성','Stride','decode/reset ms','전처리 ms','H2D ms','GPU 특징 ms','D2H ms','추적·점수 ms'],[[v['readout'],v['stride']]+[f(v['stages_ms'][k]['mean']) for k in ['decode','preprocess','h2d','gpu_features','d2h','detector']] for v in rows])
    text+='단계 값은 새 관측 프레임의 평균입니다. CUDA를 구간마다 동기화하여 GPU 비동기 호출 시간을 처리 시간으로 오인하지 않았습니다. 구간 계측과 Python 호출의 작은 잔여 비용 때문에 단계 합과 전체 지연은 완전히 같지 않을 수 있습니다.\n\n'
    text+=table(['장면','C+P stride 1 FPS','p95 ms','C+P stride 2 입력 FPS','p95 ms'],[[sc]+[f(next(v for v in by_scene if v['scene']==sc and v['readout']=='CP' and v['stride']==stride)[key] if key=='input_fps' else next(v for v in by_scene if v['scene']==sc and v['readout']=='CP' and v['stride']==stride)['fresh_latency_ms']['p95']) for stride,key in [(1,'input_fps'),(1,'p95'),(2,'input_fps'),(2,'p95')]] for sc in ['R01','R02','R03','R04']])
    text+='### 설정한 30/60 FPS 도착 시각 재생\n\n장면당 별도 128프레임, 총 512프레임씩 CP 경로를 재생했습니다. 프레임 i의 도착 예정 시각은 시작+i/목표 FPS이고, 도착 전에는 이미지를 읽지 않습니다. 늦으면 FIFO 순서로 처리하며 프레임을 버리지 않습니다. 영상마다 시계와 상태를 초기화했습니다. 실제 카메라의 원본 FPS를 추정한 결과가 아닙니다.\n\n'
    text+=table(['Stride','목표 FPS','새 판정 도착→완료 p50 ms','p95 ms','p99 ms','최대 추가 대기 프레임','한 프레임 주기 초과 %'],[[r['stride'],r['target_rate']]+[f(r['fresh_arrival_to_output_ms'][k]) for k in ['p50','p95','p99']]+[r['max_backlog_frames'],f(100*r['deadline_miss_fraction'])] for r in sorted(paced,key=lambda v:(v['stride'],v['target_rate']))])
    text+='추가 대기 프레임은 각 처리 시작 시 현재 처리할 프레임을 제외한 대기 수입니다. 현재 프레임 자체의 대기 시간은 도착→완료 지연에 포함됩니다. 한 프레임 주기 초과율은 held 판정까지 포함한 전체 입력 기준입니다. Stride 2에서는 신규 점수 간격이 입력 주기의 2배이며, 새 이상이 건너뛴 프레임에 시작하면 다음 관측까지 최대 한 입력 주기의 추가 지연이 생길 수 있습니다. 한 번의 짧은 재생 결과가 장시간 카메라 운영을 보장하지는 않습니다.\n\n'
    text+=f'모델 로딩 {env["encoder_load_seconds"]:.2f}초, warmup 전 첫 JPEG→판정 {env["first_frame_before_warmup_ms"]:.2f}ms였습니다. steady-state 표에서는 이 초기 비용을 제외했습니다. PyTorch peak GPU allocated는 최대 {max(v["peak_gpu_allocated_MiB"] for v in rows):.1f} MiB, reserved는 {max(v["peak_gpu_reserved_MiB"] for v in rows):.1f} MiB입니다. process-lifetime peak RSS는 {max(v["process_lifetime_peak_rss_MiB"] for v in rows):.1f} MiB이며, GPU 전체 장치 점유나 런별 증가량이 아닙니다.\n\n'
    text+='인과성 검증: 추적기는 누적 posterior와 최대 32 source-frame의 lag buffer를 유지하며 현재까지의 관측만 사용합니다. 같은 prefix 뒤의 미래 입력을 바꾼 단위 검사가 통과했고, 실제 새로 추출한 특징 768개에서 CP 순차 실행과 기존 인과적 reference의 위치·C/P·경보가 일치했습니다. 3회 반복에서도 점수와 경보가 일치했습니다.\n\n'
    text+='측정 범위의 한계: IPAD는 현재 로컬 JPEG frame sequence이므로 H.264/H.265 디코딩, 실제 카메라/네트워크 수신, 화면 렌더링·알림 전송은 포함하지 않습니다. OS 파일 캐시는 비우지 않았습니다. 이 실험은 정확도 개선이나 R01 오경보 해결의 근거가 아닙니다.\n\n'
    fig,axes=plt.subplots(1,2,figsize=(11,4.6));labels=[f'{v["readout"]}\nstride {v["stride"]}' for v in rows];x=np.arange(4);color=['#3873a1','#3873a1','#b78730','#b78730']
    axes[0].bar(x-.18,[v['input_fps'] for v in rows],.36,label='Input frames/s',color=color)
    axes[0].bar(x+.18,[v['fresh_fps'] for v in rows],.36,label='Fresh scores/s',facecolor='white',edgecolor=color,hatch='//')
    axes[0].set_xticks(x,labels);axes[0].set_ylabel('Frames or fresh decisions / second');axes[0].set_ylim(0,max(v['input_fps'] for v in rows)*1.28);axes[0].legend(frameon=False)
    p50=np.array([v['fresh_latency_ms']['p50'] for v in rows]);p95=np.array([v['fresh_latency_ms']['p95'] for v in rows]);p99=np.array([v['fresh_latency_ms']['p99'] for v in rows])
    axes[1].bar(x,p50,color=color,label='p50');axes[1].scatter(x,p95,color='#222',marker='D',label='p95');axes[1].scatter(x,p99,color='#222',marker='x',label='p99');axes[1].set_xticks(x,labels);axes[1].set_ylabel('Fresh JPEG-to-alarm latency (ms)');axes[1].set_ylim(0,max(p99)*1.3);axes[1].legend(frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(ROOT/'docs/figures/E16_runtime.svg',bbox_inches='tight',metadata={'Date':None});fig.savefig(ROOT/'docs/figures/E16_runtime.png',bbox_inches='tight',dpi=150);plt.close(fig)
    if (ROOT/'results/E17/summary.json').exists():
        text=text.replace('Stride 1의 이상 탐지 정확도·임계값 보정은 재검증하지 않았습니다.', 'E16에서는 Stride 1 정확도를 미평가했으며, 후속 E17에서 전체 영상의 온라인 AUROC/AP를 평가했습니다. 임계값 재보정은 하지 않았습니다.')
    (ROOT/'docs/STREAMING_RESULTS.md').write_text(text.replace('](docs/','](').replace('](configs/','](../configs/').replace('](results/','](../results/'))
    p=ROOT/'README.md';old=p.read_text();title='## 실제 입력 스트리밍 성능 E16';start=old.find(title)
    if start>=0:
        end=old.index('## 보정과 경보 결합',start);old=old[:start]+old[end:]
    position=old.index('## 보정과 경보 결합');p.write_text(old[:position]+text+old[position:])
    status=read(ROOT/'results/followup_status.json');status['E16']='jpeg_streaming_complete_camera_codec_pipeline_not_measured';write_json(ROOT/'results/followup_status.json',status)
    print(json.dumps(rows,indent=2));print('Validated E16: 12 throughput + 4 paced + 2 actual-feature parity runs.')
if __name__=='__main__':main()
