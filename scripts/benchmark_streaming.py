"""Batch-one, no-lookahead JPEG-to-alarm benchmark on the local CUDA GPU."""
import os
os.environ.setdefault('HF_HOME',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse,copy,json,time,platform,resource,subprocess
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from cycle_vad.features import VisualEncoder
from cycle_vad.streaming import StreamingDetector
from cycle_vad.data import inventory,frame_paths,write_json,fingerprint
from cycle_vad.paper_diagnostics import sha
from cycle_vad.paper_experiments import conditional_raw
from cycle_vad.phase_experiment import compact
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/E16';LOCAL=ROOT/'runs/E16';LOCAL.mkdir(parents=True,exist_ok=True)
STAGES=['decode','preprocess','h2d','gpu_features','d2h','detector']
def read(p):return json.loads(Path(p).read_text())

def encode(encoder,image):
    clocks=[];a=time.perf_counter();pixels=encoder.preprocess(image).unsqueeze(0);b=time.perf_counter();clocks.append(b-a)
    pixels=pixels.to(encoder.device);torch.cuda.synchronize();a=time.perf_counter();clocks.append(a-b)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        out=encoder.model(pixel_values=pixels,output_hidden_states=True,return_dict=True)
        global_z=F.normalize(out.last_hidden_state[:,0].float(),dim=-1);side=encoder.size//encoder.model.config.patch_size;layers=[]
        for layer in encoder.layers:
            tokens=out.hidden_states[layer][:,1:].float();spatial=F.normalize(tokens,dim=-1).transpose(1,2).reshape(1,-1,side,side)
            spatial=F.adaptive_avg_pool2d(spatial,(encoder.grid,encoder.grid)).flatten(2).transpose(1,2);layers.append(F.normalize(spatial,dim=-1))
    # Exactly the existing encoder's FP32 projection outside AMP.
    with torch.inference_mode():patches=F.normalize(torch.cat(layers,dim=-1)@encoder.projection,dim=-1)
    torch.cuda.synchronize();b=time.perf_counter();clocks.append(b-a)
    global_z=global_z.cpu().numpy();patches=patches.cpu().numpy();torch.cuda.synchronize();a=time.perf_counter();clocks.append(a-b)
    return global_z[0],patches[0],clocks

def stats(values):
    v=np.asarray(values,float)
    return {'n':len(v),'mean':float(v.mean()),'p50':float(np.quantile(v,.5)),'p95':float(np.quantile(v,.95)),'p99':float(np.quantile(v,.99)),'max':float(v.max())} if len(v) else None

def check_sequence(records,m,stride):
    # Reference sees the SAME newly extracted batch-one features as the adapter.
    t=copy.copy(m['base']['tracker']);t.stride=stride
    x=np.array([r['x'] for r in records]);z=np.array([r['z'] for r in records]);ids=np.arange(len(x))*stride
    reference=t.predict_latent(x,ids);errors={}
    for key in reference:
        values=np.array([r['trace'][key] for r in records]);errors[key]=float(np.max(abs(values-reference[key])))
        np.testing.assert_allclose(values,reference[key],atol=1e-10,rtol=1e-10)
    co,ci,_=conditional_raw(z,reference['angle'],m['coef'],m['h'],m['space'])
    c=np.maximum(m['ccal'][0].score(co),m['ccal'][1].score(ci));p=np.maximum.reduce([m['pcal'][k].score(reference[k]) for k in m['pcal']])
    np.testing.assert_allclose(c,[r['C'] for r in records],atol=1e-9,rtol=1e-9);np.testing.assert_allclose(p,[r['P'] for r in records],atol=1e-9,rtol=1e-9)
    cp=np.maximum(c,p);actual=np.maximum([r['C'] for r in records],[r['P'] for r in records]);np.testing.assert_array_equal(cp>records[0]['threshold'],actual>records[0]['threshold'])
    return {'frames':len(records),'max_trace_error':max(errors.values()),'max_C_error':float(max(abs(c-np.array([r['C'] for r in records])))),'max_P_error':float(max(abs(p-np.array([r['P'] for r in records])))),'alarm_agreement':1.0}

def pass_once(encoder,models,selected,stride,readout,repeat,limit,target_rate=None,validate=False):
    tag=f'{readout}_stride{stride}_repeat{repeat}'+('' if target_rate is None else f'_paced{target_rate}')
    times=[];validation=[];dimensions=set();file_summaries=[];torch.cuda.reset_peak_memory_stats();runstart=time.perf_counter();freshcount=0
    # Every file is a separate recording; reset is included in first-frame latency.
    for item in selected:
        m= models[item['scene']];saved=read(ROOT/f'results/E11B/seed42/{item["scene"]}/fold0/run.json')
        threshold=saved['window_thresholds'][3][6 if readout=='CP' else 3]
        detector=None;sequence=[];file_start=time.perf_counter();paths=frame_paths(item['directory'])[:limit]
        # Filename enumeration uses metadata only; no future image is opened/prefetched.
        arrival_base=time.perf_counter();previous_score=None
        for index,path in enumerate(paths):
            due=arrival_base+index/target_rate if target_rate else None
            if due is not None:
                remaining=due-time.perf_counter()
                if remaining>0:time.sleep(remaining)
            start=time.perf_counter()
            if detector is None:detector=StreamingDetector(m,threshold,stride,readout)
            with Image.open(path) as f:image=f.convert('RGB')
            decoded=time.perf_counter();dimensions.add(image.size);parts=[decoded-start];fresh=index%stride==0
            if fresh:
                g,p,stage=encode(encoder,image);parts+=stage;head=time.perf_counter();prediction=detector.step(g,p);done=time.perf_counter();parts.append(done-head);freshcount+=1
                if validate:sequence.append(prediction)
            else:
                head=time.perf_counter();prediction=detector.step();done=time.perf_counter();parts += [0.,0.,0.,0.,done-head]
                assert prediction['score']==previous_score
            previous_score=prediction['score'];image.close()
            record={'source_id':item['id'],'index':index,'fresh':fresh,'service_ms':1000*(done-start),
                    'stage_ms':[1000*x for x in parts],'score':prediction['score'],'theta':prediction['theta'],'alarm':prediction['alarm']}
            if due is not None:
                record.update(arrival_to_output_ms=1000*(done-due),queue_wait_ms=1000*(start-due),
                              backlog_frames=min(len(paths)-1-index,max(0,int((start-arrival_base)*target_rate)-index)),
                              missed_period=bool(done-due>1/target_rate))
            times.append(record)
        file_summaries.append({'source_id':item['id'],'frames':len(paths),'fresh_updates':sum(t['fresh'] for t in times if t['source_id']==item['id']), 'wall_seconds':time.perf_counter()-file_start})
        # Expensive checks are deliberately outside benchmark wall-time accounting.
        if validate:
            before=time.perf_counter();validation.append({'source_id':item['id'],**check_sequence(sequence,m,stride)});duration=time.perf_counter()-before;runstart+=duration
            np.savez_compressed(LOCAL/f'{tag}_{item["id"].replace("/","_")}.npz',x=np.array([q['x'] for q in sequence]),z=np.array([q['z'] for q in sequence]),scores=np.array([[q['C'],q['P']] for q in sequence]))
    elapsed=time.perf_counter()-runstart
    # Detailed score/latency rows are small numeric artifacts, no raw images.
    compact(OUT/'timings'/f'{tag}.json',times)
    fresh=[r for r in times if r['fresh']]
    result={'id':tag,'stride':stride,'readout':readout,'repeat':repeat,'target_rate':target_rate,'input_frames':len(times),'fresh_updates':freshcount,
            'wall_seconds':elapsed,'input_fps':len(times)/elapsed,'fresh_updates_per_second':freshcount/elapsed,
            'all_service_ms':stats([r['service_ms'] for r in times]),'fresh_service_ms':stats([r['service_ms'] for r in fresh]),
            'held_service_ms':stats([r['service_ms'] for r in times if not r['fresh']]),
            'fresh_stage_ms':{k:stats([r['stage_ms'][j] for r in fresh]) for j,k in enumerate(STAGES)},
            'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'gpu_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
            'process_peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'source_dimensions':sorted(dimensions),'files':file_summaries,'validation':validation}
    if target_rate is not None:
        result.update(arrival_to_output_ms=stats([r['arrival_to_output_ms'] for r in times]),fresh_arrival_to_output_ms=stats([r['arrival_to_output_ms'] for r in fresh]),queue_wait_ms=stats([r['queue_wait_ms'] for r in times]),max_backlog_frames=max(r['backlog_frames'] for r in times),deadline_miss_fraction=float(np.mean([r['missed_period'] for r in times])))
    write_json(OUT/'runs'/f'{tag}.json',result)
    print(f'{tag}: input {result["input_fps"]:.2f} FPS; new scores {result["fresh_updates_per_second"]:.2f}/s; fresh p50/p95 {result["fresh_service_ms"]["p50"]:.2f}/{result["fresh_service_ms"]["p95"]:.2f} ms',flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available();cfg=read(ROOT/'configs/experiments/replay_pinned.json')['encoder'];began=time.perf_counter()
    with threadpool_limits(limits=4):
        encoder=VisualEncoder(cfg);torch.cuda.synchronize();load_seconds=time.perf_counter()-began
        models={sc:joblib.load(ROOT/f'runs/E11/seed42/{sc}/fold0/model.joblib') for sc in ['R01','R02','R03','R04']}
        rows=inventory(ROOT.parent/'IPAD_dataset',['R01','R02','R03','R04']);lookup={r['id']:r for r in rows};selected=[]
        for sc in models:
            r=read(ROOT/f'results/E11B/seed42/{sc}/fold0/run.json')
            chosen=sorted(r['splits']['normal_evaluation'])[:2]+[x['id'] for x in rows if x['scene']==sc and x['partition']=='testing'][:2]
            selected.extend(lookup[i] for i in chosen)
        assert all(r['frames']>=128 for r in selected)
        # Warmup uses FIT frames, discarded and reset before every measured recording.
        warm_id=read(ROOT/'results/E11B/seed42/R01/fold0/run.json')['splits']['fit'][0];warm_paths=frame_paths(lookup[warm_id]['directory'])
        warm=StreamingDetector(models['R01'],read(ROOT/'results/E11B/seed42/R01/fold0/run.json')['window_thresholds'][3][6],2,'Full');cold=None;wb=time.perf_counter()
        for i in range(100):
            before=time.perf_counter()
            with Image.open(warm_paths[i%len(warm_paths)]) as f:im=f.convert('RGB')
            g,patches,_=encode(encoder,im)
            if i%2==0:warm.step(g,patches)
            else:warm.step()
            if i==0:cold=time.perf_counter()-before
        with Image.open(warm_paths[0]) as f:parity_image=f.convert('RGB')
        reference_g,reference_p=encoder.encode([parity_image]);measured_g,measured_p,_=encode(encoder,parity_image)
        np.testing.assert_array_equal(reference_g[0],measured_g);np.testing.assert_array_equal(reference_p[0],measured_p)
        parity_image.close()
        environment={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),'cpu':platform.processor(),'torch_threads':4,'interop_threads':1,'encoder_identity':encoder.identity,'batch_size':1,'prefetch':False,'instrumented_encoder_matches_original_batch1':True,'warmup_frames':100,'warmup_seconds':time.perf_counter()-wb,'encoder_load_seconds':load_seconds,'first_frame_before_warmup_ms':1000*cold,'os_page_cache':'not flushed; repeated files may be cached','input_format':'source JPEG frame sequence; no H264/H265/camera/network capture','readouts':['CP','Full'],'strides':[1,2],'cadence1_scope':'runtime-only sensitivity of frozen stride2 model with dt=1; accuracy/calibration not revalidated','checkpoint_sha256':{sc:sha(ROOT/f'runs/E11/seed42/{sc}/fold0/model.joblib') for sc in models},'scoring_source_sha256':sha(ROOT/'src/cycle_vad/streaming.py'),'runner_source_sha256':sha(Path(__file__))}
        write_json(OUT/('pilot_environment.json' if a.pilot else 'environment.json'),environment)
        if a.pilot:
            pass_once(encoder,models,selected[:1],2,'CP',-1,32,validate=True)
        else:
            config={'status_at_freeze':'configured_before_measurement','batch_size':1,'seed':42,'fold':0,'frames_per_file':128,'files':[r['id'] for r in selected],'warmup':100,'repetitions':3,'order':'rotate 4 conditions by repeat','paced_rates':[30,60],'paced_frames_per_file':128,'paced_sources':[next(r['id'] for r in selected if r['scene']==sc) for sc in models],'input_scope':environment['input_format'],'cadence1_scope':environment['cadence1_scope'],'no_feature_cache_in_timing':True}
            write_json(ROOT/'configs/experiments/followup/streaming_v1.json',config)
            conditions=[(2,'CP'),(1,'CP'),(2,'Full'),(1,'Full')]
            for repeat in range(3):
                for stride,readout in conditions[repeat:]+conditions[:repeat]:
                    tag=f'{readout}_stride{stride}_repeat{repeat}'
                    if a.resume and (OUT/'runs'/f'{tag}.json').exists():continue
                    pass_once(encoder,models,selected,stride,readout,repeat,128,validate=False)
            # Separate untimed parity replay, so score archival/checking never inflates FPS.
            parity=[next(r for r in selected if r['scene']==sc) for sc in models]
            for stride in [1,2]:
                for target in [30,60]:
                    tag=f'CP_stride{stride}_repeat0_paced{target}'
                    if a.resume and (OUT/'runs'/f'{tag}.json').exists():continue
                    pass_once(encoder,models,parity,stride,'CP',0,128,target_rate=target,validate=False)
            for stride in [1,2]:pass_once(encoder,models,parity,stride,'CP',99,128,validate=True)
