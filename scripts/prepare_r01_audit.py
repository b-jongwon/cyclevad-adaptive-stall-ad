"""Prepare a blinded R01 review packet; never author human judgments."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from cycle_vad.data import write_json,frame_paths
from cycle_vad.paper_calibration import load_bank,read
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/r01_audit';OUT.mkdir(parents=True,exist_ok=True)
rel=Path('seed42/R01/fold0');r=read(ROOT/'results/E11B'/rel/'run.json');bank=load_bank(ROOT,rel,True);rng=np.random.default_rng(20261009)
pools={}
for part in ['reference','threshold','normal_evaluation']:
    pools[part]={i:np.flatnonzero(bank[i]['labels']==0) for i in r['splits'][part]}
for name,after in [('historical_before',False),('historical_after',True)]:
    pools[name]={}
    for i,d in bank.items():
        if '/testing/' not in i:continue
        mask=(d['labels']==0)&((np.cumsum(d['labels']==1)>0)==after)
        if mask.any():pools[name][i]=np.flatnonzero(mask)
queue=[]
for part,pool in pools.items():
    ids=sorted(pool);chosen=rng.choice(ids,min(5,len(ids)),replace=False).tolist()
    for i in chosen:
        center=int(rng.choice(pool[i]));queue.append({'cohort':'uniform','stratum':part,'source_id':i,'center':center})
    # Separate score-enriched cohort: two source-level low/high median tails.
    ordered=sorted(ids,key=lambda i:float(np.median(bank[i]['scores'][pool[i],3])))
    for tail,selected in [('low',ordered[:2]),('high',ordered[-2:])]:
        for i in selected:
            candidates=pool[i];vals=bank[i]['scores'][candidates,3];target=np.quantile(vals,.1 if tail=='low' else .9)
            center=int(candidates[np.argmin(abs(vals-target))]);queue.append({'cohort':'score_enriched','stratum':part,'tail':tail,'source_id':i,'center':center})
rng.shuffle(queue);items=[]
for j,q in enumerate(queue):
    q['review_id']=f'Q{j+1:03d}';d=bank[q['source_id']];lo=max(0,q['center']-32);hi=min(len(d['labels']),q['center']+33);q.update(start=lo,end=hi)
    sc,part,seq=q['source_id'].split('/');frames=frame_paths(ROOT.parent/'IPAD_dataset'/sc/part/'frames'/seq)
    canvas=Image.new('RGB',(4*220,3*175),'white');draw=ImageDraw.Draw(canvas)
    for k,f in enumerate(np.linspace(lo,hi-1,12).round().astype(int)):
        with Image.open(frames[f]) as im:thumb=ImageOps.contain(im.convert('RGB'),(216,148))
        x=(k%4)*220;y=(k//4)*175;canvas.paste(thumb,(x,y));draw.text((x+3,y+150),f'{q["review_id"]} relative frame {f-lo}',fill='black')
    sheet=q['review_id']+'.jpg';canvas.save(OUT/sheet,quality=88)
    items.append({'id':q['review_id'],'frames':[p.as_uri() for p in frames[lo:hi]],'sheet':sheet})
html='''<!doctype html><meta charset="utf-8"><title>R01 blinded source review</title><style>body{font:17px system-ui;max-width:1100px;margin:28px auto}select,button,input{font:inherit;margin:8px}img{max-width:100%;max-height:70vh}</style><h1>R01 원본 문맥 검토</h1><p>모델 점수·예측·라벨·선정 이유는 표시하지 않습니다. 조명/시점/배경, 공정 상태, 잔류 물체·결함, 판독 불확실성을 기록하세요. 각 검토자는 상대방의 기록을 보지 않고 독립 작성합니다. AI 판독을 인간 주석으로 기록하지 마세요.</p><select id="q"></select><button id="prev">이전</button><button id="next">다음</button><input id="t" type="range" min="0" value="0"><span id="n"></span><br><img id="im"><br><a id="sheet" target="_blank">문맥 접촉시트</a><script>const items=ITEMS;const q=document.querySelector('#q'),t=document.querySelector('#t');items.forEach((x,i)=>q.add(new Option(x.id,i)));function show(){const x=items[+q.value];document.querySelector('#im').src=x.frames[+t.value];document.querySelector('#n').textContent='relative '+t.value+' / '+(x.frames.length-1);document.querySelector('#sheet').href=x.sheet;}function change(){t.max=items[+q.value].frames.length-1;t.value=0;show()}function step(d){t.value=Math.max(0,Math.min(+t.max,+t.value+d));show()}q.onchange=change;t.oninput=show;document.querySelector('#prev').onclick=()=>step(-1);document.querySelector('#next').onclick=()=>step(1);change();</script>'''
(OUT/'index.html').write_text(html.replace('ITEMS',json.dumps(items)))
fields=['lighting_view_background','process_state','visible_residual_defect','temporal_boundary_notes','uncertainty','notes']
for name in ['A','B']:
    p=OUT/f'annotator_{name}.json'
    if not p.exists():write_json(p,{'annotator':None,'status':'not_started','items':[{'review_id':x['id'],**{k:None for k in fields}} for x in items]})
write_json(ROOT/'configs/experiments/followup/r01_audit_queue.json',{'sampling_seed':20261009,'status':'packet_prepared_human_review_required','scoring_checkpoint':'seed42/R01/fold0','uniform_target_sources_per_stratum':5,'cohorts_must_not_be_pooled_for_prevalence':True,'queue':queue})
write_json(ROOT/'results/E15BC/r01_audit_packet.json',{'status':'packet_prepared_human_review_required','clips':len(items),'uniform_clips':sum(q['cohort']=='uniform' for q in queue),'score_enriched_clips':sum(q['cohort']=='score_enriched' for q in queue),'independent_human_reviews_completed':0,'viewer':'runs/r01_audit/index.html','original_images_published':False})
print(f'Prepared {len(items)} blinded local clips; 0 human reviews completed.')
