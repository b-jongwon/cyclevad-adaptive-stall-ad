"""Build a local, blinded frame browser and contact sheets; no model labels."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from cycle_vad.data import frame_paths, write_json
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent/'IPAD_dataset'
OUT = ROOT/'runs/annotation_packet'
OUT.mkdir(parents=True, exist_ok=True)
queue = json.loads((ROOT/'configs/experiments/followup/annotation_queue.json').read_text())
items = []
for item in queue['items']:
    scene, part, sequence = item['id'].split('/')
    frames = frame_paths(DATA/scene/part/'frames'/sequence)
    samples = sorted(set(round(i*(len(frames)-1)/23) for i in range(24)))
    canvas = Image.new('RGB', (6*200, 4*162), 'white')
    draw = ImageDraw.Draw(canvas)
    for j, i in enumerate(samples):
        with Image.open(frames[i]) as im:
            thumb = ImageOps.contain(im.convert('RGB'), (196, 138))
        x, y = (j%6)*200, (j//6)*162
        canvas.paste(thumb, (x, y))
        draw.text((x+3, y+140), f'{item["id"]} frame {i}', fill='black')
    name = item['id'].replace('/', '_')+'.jpg'
    canvas.save(OUT/name, quality=88)
    items.append({'id': item['id'], 'frames': [p.as_uri() for p in frames], 'contact_sheet': name})
html = '''<!doctype html><meta charset="utf-8"><title>CycleVAD independent annotation packet</title>
<style>body{font:17px system-ui;max-width:1100px;margin:30px auto;background:#f5f7fa}img{max-width:100%;max-height:65vh}button,select,input{font:inherit;margin:8px}textarea{width:100%;height:160px}</style>
<h1>CycleVAD — 독립 주석 자료</h1><p>모델 phase·점수·이상 예측을 표시하지 않습니다. 프레임 번호는 0부터 시작합니다. 관측 가능한 반복 경계와 의미 있는 동작 전환을 기록하고 모호한 구간은 별도 표시하세요. 추정된 균등 phase를 정답으로 쓰지 마세요.</p>
<select id="video"></select><button onclick="step(-1)">이전 프레임</button><button onclick="step(1)">다음 프레임</button><br>
<input type="range" id="frame" min="0" value="0" style="width:70%"><span id="number"></span><br><img id="view"><br><a id="sheet" target="_blank">24프레임 접촉시트</a>
<p>주석 양식은 configs/experiments/followup/annotation_queue.json입니다. cycle_count, complete_cycles [시작, 끝-exclusive], anchors [{frame, name}], ambiguous_intervals, annotator를 기록하고 독립 검토자가 검토합니다. 단순 접촉시트만으로 경계를 확정하지 말고 전후 프레임을 확인하세요.</p>
<script>const items=ITEMS;const v=document.querySelector('#video'),f=document.querySelector('#frame');
items.forEach((x,i)=>v.add(new Option(x.id,i)));function show(){let x=items[+v.value];document.querySelector('#view').src=x.frames[+f.value];document.querySelector('#number').textContent=f.value+' / '+(x.frames.length-1);document.querySelector('#sheet').href=x.contact_sheet}
function change(){f.max=items[+v.value].frames.length-1;f.value=0;show()}function step(n){f.value=Math.max(0,Math.min(+f.max,+f.value+n));show()}v.onchange=change;f.oninput=show;document.onkeydown=e=>{if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1)};change();</script>'''
(OUT/'index.html').write_text(html.replace('ITEMS', json.dumps(items)), encoding='utf-8')
write_json(ROOT/'results/E6/annotation_packet.json', {'videos': len(items), 'contact_sheets': len(items),
    'viewer': 'runs/annotation_packet/index.html', 'annotations_completed': 0,
    'model_predictions_hidden': True, 'raw_frames_published': False, 'tracking_accuracy': None,
    'ids': [i['id'] for i in items], 'status': 'packet_prepared_annotations_required'})
print(f'Prepared {len(items)} local contact sheets and frame browser. No annotations fabricated.')
