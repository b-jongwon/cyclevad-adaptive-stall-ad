"""Make bounded publication chunks from an allowlisted artifact snapshot."""
import json, subprocess, sys
from pathlib import Path
stage=sys.argv[1]
files=json.loads(subprocess.check_output([sys.executable,'scripts/export_bundle.py'],text=True))
allowed={f'E{i}' for i in range(int(stage[1:])+1)}
files=[f for f in files if not (f['path'].startswith('results/E') and f['path'].split('/')[1] not in allowed)]
folder=Path('.cache/publish'); folder.mkdir(parents=True,exist_ok=True)
chunks=[]; current=[]; size=0
for f in files:
    n=len(json.dumps(f,ensure_ascii=False).encode())
    if current and size+n>240000: chunks.append(current); current=[]; size=0
    current.append(f); size+=n
if current: chunks.append(current)
for i,chunk in enumerate(chunks): (folder/f'{i:03d}.json').write_text(json.dumps(chunk,ensure_ascii=False))
print(json.dumps({'chunks':len(chunks),'files':len(files),'stage':stage}))
