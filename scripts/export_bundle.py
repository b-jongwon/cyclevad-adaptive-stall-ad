"""Export only reviewable repository artifacts, never frames, caches or weights."""
import json
from pathlib import Path
paths=[Path('.gitignore'),Path('pyproject.toml'),Path('README.md'),Path('requirements-local.txt'),Path('requirements-cycle-colab.txt')]
for folder in ['src','scripts','tests','configs','docs','results','notebooks']:
    paths.extend(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix in {'.py','.md','.json','.csv','.svg','.txt','.ipynb','.toml'} and '__pycache__' not in p.parts)
files=[]
for p in sorted(set(paths)):
    if p.exists(): files.append({'path':p.as_posix(),'mode':'100644','type':'blob','content':p.read_text(encoding='utf-8')})
print(json.dumps(files,ensure_ascii=False))
