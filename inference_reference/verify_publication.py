from pathlib import Path
import json,hashlib,csv,ast
root=Path(__file__).resolve().parent
m=json.loads((root/'PUBLICATION_SHA256.json').read_text(encoding='utf-8'))
for n,h in m['files'].items():
 p=root/n
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h,n
for p in root.rglob('*.py'):
 s=p.read_text(encoding='utf-8-sig')
 if p.name=='Attention.py':
  import textwrap
  s=s.splitlines()[0]+'\n'+textwrap.dedent('\n'.join(s.splitlines()[1:]))
 ast.parse(s,filename=str(p))
with (root/'inputs/new_panel_queries.tsv').open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f,delimiter='\t'))
assert len(r)==169 and len({x['sequence_id'] for x in r})==169
assert all(x['sequence_id']=='s_'+hashlib.sha256(x['sequence'].encode('ascii')).hexdigest() for x in r)
print('INFERENCE PUBLICATION CHECK PASS',len(m['files']),'files; 169 identity-verified inputs; no model inference executed')
