"""Package already-audited local results, omitting rebuildable large caches."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE='amp_reliability_linked_v1_20261102.zip'
PREFIX='amp_reliability_linked_v1_20261102'

def main():
    qa=json.loads((ROOT/'qa/final_statistical_audit.json').read_text(encoding='utf8'))
    assert qa['passed']
    for stem in ['A_high_confidence_error_map','B_paired_added_distance_value','C_AMPlify_experimental_selection_utility']:
        for suffix in ['pdf','svg','png','tiff']:assert (ROOT/'figures'/f'{stem}.{suffix}').is_file()
        assert 'verdict: PASS' in (ROOT/'qa'/f'{stem}.audit_figure_collisions.py.log').read_text(encoding='utf8')
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(ROOT)
        if any(part in ['cache','python_deps','__pycache__'] for part in relative.parts):continue
        if p.name in [ARCHIVE,'MANIFEST_SHA256.json','PACKAGE_VERIFICATION.json']:continue
        if p.suffix=='.pyc':continue
        assert str(relative).isascii(),str(relative)
        files.append(p)
    files.sort(key=lambda p:p.relative_to(ROOT).as_posix())
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'MANIFEST_SHA256.json').write_text(json.dumps(dict(files=manifest,excluded=['cache/','python_deps/','__pycache__/'],purpose='local reproducibility; original third-party rights retained'),indent=2)+'\n',encoding='utf8')
    files.append(ROOT/'MANIFEST_SHA256.json')
    out=ROOT/ARCHIVE
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,PREFIX+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(PREFIX+'/'+name)).hexdigest()==digest,name
    receipt=dict(passed=True,archive=ARCHIVE,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),verified_files=len(manifest),all_archive_member_hashes_match=True,crc_pass=True,cache_excluded=True,all_member_names_ascii=True)
    (ROOT/'PACKAGE_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
