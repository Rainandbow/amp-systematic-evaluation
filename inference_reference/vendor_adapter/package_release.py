import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
               and p.name not in ("SHA256SUMS.txt","scheduler_console.log"))
with (root / "SHA256SUMS.txt").open("w",encoding="utf-8",newline="\n") as handle:
    for path in files:
        handle.write(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+path.relative_to(root).as_posix()+"\n")
files.append(root / "SHA256SUMS.txt")
target = Path("E:/北鲳/published_amp_six_by_six_v1_2.zip")
if target.exists():
    raise RuntimeError("Release ZIP already exists; preserve it")
with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        info = zipfile.ZipInfo(root.name+"/"+path.relative_to(root).as_posix(),date_time=(2026,10,10,0,0,0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info,path.read_bytes())
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
print("SHA256",hashlib.sha256(target.read_bytes()).hexdigest())
print("FILES",len(files),"SIZE",target.stat().st_size)
print(str(target))
