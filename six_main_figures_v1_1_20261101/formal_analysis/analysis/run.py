#!/usr/bin/env python3
"""Portable saved-output analyses. Run check, test, smoke, baseline, honest or all."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import common as C

def verify_package():
    p=ROOT/'package_manifest.json'
    if not p.is_file():raise RuntimeError('Missing package_manifest.json; use the complete delivered package')
    manifest=json.loads(p.read_text(encoding='utf-8'))
    for rel,h in manifest['files'].items():
        file=ROOT/rel
        if not file.is_file() or C.sha(file)!=h:raise RuntimeError('Package checksum mismatch: '+rel)
    return len(manifest['files'])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['check','test','smoke','baseline','honest','all'])
    ap.add_argument('--output',type=Path,help='New nonexisting output directory')
    ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--repeats',type=int,default=20)
    args=ap.parse_args()
    files=verify_package();C.verify_inputs()
    print('PACKAGE/INPUT HASHES PASS',files,'files; numpy',C.np.__version__,flush=True)
    if args.action in ['check','test']:
        panel,si,labels=C.load_panel();historical=C.load_historical()
        print('ALIGNMENT PASS 151 members; 906 saved model outputs; 29104 historical common-support records',flush=True)
        if args.action=='test':
            import selftest
            checks=selftest.run()
            for s in checks:print('TEST PASS',s,flush=True)
            print('SOFTWARE TESTS PASS',len(checks),flush=True)
        return
    smoke=args.action=='smoke'
    do_baseline=args.action in ['baseline','all','smoke']
    do_honest=args.action in ['honest','all','smoke']
    B=100 if smoke else args.bootstrap;R=2 if smoke else args.repeats
    if B<10000 and not smoke:raise ValueError('Use smoke for QA; formal baseline requires >=10000 resamples')
    if R!=20 and not smoke:raise ValueError('Formal honest protocol requires 20 group partitions')
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(os.getpid())
    out=(args.output or ROOT/'results'/(( 'SMOKE_' if smoke else 'FORMAL_')+suffix)).resolve()
    if out.exists():raise RuntimeError('Output already exists; choose a new --output directory')
    out.mkdir(parents=True)
    C.dump(out/'RUNNING.json',dict(action=args.action,smoke_only=smoke,bootstrap=B if do_baseline else 0,repeats=R if do_honest else 0,protocol_sha256=C.sha(ROOT/'protocol_v1.json')))
    begun=time.monotonic();result={}
    import selftest
    checks=selftest.run();C.dump(out/'software_checks.json',checks)
    if do_baseline:
        import baseline_family
        result['baseline']=baseline_family.run(out/'baseline_family',B,smoke)
    if do_honest:
        import honest_transfer
        result['honest']=honest_transfer.run(out/'honest_transfer',R,smoke)
    C.verify_inputs()
    C.dump(out/'COMPLETE.json',dict(action=args.action,smoke_only=smoke,bootstrap=B if do_baseline else 0,repeats=R if do_honest else 0,results=result,elapsed_seconds=time.monotonic()-begun,new_model_inference=False,new_wet_lab=False,protocol_sha256=C.sha(ROOT/'protocol_v1.json'),input_manifest_sha256=C.sha(ROOT/'input_manifest.json')))
    print('COMPLETE',out,flush=True)
    print('SMOKE OUTPUTS ARE SOFTWARE QA ONLY' if smoke else 'FORMAL SAVED-OUTPUT ANALYSES COMPLETE; inspect availability and audit tables before interpretation',flush=True)

if __name__=='__main__':main()
