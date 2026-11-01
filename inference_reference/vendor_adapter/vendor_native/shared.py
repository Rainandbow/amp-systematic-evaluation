"""Integrity, ID alignment and read-only feature access (Python >=3.6)."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

VERSION = 'amp_amplify_ml_tools_v1.0'
REPS = ['ESMC_6B', 'ESM2_15B']
HEADS = ['LR', 'Mean_MLP', 'RBF_SVM', 'RF', 'XGBoost']
TOOLS = ['Macrel', 'AMPlify', 'ampir', 'tsAMP_I', 'HMD_AMP', 'c_AMPs']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(str(temp), str(path))


def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    os.replace(str(temp), str(path))


def write_fasta(path, rows):
    with Path(path).open('w', encoding='ascii', newline='\n') as f:
        for r in rows:
            f.write('>' + r['sequence_id'] + '\n' + r['sequence'] + '\n')


def verify_files(root, files):
    root = Path(root).resolve()
    require(bool(files), 'Empty integrity manifest')
    for name, digest in files.items():
        p = (root / name).resolve()
        p.relative_to(root)
        require(sha(p) == digest, 'Checksum changed: ' + str(p))


def verify_source(plan):
    source = Path(plan['source_work'])
    manifest = read_json(source / 'dataset_complete.json')
    require(manifest['experiment_version'] == 'external_plm_amplify_v1.0', 'Wrong source experiment')
    verify_files(source, manifest['files'])
    complete = read_json(source / 'experiment_complete.json')
    require(complete['complete'] and complete['completed_heads'] == 8, 'Source run is not complete')
    require(complete['dataset_manifest_sha256'] == sha(source / 'dataset_complete.json'), 'Source dataset mismatch')
    require(complete['results_sha256'] == sha(source / '07_reports/final_results.tsv'), 'Source results changed')
    rows = read_tsv(source / '04_datasets/AMPlify/all.tsv')
    require(len(rows) == len({r['sequence_id'] for r in rows}), 'Duplicate sequence IDs')
    require(len(rows) == len({r['sequence'] for r in rows}), 'Duplicate sequences')
    for r in rows:
        require(r['sequence'] and not set(r['sequence']) - set('ACDEFGHIKLMNPQRSTVWY'), 'Invalid sequence')
        require(r['sequence_id'] == 's_' + hashlib.sha256(r['sequence'].encode('ascii')).hexdigest(), 'Sequence hash mismatch')
        require(int(r['label']) in (0, 1) and r['split'] in ('train', 'validation', 'test'), 'Invalid partition/label')
        require(int(r['length']) == len(r['sequence']), 'Wrong sequence length')
    counts = {s: {str(y): sum(r['split'] == s and int(r['label']) == y for r in rows) for y in (0, 1)}
              for s in ('train', 'validation', 'test')}
    require(counts == plan['expected_counts'], 'Source counts changed')
    for s in counts:
        separate = read_tsv(source / ('04_datasets/AMPlify/' + s + '.tsv'))
        require(separate == [r for r in rows if r['split'] == s], 'all/split membership or order differs')
    groups = {s: {r['group_id'] for r in rows if r['split'] == s} for s in counts}
    require(not groups['train'] & groups['validation'], 'Train/validation detected group overlap')
    return rows


def verified_head(source, rep):
    directory = Path(source) / '06_heads/AMPlify' / rep
    stamp = read_json(directory / 'complete.json')
    require(stamp['inputs']['representation'] == rep, 'Wrong original classifier')
    require(stamp['inputs']['dataset_manifest_sha256'] == sha(Path(source) / 'dataset_complete.json'), 'Head dataset mismatch')
    require(stamp['inputs']['dataset_sha256'] == sha(Path(source) / '04_datasets/AMPlify/all.tsv'), 'Head cohort mismatch')
    verify_files(directory, stamp['files'])
    return stamp


def vectors(source, rep, ids):
    """Never call cache_index: it can write a protocol file in the source."""
    import numpy as np
    directory = Path(source) / '05_embeddings' / rep
    stamp = read_json(directory / 'cache_protocol.json')
    require(fingerprint(stamp['inputs']) == stamp['fingerprint'], 'Invalid cache fingerprint')
    head_stamp = verified_head(source, rep)
    require(head_stamp['inputs']['embedding_fingerprint'] == stamp['fingerprint'], 'Wrong encoder/cache for original LR')
    complete = read_json(directory / 'AMPlify_complete.json')
    require(complete['fingerprint'] == stamp['fingerprint'] and complete['n'] == len(ids), 'Incomplete source features')
    require(complete['dataset_sha256'] == sha(Path(source) / '04_datasets/AMPlify/all.tsv'), 'Features cohort differs')
    index, metadata = {}, {}
    for marker in sorted(directory.glob('shard_*.json')):
        m = read_json(marker)
        require(m['fingerprint'] == stamp['fingerprint'] and m['dimension'] == stamp['dimension'], 'Wrong shard')
        require(len(m['sequence_ids']) == m['n'], 'Wrong shard row count')
        archive = marker.with_suffix('.npz')
        require(sha(archive) == m['sha256'], 'Corrupt committed feature shard')
        metadata[archive] = m
        for i, uid in enumerate(m['sequence_ids']):
            require(uid not in index, 'Duplicate committed feature ID')
            index[uid] = (archive, i)
    require(len(set(ids)) == len(ids) and all(uid in index for uid in ids), 'Missing/duplicate requested vector')
    result = np.empty((len(ids), stamp['dimension']), dtype=np.float32)
    selections = {}
    for row, uid in enumerate(ids):
        archive, i = index[uid]
        selections.setdefault(archive, []).append((row, i, uid))
    for archive, selected in selections.items():
        m = metadata[archive]
        with np.load(archive, allow_pickle=False) as data:
            x, saved_ids = data['embeddings'], data['sequence_ids']
            require(x.shape == (m['n'], stamp['dimension']) and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid vector matrix')
            require(list(map(str, saved_ids)) == m['sequence_ids'], 'Feature row IDs changed')
            for row, i, uid in selected:
                require(str(saved_ids[i]) == uid, 'Vector ID mismatch')
                result[row] = x[i]
        require(sha(archive) == m['sha256'], 'Shard changed during reading')
    return result, stamp


def align_predictions(records, expected):
    import math
    require(len(records) == len(expected), 'Prediction row coverage mismatch')
    by_id = {}
    for r in records:
        uid = r['sequence_id']
        require(uid not in by_id, 'Duplicate prediction ID')
        p = float(r['probability'])
        require(math.isfinite(p) and 0 <= p <= 1 and int(r['prediction']) in (0, 1), 'Invalid probability/prediction')
        by_id[uid] = r
    require(set(by_id) == {r['sequence_id'] for r in expected}, 'Missing/extra prediction IDs')
    result = []
    for r in expected:
        saved = by_id[r['sequence_id']]
        require(saved['sequence'] == r['sequence'] and int(saved['label']) == int(r['label']) and saved['split'] == r['split'], 'ID-aligned metadata differs')
        result.append(saved)
    return result


def vendor(name):
    root = Path(__file__).resolve().parent
    inventory = read_json(root / 'vendor_manifest.json')
    require(name in inventory, 'Unpinned vendor file')
    path = root / 'vendor' / name
    require(sha(path) == inventory[name], 'Uploaded reference adapter changed: ' + name)
    spec = importlib.util.spec_from_file_location('vendor_' + name.replace('.', '_'), str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eligible(tool, row):
    """Prespecified adapter support, not an assertion of biological validity."""
    n = len(row['sequence'])
    if tool == 'AMPlify':
        return 2 <= n <= 200
    if tool in ('Macrel', 'ampir'):
        return n >= 10  # retain the safety gate of the supplied, tested adapters
    if tool in ('tsAMP_I', 'HMD_AMP'):
        return 1 <= n <= 1022  # no truncation or alteration
    return False  # complete c_AMPs integration is not yet verified


PRED_FIELDS = ['sequence_id', 'sequence', 'group_id', 'split', 'label', 'probability', 'prediction']


def completed(directory, expected):
    marker = Path(directory) / 'complete.json'
    if not marker.exists():
        return False
    m = read_json(marker)
    require(m['inputs'] == expected, 'Completed run protocol changed; use a new version')
    verify_files(directory, m['files'])
    return True


def finish(directory, expected, files):
    directory = Path(directory)
    dump(directory / 'complete.json', {'inputs': expected, 'files': {n: sha(directory / n) for n in files}})
