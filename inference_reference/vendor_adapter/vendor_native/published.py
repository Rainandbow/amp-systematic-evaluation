"""Pinned published inference in each existing environment. Python >=3.6.

No installs, downloads, checkpoint substitutions, or silent missing predictions.
The upload's pinned adapter functions are reused, not their old cohort mains.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

import shared as s


def limits(gpu):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu or '-1'
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])


def file_inventory(root):
    root = Path(root)
    files = {}
    for path in root.rglob('*'):
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            files[str(path.resolve())] = s.sha(path)
    s.require(files, 'Missing installed package files: ' + str(root))
    return files


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def identify(tool, plan):
    """Read-only package/weight checks before trusted released deserialization."""
    spec = plan['tools'][tool]
    if tool == 'Macrel':
        import macrel
        import macrel.macrel_version as v
        import onnxruntime
        s.require(v.__version__ == '1.6.1', 'Need Macrel 1.6.1')
        return {'version': v.__version__, 'onnxruntime': onnxruntime.__version__,
                'files': file_inventory(Path(macrel.__file__).parent)}
    if tool == 'AMPlify':
        a = s.vendor('04_predict_amplify_cpu.py')
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        import tensorflow as tf
        import keras
        import numpy as np
        s.require(tf.__version__ == '1.12.0' and keras.__version__ == '2.2.4', 'Need existing AMPlify legacy TF1.12/Keras2.2.4 environment')
        s.require(np.__version__.split('.')[:2] == ['1', '16'], 'Need AMPlify NumPy 1.16')
        source = Path(sys.prefix) / 'share/amplify/src/AMPlify.py'
        files = {}
        for name, digest in a.SOURCE_SHA.items():
            path = source.parent / name
            s.require(s.sha(path) == digest, 'Official AMPlify source changed')
            files[str(path.resolve())] = digest
        for i in range(1, 6):
            path = source.parent.parent / ('models/balanced/AMPlify_balanced_model_weights_{}.h5'.format(i))
            s.require(path.stat().st_size > 1024, 'Missing AMPlify weights')
            files[str(path.resolve())] = s.sha(path)
        packages = list((Path(sys.prefix) / 'conda-meta').glob('amplify-2.0.1-*.json'))
        s.require(len(packages) == 1 and s.read_json(packages[0])['version'] == '2.0.1', 'Wrong AMPlify package')
        files[str(packages[0].resolve())] = s.sha(packages[0])
        return {'version': '2.0.1', 'variant': 'balanced_five_submodel_mean', 'tensorflow': tf.__version__,
                'keras': keras.__version__, 'source': str(source), 'files': files}
    if tool == 'tsAMP_I':
        a = s.vendor('08_smoke_tsamp_released.py')
        from importlib.metadata import version
        import torch
        s.require(version('fair-esm') == '2.0.0' and version('biopython') == '1.81', 'Wrong isolated tsAMP versions')
        root = Path(spec['resources'])
        a.verify_resources(root / 'official')
        backbone = root / (a.BACKBONE + '.pt')
        s.require(backbone.stat().st_size == a.BACKBONE_BYTES, 'Wrong tsAMP ESM1v checkpoint size')
        digest = s.sha(backbone)
        s.require(digest == '9519ee60f1cddad3c101afb1f42612499e188534969c3f682e94850870f70433', 'Wrong tsAMP backbone hash')
        files = {str((root / 'official' / n).resolve()): h for n, (_, h) in a.RESOURCES.items()}
        files[str(backbone.resolve())] = digest
        return {'commit': a.COMMIT, 'torch': torch.__version__, 'files': files,
                'backbone': str(backbone), 'reference_feature_parity_required': True}
    if tool == 'HMD_AMP':
        a = s.vendor('09_audit_hmd_amp_released.py')
        versions = a.runtime()
        root = Path(spec['resources'])
        a.verify_resources(root)
        backbone = root / (a.BACKBONE + '.pt')
        s.require(backbone.stat().st_size == 2604537549, 'Wrong HMD ESM2-650M checkpoint size')
        digest = s.sha(backbone)
        s.require(digest == 'ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c', 'Wrong HMD backbone hash')
        audit = s.read_json(root / 'classifier_audit.json')
        s.require(audit['versions'] == versions and audit['weight_files_sha256'] == a.WEIGHTS and audit['classes'] == [0, 1]
                  and audit['synthetic_feature_sanity_passed'], 'Prior HMD classifier audit differs')
        files = {str((root / 'official' / n).resolve()): h for n, h in a.FILES.items()}
        files.update({str((root / 'released' / n).resolve()): h for n, h in a.WEIGHTS.items()})
        files[str(backbone.resolve())] = digest
        files[str((root / 'classifier_audit.json').resolve())] = s.sha(root / 'classifier_audit.json')
        return {'commit': a.COMMIT, 'versions': versions, 'files': files, 'backbone': str(backbone)}
    raise ValueError('No complete verified adapter for ' + tool)


def macrel(rows, directory):
    import pandas as pd
    import onnxruntime as ort
    from macrel.main import main
    a = s.vendor('02_prepare_and_compare_published_amp.py')
    query = directory / 'query.fasta'
    s.write_fasta(query, rows)
    output = directory / 'native'
    tmp = directory / 'tmp'
    tmp.mkdir()
    original, argv = ort.InferenceSession, sys.argv
    def session(model, sess_options=None, **kwargs):
        opts = sess_options or ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = 2, 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return original(model, sess_options=opts, **kwargs)
    ort.InferenceSession = session
    sys.argv = ['macrel', 'peptides', '--fasta', str(query), '--output', str(output),
                '--keep-negatives', '--threads', '2', '--tag', 'benchmark', '--tmpdir', str(tmp)]
    try:
        main()
    finally:
        sys.argv, ort.InferenceSession = argv, original
    expected = pd.DataFrame([{'seq_uid': r['sequence_id'], 'sequence': r['sequence'], 'label': int(r['label'])} for r in rows])
    result = a.macrel_frame(output / 'benchmark.prediction.gz', expected)
    return result.probability.to_numpy(float), result.prediction.to_numpy(int), {'native_leading_M_removal': True}


def amplify(rows, directory, identity):
    import pandas as pd
    a = s.vendor('04_predict_amplify_cpu.py')
    query = directory / 'query.fasta'
    s.write_fasta(query, rows)
    runtime = a.execute_native(Path(identity['source']), query, directory)
    outputs = list(directory.glob('AMPlify_balanced_results_*.tsv'))
    s.require(len(outputs) == 1, 'Missing/ambiguous native AMPlify output')
    expected = pd.DataFrame([{'seq_uid': r['sequence_id'], 'sequence': r['sequence'], 'label': int(r['label'])} for r in rows])
    result = a.amplify_frame(outputs[0], expected)
    return result.probability.to_numpy(float), result.prediction.to_numpy(int), runtime


def tsamp(rows, directory, identity, plan):
    import argparse
    import numpy as np
    import pandas as pd
    import torch
    import esm
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    a = s.vendor('08_smoke_tsamp_released.py')
    root = Path(plan['tools']['tsAMP_I']['resources']) / 'official'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    s.require(torch.cuda.is_available(), 'GPU unavailable')
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(identity['backbone'], map_location='cpu', weights_only=True)
    encoder, alphabet = esm.pretrained.load_model_and_alphabet_core(a.BACKBONE, state)
    del state
    encoder.float().eval().to('cuda')
    convert = alphabet.get_batch_converter()
    def embedding(uid, sequence):
        _, _, tokens = convert([(uid, sequence)])
        with torch.inference_mode():
            out = encoder(tokens.to('cuda'), repr_layers=[33], return_contacts=False)
        return out['representations'][33][0, 1:len(sequence)+1].mean(0).cpu()
    references = dict(a.fasta_records(root / 'train_amp.fasta'))
    references['L01A000044'] = 'GIGAAILSAGKSIIKGLANGLAEHF'
    audit = []
    for uid in ('DBAASPN_10400', 'DBAASPN_10401', 'L01A000044'):
        expected = torch.load(root / (uid + '.pt'), map_location='cpu', weights_only=True)['mean_representations'][33].float()
        actual = embedding(uid, references[uid])
        passed = actual.shape == expected.shape == (1280,) and torch.allclose(actual, expected, atol=2e-4, rtol=1e-4)
        audit.append({'id': uid, 'passed': bool(passed), 'max_abs_difference': float((actual-expected).abs().max())})
    s.dump(directory / 'reference_parity.json', audit)
    s.require(all(r['passed'] for r in audit), 'Official tsAMP reference feature parity failed')
    formula = a.property_functions(root / 'charge.py')
    def properties(seq):
        obj = ProteinAnalysis(seq)
        return [formula['calculate_hydrophobicity'](seq), round(obj.charge_at_pH(7.4)),
                obj.isoelectric_point(), formula['calculate_amphipathicity'](seq)]
    sheet = pd.read_excel(root / 'testLAMP2.xlsx')
    reference = sheet[sheet.iloc[:, 0].astype(str).str.split('|').str[0] == 'L01A000044']
    s.require(len(reference) == 1 and reference.iloc[0, 1] == references['L01A000044'] and np.allclose(
        properties(references['L01A000044']), reference.iloc[0, 2:6].to_numpy(float), atol=1e-7, rtol=1e-7), 'tsAMP property parity failed')
    model = load_file(root / 'model.py', 'tsamp_native_model').MLP(1284, 256, 1)
    model.load_state_dict(torch.load(root / 'tsAMP-I.pt', map_location='cpu', weights_only=True), strict=True)
    model.float().eval().to('cuda')
    p = []
    for r in rows:
        feature = torch.cat((embedding(r['sequence_id'], r['sequence']), torch.tensor(properties(r['sequence']), dtype=torch.float32)))
        s.require(feature.shape == (1284,) and torch.isfinite(feature).all(), 'Invalid tsAMP features')
        with torch.inference_mode():
            p.append(torch.sigmoid(model(feature.to('cuda')[None])).item())
        if len(p) == 1 or len(p) % 100 == 0 or len(p) == len(rows):
            print('tsAMP predicted {}/{}'.format(len(p), len(rows)), flush=True)
    p = np.asarray(p, float)
    return p, (p > 0.5).astype(int), {'reference_feature_parity_passed': True, 'batch_size': 1, 'precision': 'FP32_TF32_off'}


def hmd(rows, directory, identity, plan):
    import argparse
    import numpy as np
    import torch
    import esm
    a = s.vendor('09_audit_hmd_amp_released.py')
    s.require(torch.cuda.is_available(), 'GPU unavailable')
    root = Path(plan['tools']['HMD_AMP']['resources'])
    classifier = a.load_classifier(root)
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(identity['backbone'], map_location='cpu', weights_only=True)
    encoder, alphabet = esm.pretrained.load_model_and_alphabet_core(a.BACKBONE, state)
    del state
    encoder.float().eval()
    sys.path.insert(0, str(root / 'official'))
    author = load_file(root / 'official/src/utils.py', 'hmd_native_utils')
    ps, decisions = [], []
    # Native batch_size=64 remains unchanged. Chunking limits simultaneous
    # query padding/memory; there is no sequence cropping or model modification.
    for offset in range(0, len(rows), 16):
        subset = rows[offset:offset+16]
        query = directory / ('chunk_{:05d}.fasta'.format(offset))
        s.write_fasta(query, subset)
        with a.local_backbone_factory(esm, encoder, alphabet):
            features, _, ids = author.amp_feature_extraction(str(root / 'released/AMP/ft_parts.pth'), 'cuda', str(query))
        s.require(list(ids) == [r['sequence_id'] for r in subset] and features.shape == (len(subset), 320) and np.isfinite(features).all(), 'HMD features coverage mismatch')
        if offset == 0:
            with a.local_backbone_factory(esm, encoder, alphabet):
                repeated, _, repeated_ids = author.amp_feature_extraction(str(root / 'released/AMP/ft_parts.pth'), 'cuda', str(query))
            s.require(list(repeated_ids) == list(ids) and np.allclose(features, repeated, atol=2e-5, rtol=1e-5), 'HMD eval repeatability failed')
        p, d = a.checked_scores(classifier, features)
        ps.extend(map(float, p))
        decisions.extend(map(int, d))
        print('HMD predicted {}/{}'.format(len(ps), len(rows)), flush=True)
    return np.asarray(ps), np.asarray(decisions), {'unchanged_author_extractor': True,
        'query_chunk_size': 16, 'native_batch_size': 64, 'repeatability_passed': True, 'precision': 'FP32_TF32_off'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--tool', required=True, choices=['Macrel', 'AMPlify', 'tsAMP_I', 'HMD_AMP'])
    parser.add_argument('--action', required=True, choices=['check', 'predict'])
    parser.add_argument('--stage', choices=['smoke', 'full'])
    parser.add_argument('--gpu', default='')
    args = parser.parse_args()
    limits(args.gpu)
    plan = s.read_json(args.plan)
    work = Path(plan['work'])
    identity = identify(args.tool, plan)
    if args.action == 'check':
        s.dump(work / '00_checks' / (args.tool + '.json'), identity)
        print('READ-ONLY TOOL CHECK PASSED:', args.tool, flush=True)
        return
    frozen = s.read_json(work / '00_checks' / (args.tool + '.json'))
    s.require(identity == frozen, 'Tool/runtime/weights changed after preparation')
    rows = s.read_tsv(work / '01_inputs' / args.tool / 'queries.tsv')
    s.require(args.stage is not None, 'Missing stage')
    if args.stage == 'smoke':
        rows = [r for r in rows if r['split'] == 'validation'][:10]
    s.require(rows, 'No supported queries')
    inputs = {'tool_identity': identity, 'stage': args.stage, 'protocol_sha256': s.sha(work / 'protocol.json'),
              'queries_sha256': s.sha(work / '01_inputs' / args.tool / 'queries.tsv')}
    parent = work / '03_tools' / args.tool
    directory = parent / args.stage
    if s.completed(directory, inputs):
        print('SKIP VERIFIED TOOL:', args.tool, args.stage, flush=True)
        return
    if args.stage == 'full':
        gate = s.read_json(parent / 'smoke/complete.json')
        s.require(gate['inputs']['tool_identity'] == identity and gate['inputs']['protocol_sha256'] == inputs['protocol_sha256'], 'Missing/changed smoke gate')
        s.verify_files(parent / 'smoke', gate['files'])
    # Failed attempts remain separately archived in place; no overwrite/deletion.
    if directory.exists():
        preserved = directory.with_name(directory.name + '_failed_' + uuid.uuid4().hex[:12])
        directory.rename(preserved)
        print('Preserved incomplete attempt:', preserved, flush=True)
    directory.mkdir(parents=True)
    print('PUBLISHED START:', args.tool, args.stage, len(rows), flush=True)
    start = time.monotonic()
    if args.tool == 'Macrel':
        p, d, details = macrel(rows, directory)
    elif args.tool == 'AMPlify':
        p, d, details = amplify(rows, directory, identity)
    elif args.tool == 'tsAMP_I':
        p, d, details = tsamp(rows, directory, identity, plan)
    else:
        p, d, details = hmd(rows, directory, identity, plan)
    result = []
    for r, prob, decision in zip(rows, p, d):
        out = {k: r[k] for k in s.PRED_FIELDS if k not in ('probability', 'prediction')}
        out.update(probability=float(prob), prediction=int(decision))
        result.append(out)
    s.require(len(p) == len(d) == len(rows), 'Wrong prediction coverage')
    s.align_predictions(result, rows)
    for path, digest in identity['files'].items():
        s.require(s.sha(path) == digest, 'Released source or weights changed during prediction')
    s.write_tsv(directory / 'predictions.tsv', result, s.PRED_FIELDS)
    s.dump(directory / 'run_manifest.json', {'tool': args.tool, 'seconds': time.monotonic()-start,
        'identity': identity, 'details': details, 'inference_only': True, 'training_overlap': 'not proved independent'})
    artifacts = [p.relative_to(directory).as_posix() for p in directory.rglob('*')
                 if p.is_file() and 'tmp' not in p.relative_to(directory).parts and p.name != 'complete.json']
    s.finish(directory, inputs, artifacts)
    print('PUBLISHED COMPLETE:', args.tool, args.stage, flush=True)


if __name__ == '__main__':
    main()
