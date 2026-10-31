"""Reuse frozen real score diagnostics, preserving original budget/statistics functions.
No fabricated raw predictions, model fitting, inference, chemical conversion, or wet lab.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,importlib.util,json,sys,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'rebuild_analyses/membership'
BASE=ROOT/'frozen/budget153'
EXT=ROOT/'frozen/experimental_inputs'
AUDIT=ROOT/'frozen/evidence_adjudication'
SPEC=importlib.util.spec_from_file_location('original_budget',Path(__file__).with_name('original_budget_task_v1.py'))
core=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(core)

EXCLUDE={'HydrAMP__Pex-P1-4','HydrAMP__Temp-P1-5'}
ALL_PATHS={'candidates':EXT/'inputs/new_panel_candidates.tsv','assays':EXT/'inputs/new_panel_assays.tsv','frozen_score_diagnostics':BASE/'sequence_score_diagnostics.tsv','old_run_settings':BASE/'run_settings.json','adjudicated_membership':AUDIT/'01_成员与活动标签_核定总表.tsv','active_design_151':AUDIT/'08_新增151活性设计序列_排除定向阴性对照.tsv','old_budget_points':BASE/'budget_points.tsv','old_budget_intervals':BASE/'budget_intervals.tsv','old_pairwise_primary':BASE/'paired_primary_budget_differences.tsv'}

# Verify hashes of the real downloaded server outputs before deriving anything.
def verified_load(scope):
    paths=dict(ALL_PATHS);settings=json.loads(paths['old_run_settings'].read_text(encoding='utf-8'))
    assert settings['smoke_only'] is False and settings['synthetic_fixture'] is False
    assert core.sha(paths['candidates'])==settings['input_manifest']['candidates']['sha256']
    assert core.sha(paths['assays'])==settings['input_manifest']['assays']['sha256']
    for key,name in [('frozen_score_diagnostics','sequence_score_diagnostics.tsv'),('old_budget_points','budget_points.tsv'),('old_budget_intervals','budget_intervals.tsv'),('old_pairwise_primary','paired_primary_budget_differences.tsv')]:
        assert core.sha(paths[key])==settings['output_files'][name],name
    assert settings['seed']==core.SEED and settings['budgets']==core.BUDGETS
    cc=core.read(paths['candidates']);aa=core.read(paths['assays']);ss=core.read(paths['frozen_score_diagnostics'])
    audit={r['record_id']:r for r in core.read(paths['adjudicated_membership']) if r['panel']=='new'}
    accepted={r['record_id'] for r in core.read(paths['active_design_151'])}
    generated=[c for c in cc if core.yes(c['primary_candidate'])]
    assert len(generated)==153 and len(accepted)==151 and {c['record_id'] for c in generated}-accepted==EXCLUDE
    index={(s['sequence_id'],s['model']):s for s in ss};assert len(index)==len(ss)==153*6
    gids={};sm={};pred={}
    for c in generated:
        sid=c['sequence_id'];assert sid=='s_'+hashlib.sha256(c['sequence'].encode('ascii')).hexdigest()
        assert audit[c['record_id']]['activity_label_binary']==c['panel_label_any_tested_strain']
        for m in core.MODELS:
            d=index[sid,m];assert d['cohort']==c['cohort'] and int(d['label'])==int(c['panel_label_any_tested_strain'])
            assert int(d['length'])==int(c['length']) and 0<=float(d['score'])<=1 and int(d['prediction']) in (0,1)
            if sid in gids:assert gids[sid]==d['block'] and sm[sid]==core.yes(d['inventory_pass'])
            gids[sid]=d['block'];sm[sid]=core.yes(d['inventory_pass'])
            expected='exploratory_min_branch_extension' if m=='c_AMPs' else 'saved_AMP_class_score'
            assert d['score_definition']==expected
            pred[sid,m]=dict(score=d['score'],prediction=d['prediction'],supported='1',derived_score_definition=d['score_definition'])
    if scope=='151':
        for c in cc:
            if c['record_id'] in EXCLUDE:c['primary_candidate']='False'
    assert sum(core.yes(c['primary_candidate']) for c in cc)==int(scope)
    # Explicit references, not a substitute inference completion stamp.
    return cc,aa,gids,sm,pred,paths

# A provenance-explicit adapter to the old evaluator: values are frozen derived scores.
# For c_AMPs use the already verified min-branch extension exactly as archived.
# Do not manufacture its branch outputs or call it an ensemble probability.
def frozen_score(row,model):
    assert row['derived_score_definition']==('exploratory_min_branch_extension' if model=='c_AMPs' else 'saved_AMP_class_score')
    return float(row['score'])

def run(scope,B):
    core.load=lambda unused:verified_load(scope)
    core.score_for=frozen_score
    # The evaluator expects complete.json only to label a synthetic fixture in settings.
    # Replace that metadata lookup explicitly; no inference stamp is manufactured.
    source=Path(__file__).with_name('original_budget_task_v1.py').read_text(encoding='utf-8')
    start=source.index('def evaluate(args):');end=source.index('\ndef check(args):',start)
    body=source[start:end].replace("'synthetic_fixture':json.loads(paths['complete'].read_text(encoding='utf8')).get('software_test_fixture',False)","'synthetic_fixture':False")
    body=body.replace("'version':'budget_task_v1'","'version':'budget_task_v1_membership_sensitivity_'+str(ACTIVE_SCOPE)")
    ns=core.__dict__;ns['ACTIVE_SCOPE']=scope;exec(compile(body,'preserved_budget_evaluator','exec'),ns)
    out=OUT/('reproduced_153' if scope=='153' else 'active_design_151')
    core.evaluate(argparse.Namespace(root=str(OUT/'no_raw_prediction_root'),output=str(out),bootstrap=B))
    settings_path=out/'run_settings.json';settings=json.loads(settings_path.read_text(encoding='utf-8'))
    settings.update(bootstrap_replicates=B,input_scope=scope,excluded_record_ids=sorted(EXCLUDE) if scope=='151' else [],score_reuse_source='SHA256-verified sequence_score_diagnostics.tsv from the real server run; six frozen outputs per generated candidate',membership_only_change=True,grouping_rule='restrict original dependence blocks to retained members; no reclustering or group splitting',software_note='same NumPy version as archived server run; original selection, tie, bootstrap, endpoint and pairwise functions preserved',no_new_inference=True)
    settings_path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['151','153'],required=True);parser.add_argument('--bootstrap',type=int,default=10000);args=parser.parse_args()
    run(args.scope,args.bootstrap)
