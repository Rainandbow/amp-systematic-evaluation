"""Rebuild paired hit-difference displays from frozen formal results; no resampling."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import csv,json,math,sys

OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
def read(name):
 with (OUT/'inputs'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(name,rows):
 with (OUT/'tables'/name).open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
MODELS=['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP','c_AMPs']
COHORTS=['ApexGO_template_based','ApexGO_template_free','HydrAMP_auxiliary_preselected','HydrAMP_no_auxiliary_preselection']
TASKS=['any_eligible_measured_strain','source_reference_Ecoli']
manifest=json.loads((OUT/'input_manifest.json').read_text(encoding='utf8'))
for name,item in manifest['inputs'].items():assert sha256((OUT/'inputs'/name).read_bytes()).hexdigest()==item['sha256'],name
points=read('baseline_and_model_points.tsv');pairs=read('paired_model_minus_baseline.tsv')
members=read('endpoint_membership.tsv');descriptors=read('descriptor_values.tsv');scores=read('scores151.tsv')
intervals=read('baseline_and_model_intervals.tsv')
assert len(points)==96 and len(pairs)==1440 and len(members)==302 and len(descriptors)==151 and len(scores)==906
assert all(r['smoke_only']=='False' for r in points+pairs)
assert len({r['sequence_id'] for r in descriptors})==151
di={r['sequence_id']:r for r in descriptors};si={(r['sequence_id'],r['model']):r for r in scores}
assert len(si)==906
pi={(r['cohort'],r['task'],r['method']):r for r in points}

def selected_hits(values,labels,K):
 # Independent tier implementation: all tied records receive the same label-blind weight.
 remaining=float(K);hit=0.;selected=0.
 for score in sorted(set(values),reverse=True):
  ix=[i for i,x in enumerate(values) if x==score]
  weight=min(1.,remaining/len(ix))
  if weight<=0:break
  selected+=weight*len(ix);hit+=weight*sum(labels[i] for i in ix);remaining-=weight*len(ix)
 assert abs(selected-K)<1e-10
 return hit

checks=[]
for cohort in COHORTS:
 for task in TASKS:
  mm=[r for r in members if r['cohort']==cohort and r['task']==task]
  n=len(mm);K=math.ceil(.2*n);y=[int(r['label']) for r in mm]
  assert len({r['record_id'] for r in mm})==n
  for method in MODELS+['length_high','length_low','KR_minus_DE_high','KR_minus_DE_low','mean_Kyte_Doolittle_high','mean_Kyte_Doolittle_low']:
   values=[float(si[r['sequence_id'],method]['score']) if method in MODELS else float(di[r['sequence_id']][method]) for r in mm]
   for r in mm:
    assert di[r['sequence_id']]['block']==r['block']
    if method in MODELS:assert si[r['sequence_id'],method]['block']==r['block']
   hit=selected_hits(values,y,K);saved=pi[cohort,task,method]
   assert math.isclose(hit,float(saved['positive_hits']),abs_tol=1e-10),(cohort,task,method)
   assert float(saved['budget_slots'])==K and int(saved['n'])==n and int(saved['positive'])==sum(y)
   checks.append({'cohort':cohort,'task':task,'method':method,'recomputed_hits':hit,'saved_hits':float(saved['positive_hits']),'passed':True})
rows=[]
for cohort in COHORTS:
 for task in TASKS:
  for model in MODELS:
   rr=[r for r in pairs if r['cohort']==cohort and r['task']==task and r['model']==model and r['baseline']=='length_low' and r['metric']=='positive_hits']
   assert len(rr)==1;r=rr[0]
   m=pi[cohort,task,model];b=pi[cohort,task,'length_low'];delta=float(m['positive_hits'])-float(b['positive_hits'])
   assert math.isclose(delta,float(r['difference']),abs_tol=1e-10)
   assert int(r['bootstrap'])==10000 and r['multiplicity_adjusted']=='False'
   valid=math.isfinite(float(r['ci_low'])) and math.isfinite(float(r['ci_high']))
   assert valid==(r['status']=='pointwise_exploratory_percentile')
   rows.append(dict(cohort=cohort,task=task,reference_strain=r['reference_strain'],threshold=r['threshold'],unit=r['unit'],n=r['n'],positive=r['positive'],threshold_negative=r['threshold_negative'],blocks=r['blocks'],budget_fraction=r['budget'],budget_slots=m['budget_slots'],model=model,model_hits=m['positive_hits'],shortest_first_hits=b['positive_hits'],difference=r['difference'],ci_low=r['ci_low'],ci_high=r['ci_high'],interval_status=r['status'],valid_fraction=r['valid_fraction'],contributing_blocks=r['contributing_blocks'],bootstrap=r['bootstrap'],multiplicity_adjusted=r['multiplicity_adjusted'],ci_method='saved paired block-bootstrap percentile interval',difference_direction='model minus shortest-first',interval_includes_zero=(float(r['ci_low'])<=0<=float(r['ci_high'])) if valid else '',baseline_selection='one named comparator from six reported unfitted controls; display selected during revision'))
assert len(rows)==48
(OUT/'tables').mkdir(exist_ok=True)
write('Table_S12_all_48_paired_hit_differences.tsv',rows)
primary=[r for r in rows if r['cohort']==COHORTS[0] and r['task']==TASKS[1]]
assert len(primary)==6
write('Fig4b_six_models_minus_shortest_first.tsv',primary)
write('Fig4a_any_strain_absolute_hits.tsv',[p for p in points if p['cohort']==COHORTS[0] and p['task']==TASKS[0]])
write('Fig4cd_reference_discrimination.tsv',[r for r in intervals if r['cohort']==COHORTS[0] and r['task']==TASKS[1] and r['metric'] in ['pooled_AUROC','macro_within_family_AUROC']])
write('point_reconstruction_96.tsv',checks)
qa={'inputs_hash_verified':True,'formal_paired_rows':1440,'extracted_named_baseline_contrasts':48,'cohort_task_model_grid_complete':True,'point_hits_independently_reconstructed':96,'all_point_differences_match_saved_contrasts':True,'intervals_reused_from_formal_paired_contrasts':True,'interval_status_counts':dict(Counter(r['interval_status'] for r in rows)),'main_reference_six_rows':primary,'main_reference_all_six_intervals_include_zero':all(r['interval_includes_zero'] for r in primary),'main_reference_possible_positive_upper_max':max(float(r['ci_high']) for r in primary),'main_reference_possible_negative_lower_min':min(float(r['ci_low']) for r in primary),'new_inference':False,'new_bootstrap':False,'new_baseline_selection_or_fitting':False,'display_comparator_selected_in_revision':True}
(OUT/'qa/numerical_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in qa.items() if k!='main_reference_six_rows'},ensure_ascii=False,indent=2))
for r in primary:print(r['model'],r['model_hits'],r['shortest_first_hits'],r['difference'],r['ci_low'],r['ci_high'])
