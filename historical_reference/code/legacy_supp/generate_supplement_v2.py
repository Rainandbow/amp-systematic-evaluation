#!/usr/bin/env python3
"""Generate S1-S14 and supplementary source tables from saved study results."""
import argparse,json,csv,hashlib,sys,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import supplement_helpers as H
import draw_legacy as L
COHORTS=['ApexGO_template_based','ApexGO_template_free','HydrAMP_auxiliary_preselected','HydrAMP_no_auxiliary_preselection']
TITLES=['ApexGO: template-based','ApexGO: template-free','HydrAMP: auxiliary selection','HydrAMP: no auxiliary selection']
TASK='any_measured_strain_source_threshold'

def s3():
 fig,axes=plt.subplots(1,2,figsize=(183/25.4,110/25.4));fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.85,wspace=.46)
 a,b=axes;codes=[['Y','Y','Y','Y','S','Y'] for _ in range(5)]+[['Y','E','R','R','T','Y']]
 palette={'Y':'#daeaf2','S':'#f3e6c7','E':'#f3e6c7','R':'#eceff2','T':'#eceff2'}
 for i,row in enumerate(codes):
  for j,code in enumerate(row):a.add_patch(Rectangle((j-.5,i-.5),1,1,fc=palette[code],ec='white',lw=1));a.text(j,i,code,ha='center',va='center',fontsize=7)
 a.set_xlim(-.5,5.5);a.set_ylim(5.5,-.5);a.set_yticks(range(6),[H.name(m) for m in H.MS]);a.set_xticks(range(6),['Native labels','AP / AUROC','q-based retention','Threshold transfer','Inventory screen','New-panel labels'],rotation=60,ha='right',rotation_mode='anchor',fontsize=5.5);a.tick_params(length=0)
 for sp in a.spines.values():sp.set_visible(False)
 H.heading(a,'a','Analysis-specific inputs')
 b.axis('off');H.heading(b,'b','Output and inventory definitions')
 lines=['Y: verified analysis input','E: c_AMPs min-branch ranking extension','R: no verified native ensemble probability','S: source-qualified available inventory','T: full training sequence mapping unavailable','','c_AMPs native decision: three branches >0.5','No min-branch for q retention or transfer','Inventory availability is not training independence']
 for i,text in enumerate(lines):b.text(0,.98-i*.107,text,transform=b.transAxes,fontsize=5.7,va='top')
 H.save(fig,'FigS03_analysis_availability',list(axes))
 H.write(H.OUT/'source_data/FigS03_availability.tsv',[dict(model=m,native_decisions=row[0],ranking=row[1],score_selection=row[2],threshold_transfer=row[3],inventory=row[4],new_panel_decisions=row[5]) for m,row in zip(H.MS,codes)])

def s11():
 rows=H.read_tsv(ROOT/'data/inventory_retention.tsv');models=H.MS[:5];ix={(r['tool'],r['dataset']):r for r in rows if r['tool'] in models}
 fig,ax=plt.subplots(figsize=(183/25.4,100/25.4));fig.subplots_adjust(left=.24,right=.88,bottom=.24,top=.84)
 z=np.array([[float(ix[m,s]['retained_fraction']) for s in H.DS] for m in models]);im=ax.imshow(z,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
 labels=['Macrel: source proxy','AMPlify: supplied training','ampir: recovered training','tsAMP-I: partial training','HMD-AMP: source pool']
 ax.set_yticks(range(5),labels,fontsize=6);ax.set_xticks(range(6),[H.name(s) for s in H.DS],rotation=45,ha='right',rotation_mode='anchor');ax.tick_params(length=0);H.heading(ax,'a','Retention after available-inventory screening')
 for i,m in enumerate(models):
  for j,s in enumerate(H.DS):
   r=ix[m,s];ax.text(j,i,f"{z[i,j]*100:.1f}%\n{int(r['retained_clean90_n']):,}/{int(r['common_n']):,}",ha='center',va='center',fontsize=5.5,color='white' if z[i,j]>.65 else '#203b4e')
 for sp in ax.spines.values():sp.set_visible(False)
 cax=fig.add_axes([.91,.24,.014,.60]);cb=fig.colorbar(im,cax=cax,ticks=[0,.5,1]);cb.ax.set_yticklabels(['0%','50%','100%']);cb.ax.tick_params(labelsize=6);cb.outline.set_visible(False)
 H.save(fig,'FigS11_inventory_retention',[ax]);H.write(H.OUT/'source_data/FigS11_inventory_retention.tsv',rows)

def activity(ax,rows,cohort,letter,label,groups):
 rr=[r for r in rows if r['cohort']==cohort];g=H.one(groups,cohort=cohort)
 p=H.one(rr,model='Macrel',metric='end_to_end_positive_detection');n=H.one(rr,model='Macrel',metric='assay_FPR_supported');P=int(float(p['denominator']));N=int(float(n['denominator']))
 ax.set_title(letter+'  '+label,loc='left',fontsize=7,pad=23);ax.text(0,1.035,f"n={g['n']}; P={P}; N={N}; blocks={g['blocks']}",transform=ax.transAxes,fontsize=6,va='bottom')
 for i,m in enumerate(H.MS):
  for metric,color,offset in [('end_to_end_positive_detection','#247b89',-.18),('assay_FPR_supported','#c27136',.18)]:
   r=H.one(rr,model=m,metric=metric);den=float(r['denominator'])
   if den==0:continue
   value=float(r['value']);assert abs(value-float(r['numerator'])/den)<1e-8;lo=H.num(r['ci_low']);hi=H.num(r['ci_high']);valid=lo is not None and hi is not None
   if valid:ax.hlines(i+offset,100*lo,100*hi,color=color,lw=.9)
   ax.plot(value*100,i+offset,'o',ms=3.5,color=color,mfc=color if valid else 'white');ax.text(1.035,i+offset,f"{float(r['numerator']):.0f}/{den:.0f}",transform=ax.get_yaxis_transform(),fontsize=5.5,color=color,va='center')
 ax.set_ylim(5.5,-.5);ax.set_xlim(-3,103);ax.set_xticks([0,25,50,75,100]);ax.set_yticks(range(6),[H.name(m) for m in H.MS]);ax.set_xlabel('Positive-call proportion (%)')

def s12(new,groups):
 fig,axes=plt.subplots(1,2,figsize=(183/25.4,95/25.4));fig.subplots_adjust(left=.12,right=.91,bottom=.20,top=.72,wspace=.65)
 fig.legend(handles=[Line2D([],[],marker='o',color=c,lw=0,ms=3.5,label=t) for c,t in [('#247b89','Activity-positive sequences'),('#c27136','Assay-negative sequences')]],loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,fontsize=6)
 for ax,c,l,t in zip(axes,COHORTS[2:],'ab',TITLES[2:]):activity(ax,new,c,l,t,groups)
 H.save(fig,'FigS12_HydrAMP_small_cohorts',list(axes));H.write(H.OUT/'source_data/FigS12_HydrAMP.tsv',[r for r in new if r['cohort'].startswith('HydrAMP')])

def s13(points,settings,stability):
 ix={(r['model'],r['held_source']):r for r in points};assert len(ix)==30
 fig,axes=plt.subplots(1,2,figsize=(183/25.4,100/25.4));fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.86,wspace=.45)
 a,b=axes;z=np.array([[int(ix[m,s]['policy_status']=='feasible') for s in H.DS] for m in H.MS[:5]]+[[np.nan]*6]);a.imshow(np.ma.masked_invalid(z),vmin=0,vmax=1,cmap='Blues',aspect='auto')
 q=np.array([[float(ix[m,s]['bootstrap_policy_feasible_fraction']) for s in H.DS] for m in H.MS[:5]]+[[np.nan]*6]);b.imshow(np.ma.masked_invalid(q),vmin=0,vmax=1,cmap='Blues',aspect='auto')
 for ax,label in zip(axes,['a  Development policy availability','b  Feasible bootstrap resamples (%)']):
  ax.set_title(label,loc='left',pad=10,fontsize=7);ax.set_xticks(range(6),[H.name(s) for s in H.DS],rotation=45,ha='right',rotation_mode='anchor');ax.set_yticks(range(6),[H.name(m) for m in H.MS]);ax.tick_params(length=0)
  ax.add_patch(Rectangle((-.5,4.5),6,1,fc='#eeeeee',ec='none'));ax.text(2.5,5,'Native ensemble score unavailable',ha='center',va='center',fontsize=5.5,color='#555555')
  for sp in ax.spines.values():sp.set_visible(False)
 for i,m in enumerate(H.MS[:5]):
  for j,s in enumerate(H.DS):
   a.text(j,i,'Yes' if z[i,j] else 'No',ha='center',va='center',fontsize=6,color='white' if z[i,j] else '#203b4e')
   b.text(j,i,f'{q[i,j]*100:.1f}',ha='center',va='center',fontsize=5.5,color='white' if q[i,j]>.6 else '#203b4e')
 H.save(fig,'FigS13_policy_availability_and_stability',list(axes))
 supplement=[dict(r) for r in points]
 for s in H.DS:supplement.append(dict(model='c_AMPs',held_source=s,policy_status='ineligible_score_definition',reason='No verified native continuous ensemble probability'))
 H.write(H.OUT/'source_data/FigS13_complete_policy_results.tsv',supplement)
 if stability:H.write(H.OUT/'source_data/FigS13_threshold_stability.tsv',stability)
 H.dump(H.OUT/'source_data/FigS13_run_settings.json',settings)

def s14(points,intervals,members):
 ci={(r['cohort'],r['subset'],r['model'],r['metric']):r for r in intervals if r['task']==TASK};fig,axes=plt.subplots(2,2,figsize=(183/25.4,160/25.4));fig.subplots_adjust(left=.12,right=.89,bottom=.11,top=.83,wspace=.70,hspace=.72)
 fig.legend(handles=[Line2D([],[],marker=m,color=c,lw=0,ms=3.5,label=l) for m,c,l in [('o','#247b89','All generated candidates'),('s','#b08043','Five-inventory-pass subset')]],loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,fontsize=6)
 shown=[];availability=[]
 for ax,cohort,label,k in zip(axes.flat,COHORTS,TITLES,'abcd'):
  aa=H.one(members,cohort=cohort,task=TASK,subset='all_generated');ss=H.one(members,cohort=cohort,task=TASK,subset='five_inventory_pass');ax.set_title(k+'  '+label,loc='left',pad=23,fontsize=7)
  ax.text(0,1.035,f"All n={aa['n']}, P={aa.get('positive','?')}; pass n={ss['n']}, P={ss.get('positive','0')}",transform=ax.transAxes,fontsize=5.8,va='bottom')
  for i,model in enumerate(H.MS):
   for subset,color,marker,offset in [('all_generated','#247b89','o',-.15),('five_inventory_pass','#b08043','s',.15)]:
    rr=[r for r in points if r['cohort']==cohort and r['task']==TASK and r['subset']==subset and r['model']==model]
    if not rr:
     assert subset=='five_inventory_pass' and int(ss['n'])==0
     availability.append(dict(cohort=cohort,model=model,subset=subset,status='empty screened subset',n=0,positive=0));continue
    assert len(rr)==1;r=rr[0];P=int(r['positive']);tp=float(r['TP']);availability.append(dict(cohort=cohort,model=model,subset=subset,status='estimable' if P else 'no activity-positive member',n=r['n'],positive=P))
    if P==0:continue
    value=float(r['end_to_end_positive_detection']);assert abs(value-tp/P)<1e-8;c=ci.get((cohort,subset,model,'end_to_end_positive_detection'),{});lo=H.num(c.get('ci_low'));hi=H.num(c.get('ci_high'));valid=lo is not None and hi is not None
    if valid:ax.hlines(i+offset,lo*100,hi*100,color=color,lw=.9)
    ax.plot(value*100,i+offset,marker=marker,ls='none',ms=3.4,color=color,mfc=color if valid else 'white');ax.text(1.035,i+offset,f'{tp:.0f}/{P}',transform=ax.get_yaxis_transform(),fontsize=5.5,color=color,va='center');shown.append(dict(r,saved_ci_low=c.get('ci_low',''),saved_ci_high=c.get('ci_high',''),saved_ci_status=c.get('status','not reported')))
  ax.set_ylim(5.5,-.5);ax.set_xlim(-3,103);ax.set_xticks([0,25,50,75,100]);ax.set_yticks(range(6),[H.name(m) for m in H.MS]);ax.set_xlabel('Detected / activity-positive (%)')
 H.save(fig,'FigS14_new_panel_inventory_sensitivity',list(axes.flat));H.write(H.OUT/'source_data/FigS14_displayed_metrics.tsv',shown);H.write(H.OUT/'source_data/FigS14_subset_availability.tsv',availability)

def load_saved(args):
 inputs=list((ROOT/'data').glob('*'));meta={};data={}
 if args.results_root:
  loss=args.results_root/'loss';stamp=json.loads((loss/'baseline_replay.json').read_text(encoding='utf-8'));assert int(stamp['bootstrap'])>=10000 and not stamp['smoke_only'];points=H.read_tsv(loss/'selection_points.tsv');intervals=H.read_tsv(loss/'selection_intervals.tsv')
  for ds in H.DS:
   for model in H.MS[:5]:
    p=H.one(points,dataset=ds,model=model,policy='q095')
    for metric,key in [('high_confidence_coverage_095','coverage'),('high_confidence_error_095','retained_error')]:
     r=H.one(H.D['reliability'],dataset=ds,tool=model,metric=metric);assert abs(float(r['estimate'])-float(p[key]))<1e-10
     c=H.one(intervals,dataset=ds,model=model,policy='q095',metric=key);r.update(estimate=p[key],ci_low=c['ci_low'],ci_high=c['ci_high'],ci_status=c['status'],bootstrap=c['bootstrap'])
  paths=[]
  for p in args.results_root.glob('transfer_exact_v3_*'):
   f=p/'run_settings.json'
   if f.exists():
    s=json.loads(f.read_text(encoding='utf-8'))
    if s.get('version')=='transfer_exact_v3' and int(s.get('bootstrap',0))>=10000 and not s.get('smoke_only',True):paths.append(p)
  tr=args.transfer_root or (sorted(paths,key=lambda x:x.name)[-1] if paths else None);assert tr is not None,'Missing completed exact-v3 transfer results';settings=json.loads((tr/'run_settings.json').read_text(encoding='utf-8'));assert settings['version']=='transfer_exact_v3' and int(settings['bootstrap'])>=10000 and not settings['smoke_only']
  data['transfer']=H.read_tsv(tr/'transfer_points.tsv');data['settings']=settings;data['stability']=H.read_tsv(tr/'threshold_stability.tsv');inputs += [tr/x for x in ['run_settings.json','transfer_points.tsv','threshold_stability.tsv']]
  panel=args.results_root/'panel_metrics';s=json.loads((panel/'interpretation.json').read_text(encoding='utf-8'));assert int(s['bootstrap'])>=10000 and not s['smoke_only'];pp=H.read_tsv(panel/'panel_metrics.tsv');cc=H.read_tsv(panel/'panel_intervals.tsv');members=H.read_tsv(panel/'task_membership_summary.tsv');ix={(r['cohort'],r['subset'],r['model'],r['metric']):r for r in cc if r['task']==TASK};new=[];groups=[]
  for cohort in COHORTS:
   r=H.one(pp,cohort=cohort,task=TASK,subset='all_generated',model='Macrel');groups.append(dict(cohort=cohort,n=r['n'],blocks=r['group_n']))
  for r in pp:
   if r['task']!=TASK or r['subset']!='all_generated':continue
   for metric in ['end_to_end_positive_detection','assay_FPR_supported']:
    c=ix.get((r['cohort'],'all_generated',r['model'],metric),{});numerator=float(r['TP']) if metric=='end_to_end_positive_detection' else float(r['FP']);denominator=float(r['positive']) if metric=='end_to_end_positive_detection' else float(r['FP'])+float(r['TN']);new.append(dict(cohort=r['cohort'],model=r['model'],metric=metric,value=r[metric],numerator=numerator,denominator=denominator,ci_low=c.get('ci_low',''),ci_high=c.get('ci_high',''),display_status=c.get('status','not reported')))
  data.update(new=new,groups=groups,panel_points=pp,panel_intervals=cc,panel_members=members);inputs += [panel/x for x in ['interpretation.json','panel_metrics.tsv','panel_intervals.tsv','task_membership_summary.tsv']]+[loss/x for x in ['baseline_replay.json','selection_points.tsv','selection_intervals.tsv']];meta['transfer_source']=str(tr)
 else:
  assert args.local_review,'Supply --results-root for all S1-S14; --local-review exports only locally verified S1-S13'
  data.update(transfer=[r for r in H.read_tsv(ROOT/'data/formal_policy_availability.tsv') if r['model']!='c_AMPs'],settings={'version':'formal_exact_v3_frozen_point_export','bootstrap':10000,'local_review':True},stability=[],new=H.read_tsv(ROOT/'data/new_panel_display.tsv'),groups=H.read_tsv(ROOT/'data/new_panel_groups.tsv'))
  meta['S14']='Not generated locally: full inventory-pass statistics remain on server';meta['S13_threshold_intervals']='Not imported locally; formal source table remains on server'
 data['fingerprints']={str(p):H.sha(p) for p in inputs};data['meta']=meta
 return data

def finish(args,data,stems):
 import pymupdf
 pdfs=sorted((H.OUT/'PDF').glob('FigS*.pdf'));assert len(pdfs)==len(stems)
 doc=pymupdf.open()
 for p in pdfs:
  with pymupdf.open(p) as f:doc.insert_pdf(f)
 doc.save(H.OUT/'All_supplementary_figures.pdf');doc.close()
 text=(ROOT/'Supplementary_legends_EN.txt').read_text(encoding='utf-8-sig')
 if args.local_review:text+='\nLOCAL REVIEW: S14 was not generated because its screened-subset numerical results remain on the server. No replacement data or inferred subset values were used. S13 uses the downloaded formal point and feasibility-fraction table; full threshold interval statistics are added in server mode.\n'
 (H.OUT/'Supplementary_legends_EN.txt').write_text(text,encoding='utf-8')
 for p in (ROOT/'data').iterdir():
  if p.is_file():shutil.copyfile(p,H.OUT/'source_data'/p.name)
 H.write(H.OUT/'source_data/used_reliability_estimates.tsv',H.D['reliability']);assert data['fingerprints']=={p:H.sha(p) for p in data['fingerprints']}
 H.dump(H.OUT/'QA/run_receipt.json',dict(completed_figures=len(stems),stems=stems,source_unchanged=True,new_prediction=False,new_bootstrap=False,formal_server_mode=bool(args.results_root),notes=data['meta'],input_sha256=data['fingerprints']))
 print('SUPPLEMENTARY FIGURES COMPLETE',len(stems),H.OUT,flush=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results-root',type=Path);parser.add_argument('--transfer-root',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--local-review',action='store_true');parser.add_argument('--only',type=int,choices=range(1,15));args=parser.parse_args()
 manifest_path=ROOT/'input_manifest.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
  for n,h in manifest.items():assert H.sha(ROOT/n)==h,'Checksum mismatch: '+n
  print('BUNDLE HASHES PASS',len(manifest),flush=True)
 H.OUT=args.output.resolve();H.OUT.mkdir(parents=True,exist_ok=False)
 for n in ['PDF','SVG','PNG_600dpi','TIFF_600dpi','preview','source_data','QA']:(H.OUT/n).mkdir()
 data=load_saved(args);funcs=[L.s1,L.s2,s3,lambda:L.ranking('ap',4),lambda:L.ranking('auroc',5),L.s6,L.s7,L.s8,L.s9,L.s10,s11,lambda:s12(data['new'],data['groups']),lambda:s13(data['transfer'],data['settings'],data['stability'])]
 if args.results_root:funcs.append(lambda:s14(data['panel_points'],data['panel_intervals'],data['panel_members']))
 selected=[args.only-1] if args.only else list(range(len(funcs)))
 for i in selected:
  assert i<len(funcs),'S14 requires formal results-root'
  funcs[i]()
 if not args.only:finish(args,data,[p.stem for p in sorted((H.OUT/'PDF').glob('FigS*.pdf'))])

if __name__=='__main__':main()

