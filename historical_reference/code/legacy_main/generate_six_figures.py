#!/usr/bin/env python3
"""Six manuscript main figures from frozen data and completed server statistics."""
import argparse,csv,hashlib,json,math,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
from audit_panel_alignment import require_matplotlib_panel_alignment
MODELS=['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP','c_AMPs']
SOURCES=['AMPlify','tsAMP_I','c_AMPs_benchmark_resplit','HMD_AMP','AmPEP_raw_source','ampir_raw_mature90']
DISPLAY={'tsAMP_I':'tsAMP-I','HMD_AMP':'HMD-AMP','c_AMPs_benchmark_resplit':'c_AMPs','AmPEP_raw_source':'AmPEP','ampir_raw_mature90':'ampir'}
COLORS=dict(zip(MODELS,['#0072B2','#00875A','#A66A00','#AA62A0','#407DB1','#D55E00']))
MARKERS=dict(zip(MODELS,['o','s','^','D','P','X']))
TASK='any_measured_strain_source_threshold'
COHORTS=['ApexGO_template_based','ApexGO_template_free','HydrAMP_auxiliary_preselected','HydrAMP_no_auxiliary_preselection']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'legend.frameon':False,'savefig.facecolor':'white'})

def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def num(x):
 try:return float(x)
 except (TypeError,ValueError):return float('nan')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def name(x):return DISPLAY.get(x,x)
def one(rows,**kw):
 r=[r for r in rows if all(r.get(k)==v for k,v in kw.items())]
 assert len(r)==1,kw
 return r[0]
def title(ax,letter,text,pad=10):ax.set_title(letter+'  '+text,loc='left',pad=pad,fontsize=7)
def ci_point(ax,x,y,lo,hi,color,marker='o',vertical=False,hollow=False,ms=3.5):
 valid=math.isfinite(num(lo)) and math.isfinite(num(hi))
 if valid:
  if vertical:ax.vlines(x,float(lo),float(hi),color=color,lw=.9)
  else:ax.hlines(y,float(lo),float(hi),color=color,lw=.9)
 ax.plot(x,y,marker=marker,linestyle='none',ms=ms,color=color,mfc='white' if hollow else color,mew=.75)
 return valid
def model_legend(fig,n=6,y=.99):
 fig.legend(handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],lw=.8,ms=3,label=name(m)) for m in MODELS[:n]],loc='upper center',bbox_to_anchor=(.52,y),ncol=n if n==5 else 3,fontsize=6.2,columnspacing=1.4,handletextpad=.4)
def pdf_checks(pdf,stem,out):
 for script,extra in [('audit_pdf_text.py',[]),('audit_figure_collisions.py',['--json-out',str(out/'QA'/(stem+'.collision.json')),'--strict'])]:
  r=subprocess.run([sys.executable,str(HERE/'code'/script),str(pdf),*extra],capture_output=True,text=True,encoding='utf-8',errors='replace')
  (out/'QA'/(stem+'.'+script+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
  if r.returncode:raise RuntimeError('Figure QA failed: '+stem+'; see '+script+'.log')
def export(fig,axes,stem,args,ids=None,row_groups=None,column_groups=None):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids or list('abcdef')[:len(axes)],row_groups=row_groups,column_groups=column_groups,json_out=args.output/'QA'/(stem+'.alignment.json'),strict=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
 fig.savefig(args.output/'PDF'/(stem+'.pdf'),dpi=600)
 fig.savefig(args.output/'SVG'/(stem+'.svg'),dpi=600)
 fig.savefig(args.output/'PNG_600dpi'/(stem+'.png'),dpi=600)
 fig.savefig(args.output/'TIFF_600dpi'/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 fig.savefig(args.output/'preview'/(stem+'.png'),dpi=300)
 plt.close(fig)
 pdf_checks(args.output/'PDF'/(stem+'.pdf'),stem,args.output)
 print('FIGURE PASS',stem,flush=True)

def load_inputs(args):
 D=json.loads((HERE/'data/frozen_core.json').read_text(encoding='utf-8'))
 data=dict(core=D,patterns=read(HERE/'data/exclusive_membership_patterns.tsv'),length=read(HERE/'data/length_mcc_intervals.tsv'),new_summary=read(HERE/'data/new_panel_summary.tsv'),new_groups=read(HERE/'data/new_panel_groups.tsv'),new_display=read(HERE/'data/new_panel_display.tsv'))
 used=list((HERE/'data').glob('*'))
 if args.results_root:
  loss=args.results_root/'loss';receipt=json.loads((loss/'baseline_replay.json').read_text(encoding='utf-8'))
  assert int(receipt['bootstrap'])>=10000 and not receipt.get('smoke_only',False)
  data['selection']=[r for r in read(loss/'selection_points.tsv') if r['policy']=='top80'];data['curves']=read(loss/'selection_curves.tsv');data['selection_ci']=[r for r in read(loss/'selection_intervals.tsv') if r['policy']=='top80']
  used += [loss/x for x in ['baseline_replay.json','selection_points.tsv','selection_curves.tsv','selection_intervals.tsv']]
  if args.transfer_root:transfer=args.transfer_root
  else:
   candidates=[]
   for p in args.results_root.glob('transfer_exact_v3_*'):
    f=p/'run_settings.json'
    if not f.exists():continue
    r=json.loads(f.read_text(encoding='utf-8'))
    if r.get('version')=='transfer_exact_v3' and int(r.get('bootstrap',0))>=10000 and not r.get('smoke_only',True):candidates.append(p)
   assert candidates,'No completed formal transfer_exact_v3 results found'
   transfer=sorted(candidates,key=lambda p:p.name)[-1]
  r=json.loads((transfer/'run_settings.json').read_text(encoding='utf-8'))
  assert r['version']=='transfer_exact_v3' and int(r['bootstrap'])>=10000 and not r['smoke_only']
  data['transfer']=read(transfer/'transfer_points.tsv');data['transfer_ci']=read(transfer/'transfer_intervals.tsv');data['transfer_root']=str(transfer)
  used += [transfer/x for x in ['run_settings.json','transfer_points.tsv','transfer_intervals.tsv']]
  panel=args.results_root/'panel_metrics';r=json.loads((panel/'interpretation.json').read_text(encoding='utf-8'))
  assert int(r['bootstrap'])>=10000 and not r['smoke_only']
  pp=[r for r in read(panel/'panel_metrics.tsv') if r['task']==TASK and r['subset']=='all_generated']
  cc={(r['cohort'],r['model'],r['metric']):r for r in read(panel/'panel_intervals.tsv') if r['task']==TASK and r['subset']=='all_generated'};shown=[]
  for r in pp:
   for metric in ['end_to_end_positive_detection','assay_FPR_supported']:
    n=float(r['TP']) if metric=='end_to_end_positive_detection' else float(r['FP']);den=float(r['positive']) if metric=='end_to_end_positive_detection' else float(r['FP'])+float(r['TN']);c=cc.get((r['cohort'],r['model'],metric),{});v=num(r[metric])
    if den>0:assert math.isfinite(v) and abs(v-n/den)<1e-8
    shown.append(dict(cohort=r['cohort'],model=r['model'],metric=metric,value=v,numerator=n,denominator=den,ci_low=c.get('ci_low',''),ci_high=c.get('ci_high',''),display_status=c.get('status','no saved interval')))
  data['new_display']=shown;data['new_groups']=[dict(cohort=c,n=one(pp,cohort=c,model='Macrel')['n'],blocks=one(pp,cohort=c,model='Macrel')['group_n']) for c in COHORTS]
  used += [panel/x for x in ['interpretation.json','panel_metrics.tsv','panel_intervals.tsv']]
  print('USING TRANSFER',transfer,flush=True)
 else:
  assert args.reuse_frozen_transfer,'Supply --results-root for complete numerical redrawing, or --reuse-frozen-transfer for local frozen-vector review'
  data['selection']=read(HERE/'data/selection_points.tsv');data['curves']=read(HERE/'data/selection_curves.tsv');data['selection_ci']=read(HERE/'data/legacy_selection_intervals.tsv');data['transfer']=None
 assert len(data['selection'])==30 and len(data['curves'])==600
 assert len(D['mcc'])==36 and len(data['new_display'])==48
 data['fingerprints']={str(p):sha(p) for p in used if p.is_file()}
 return data

def fig1(data,args):
 D=data['core'];cohorts=D['cohorts'];fig=plt.figure(figsize=(183/25.4,190/25.4))
 a=fig.add_axes([.14,.64,.32,.25]);b=fig.add_axes([.64,.64,.32,.25]);bars=fig.add_axes([.14,.29,.82,.19]);matrix=fig.add_axes([.14,.12,.82,.12])
 p=np.array([int(r['common_positive']) for r in cohorts]);n=np.array([int(r['common_negative']) for r in cohorts]);yy=np.arange(6)
 a.barh(yy,p,color='#4b8498',height=.55,edgecolor='none');a.barh(yy,n,left=p,color='#dce4e9',height=.55,edgecolor='none')
 a.set_yticks(yy,[name(s) for s in SOURCES]);a.set_ylim(5.6,-.6);a.set_xlim(0,20500);a.set_xticks([0,5000,10000,15000],['0','5k','10k','15k']);a.set_xlabel('Common-support records')
 for i,r in enumerate(cohorts):a.text(int(r['common_n'])+350,i,f"{int(r['common_n']):,}",fontsize=5.5,va='center')
 a.legend(handles=[Patch(fc='#4b8498',label='Positive'),Patch(fc='#dce4e9',label='Negative')],loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,fontsize=5.5,handlelength=1);title(a,'a','Historical common support',pad=27)
 old_studies=['APEX_selected','APEX_scoring_selected','AMPGen'];old=[one(D['external'],study=s,tool='Macrel') for s in old_studies]
 total=[int(r['tested_panel_n']) for r in old]+[int(r['generated_candidates']) for r in data['new_summary']];positive=[int(r['positive_evidence_n']) for r in old]+[int(r['generated_positive']) for r in data['new_summary']]
 labels=['APEX: model-selected','APEX: scoring-selected','AMPGen','ApexGO: template-based','ApexGO: template-free','HydrAMP: auxiliary selection','HydrAMP: no\nauxiliary selection'];y=np.arange(7)
 b.barh(y,total,color='#dce4e9',height=.55,edgecolor='none');b.barh(y,positive,color='#4b8498',height=.34,edgecolor='none')
 b.set_yticks(y,labels,fontsize=5.8);b.set_ylim(6.6,-.6);b.set_xlim(0,133);b.set_xticks([0,25,50,75,100]);b.set_xlabel('Sequences within source group')
 for i,(p0,n0) in enumerate(zip(positive,total)):b.text(n0+2,i,f'{p0}/{n0}',fontsize=5.5,va='center')
 b.legend(handles=[Patch(fc='#dce4e9',label='Tested'),Patch(fc='#4b8498',label='Activity-positive')],loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,fontsize=5.5,handlelength=1);title(b,'b','Experimentally tested sequences',pad=27)
 patterns=data['patterns'];counts=np.array([int(r['exclusive_n']) for r in patterns]);bits=np.array([[int(r[s]) for s in SOURCES] for r in patterns])
 assert len(patterns)==28 and int(counts.sum())==28481 and int((counts*bits.sum(axis=1)).sum())==29104
 assert np.all(counts>0), 'Log count axis requires strictly positive values'
 x=np.arange(28);cs=['#97adbc' if sum(r)==1 else '#327d75' for r in bits]
 bars.bar(x,counts-.65,bottom=.65,color=cs,width=.7,edgecolor='none');bars.set_yscale('log');bars.set_ylim(.65,40000);bars.set_yticks([1,10,100,1000,10000],['1','10','100','1,000','10,000']);bars.yaxis.set_minor_locator(NullLocator());bars.set_xticks([]);bars.set_xlim(-.6,27.6);bars.set_ylabel('Exclusive sequences (log scale)')
 for i,c in enumerate(counts):
  if c>=400:bars.text(i,math.sqrt(.65*c),str(c),ha='center',va='center',rotation=90,rotation_mode='anchor',fontsize=5.5,color='#183247')
  else:bars.annotate(str(c),(i,c),xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=5.5)
 title(bars,'c','Exact sequence sharing across historical sources',pad=24)
 bars.legend(handles=[Patch(fc='#97adbc',label='Single source'),Patch(fc='#327d75',label='Shared across sources')],loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,fontsize=5.5,handlelength=1)
 for j in range(6):
  if j%2==0:matrix.axhspan(j-.5,j+.5,color='#f0f4f6',zorder=0)
 for i,bit in enumerate(bits):
  selected=np.flatnonzero(bit);matrix.scatter([i]*6,range(6),s=4,color='#d1dce1',edgecolors='none')
  if len(selected)>1:matrix.plot([i,i],[selected[0],selected[-1]],color=cs[i],lw=.7)
  matrix.scatter([i]*len(selected),selected,s=10,color=cs[i],edgecolors='none')
 matrix.set_xlim(bars.get_xlim());matrix.set_ylim(5.6,-.6);matrix.set_yticks(range(6),[name(s) for s in SOURCES]);matrix.set_xticks([]);matrix.set_xlabel('Mutually exclusive exact membership patterns');matrix.tick_params(length=0)
 for spine in matrix.spines.values():spine.set_visible(False)
 export(fig,[a,b,bars,matrix],'Fig01_design_and_data',args,ids=['a','b','c_count','c_matrix'],row_groups=[['a','b']],column_groups=[['c_count','c_matrix']])
 write(args.output/'source_data/Fig01_cohorts.tsv',cohorts);write(args.output/'source_data/Fig01_membership_patterns.tsv',patterns);write(args.output/'source_data/Supplementary_inventory_retention.tsv',read(HERE/'data/inventory_retention.tsv'))

def fig2(data,args):
 fig,axes=plt.subplots(2,3,figsize=(183/25.4,132/25.4));fig.subplots_adjust(left=.105,right=.98,bottom=.11,top=.91,wspace=.54,hspace=.65);rows=data['core']['mcc']
 for ax,ds,k in zip(axes.flat,SOURCES,'abcdef'):
  for i,model in enumerate(MODELS):
   r=one(rows,dataset=ds,tool=model);v=float(r['estimate']);lo=num(r['ci_low']);hi=num(r['ci_high']);ci_point(ax,v,i,lo,hi,COLORS[model],MARKERS[model],hollow=not math.isfinite(lo))
  ax.set_yticks(range(6),[name(m) for m in MODELS]);ax.set_ylim(5.6,-.6);ax.set_xlim(-.08,1.08);ax.set_xticks([0,.5,1]);ax.set_xlabel('MCC');title(ax,k,name(ds))
 export(fig,list(axes.flat),'Fig02_common_sequence_performance',args);write(args.output/'source_data/Fig02_metrics.tsv',rows)

def fig3(data,args):
 rows=[r for r in data['length'] if r['scope']=='global_primary'];assert len(rows)==144
 fig,axes=plt.subplots(3,2,figsize=(183/25.4,195/25.4));fig.subplots_adjust(left=.105,right=.98,bottom=.085,top=.87,wspace=.32,hspace=.66);model_legend(fig);shown=[];omitted=[]
 for ax,ds,k in zip(axes.flat,SOURCES,'abcdef'):
  bins=[]
  for bb in ['10_29','30_59','60_99','100_126']:
   r=one(rows,dataset=ds,model='Macrel',length_bin=bb)
   if int(r['positive'])>0 and int(r['negative'])>0:bins.append(bb)
   else:omitted.append(r)
  for j,model in enumerate(MODELS):
   xs=[];ys=[]
   for i,bb in enumerate(bins):
    r=one(rows,dataset=ds,model=model,length_bin=bb);x=i+(j-2.5)*.035;v=float(r['mcc']);lo=num(r['ci_low']);hi=num(r['ci_high']);sparse=min(int(r['positive']),int(r['negative']))<20
    ci_point(ax,x,v,lo,hi,COLORS[model],MARKERS[model],vertical=True,hollow=sparse,ms=3.1);xs.append(x);ys.append(v);shown.append(r)
   ax.plot(xs,ys,color=COLORS[model],lw=.6,alpha=.65)
  labels=[]
  for bb in bins:
   r=one(rows,dataset=ds,model='Macrel',length_bin=bb);labels.append(bb.replace('_','\u2013')+'\nP='+r['positive']+'; N='+r['negative'])
  ax.set_xticks(range(len(bins)),labels,fontsize=5.5);ax.set_xlim(-.25,len(bins)-.75);ax.set_ylim(-1.05,1.08);ax.set_yticks([-1,-.5,0,.5,1]);ax.set_ylabel('MCC');ax.set_xlabel('Length (residues)');title(ax,k,name(ds))
 export(fig,list(axes.flat),'Fig03_length_stratified_performance',args);write(args.output/'source_data/Fig03_displayed.tsv',shown);write(args.output/'source_data/Supplementary_length_omitted.tsv',omitted)

def fig4(data,args):
 points={(r['dataset'],r['model']):r for r in data['selection']};ci={(r['dataset'],r['model'],r['metric']):r for r in data['selection_ci']}
 fig,axes=plt.subplots(3,2,figsize=(183/25.4,200/25.4));fig.subplots_adjust(left=.11,right=.98,bottom=.085,top=.84,wspace=.34,hspace=.56);model_legend(fig,n=5,y=.995)
 fig.legend(handles=[Line2D([],[],color='#444444',lw=.9,label='Score-based retention'),Line2D([],[],color='#777777',lw=.8,ls=':',label='Random-retention expectation'),Line2D([],[],color='#444444',marker='D',ls='none',ms=3.5,label='Approximately 80% retained')],loc='upper center',bbox_to_anchor=(.52,.953),ncol=3,fontsize=5.8,columnspacing=1.2);displayed=[]
 for ax,ds,k in zip(axes.flat,SOURCES,'abcdef'):
  upper=[]
  for model in MODELS[:5]:
   curves=sorted([r for r in data['curves'] if r['dataset']==ds and r['model']==model],key=lambda r:float(r['target_coverage']));assert len(curves)==20
   x=np.array([float(r['positive_rejection'])*100 for r in curves]);y=np.array([float(r['retained_error'])*100 for r in curves]);assert np.isfinite(x).all() and np.isfinite(y).all()
   upper.extend(y.tolist())
   ax.plot(x,y,color=COLORS[model],lw=.9)
   p=points[ds,model];px=float(p['positive_rejection'])*100;py=float(p['retained_error'])*100;er=ci.get((ds,model,'retained_error'),{});lo=num(er.get('ci_low'));hi=num(er.get('ci_high'))
   if 'estimate' in er:assert abs(float(er['estimate'])-py/100)<1e-10
   if math.isfinite(lo) and math.isfinite(hi):
    ax.vlines(px,lo*100,hi*100,color=COLORS[model],lw=.9);upper.append(hi*100)
   xr=ci.get((ds,model,'positive_rejection'),{});xl=num(xr.get('ci_low'));xh=num(xr.get('ci_high'))
   if math.isfinite(xl) and math.isfinite(xh):ax.hlines(py,xl*100,xh*100,color=COLORS[model],lw=.9)
   ax.plot(px,py,'D',ms=3.7,color=COLORS[model],mfc=COLORS[model] if math.isfinite(lo) else 'white');ax.axhline(float(p['full_error'])*100,color=COLORS[model],ls=':',lw=.6,alpha=.5)
   displayed.append(dict(p,vertical_ci_low=lo,vertical_ci_high=hi,horizontal_ci_low=xl,horizontal_ci_high=xh))
  limit=max(10,10*math.ceil(max(upper)/10));ax.set_xlim(-3,103);ax.set_ylim(-.03*limit,1.03*limit);ax.set_xticks([0,25,50,75,100]);ax.set_yticks(np.linspace(0,limit,5));ax.set_xlabel('Source positives discarded (%)',fontsize=6);ax.set_ylabel('Retained-set error (%)',fontsize=6);title(ax,k,name(ds))
 export(fig,list(axes.flat),'Fig04_selection_error_positive_loss',args);write(args.output/'source_data/Fig04_top80.tsv',displayed);write(args.output/'source_data/Fig04_curves.tsv',data['curves'])

def fig5(data,args):
 stem='Fig05_cross_source_transfer'
 if data['transfer'] is None:
  src=HERE/'frozen_transfer';old='Fig5_transfer_valid_only_v3'
  for folder,suffix in [('PDF','pdf'),('SVG','svg'),('PNG_600dpi','png'),('TIFF_600dpi','tiff'),('preview','preview.png')]:
   target=args.output/folder/(stem+('.png' if folder=='preview' else '.'+suffix));shutil.copyfile(src/(old+'.'+suffix),target)
  shutil.copyfile(src/(old+'.alignment.json'),args.output/'QA'/(stem+'.alignment.json'));pdf_checks(args.output/'PDF'/(stem+'.pdf'),stem,args.output)
  for n in ['Fig5_main_display_data.tsv','Supplementary_full_policy_availability.tsv']:shutil.copyfile(src/n,args.output/'source_data'/n)
  print('FIGURE PASS',stem,'unchanged formal vector reuse; numerical replot available with --results-root',flush=True);return
 points=data['transfer'];intervals=data['transfer_ci'];index={(r['held_source'],r['model']):r for r in points};ci={(r['held_source'],r['model'],r['metric']):r for r in intervals};assert len(index)==30
 records=[index[s,m] for m in MODELS[:5] for s in SOURCES if index[s,m]['policy_status']=='feasible']
 fig,axes=plt.subplots(1,3,figsize=(183/25.4,175/25.4));fig.subplots_adjust(left=.285,right=.975,bottom=.09,top=.94,wspace=.38)
 for ax,(metric,label) in zip(axes,[('retained_error','a  Held-source error (%)'),('coverage','b  Coverage (%)'),('positive_rejection','c  Positive loss (%)')]):
  for i,r in enumerate(records):
   value=float(r[metric])*100;v=ci.get((r['held_source'],r['model'],metric),{});lo=num(v.get('ci_low'));hi=num(v.get('ci_high'));ci_point(ax,value,i,lo*100,hi*100,COLORS[r['model']],hollow=not math.isfinite(lo))
  ax.set_title(label,loc='left',fontsize=7,pad=10);ax.set_ylim(len(records)-.5,-.5);ax.set_yticks(range(len(records)))
  if ax is axes[0]:ax.set_yticklabels([name(r['model'])+' / '+name(r['held_source']) for r in records],fontsize=6)
  else:ax.set_yticklabels([])
  if metric=='retained_error':
   upper=[float(r[metric])*100 for r in records]+[float(v['ci_high'])*100 for key,v in ci.items() if key[2]==metric and math.isfinite(num(v['ci_high'])) and index[key[0],key[1]]['policy_status']=='feasible'];limit=max(10,5*math.ceil(max(upper)/5));ax.set_xlim(-.03*limit,1.03*limit);ax.set_xticks(list(range(0,int(limit)+1,5 if limit<=30 else 10)));ax.axvline(5,color='#999999',ls='--',lw=.6,zorder=0)
  else:ax.set_xlim(-3,103);ax.set_xticks([0,25,50,75,100])
  for i in range(len(records)-1):
   if records[i]['model']!=records[i+1]['model']:ax.axhline(i+.5,color='#dddddd',lw=.4,zorder=0)
  ax.tick_params(axis='y',length=0)
 export(fig,list(axes),stem,args);write(args.output/'source_data/Fig05_feasible_combinations.tsv',records);supplementary=[]
 for r in points:supplementary.append(dict(r,display_destination='Main figure and supplement' if r['policy_status']=='feasible' else 'Supplement only',reason='Observed-score threshold found' if r['policy_status']=='feasible' else 'No observed-score threshold met all development constraints'))
 for s in SOURCES:supplementary.append(dict(held_source=s,model='c_AMPs',policy_status='ineligible_score_definition',display_destination='Supplement only',reason='No verified native continuous ensemble probability'))
 write(args.output/'source_data/Supplementary_full_policy_availability.tsv',supplementary)

def activity_panel(ax,data,cohort,letter,label):
 meta=one(data['new_groups'],cohort=cohort);positive=int(float(one(data['new_display'],cohort=cohort,model='Macrel',metric='end_to_end_positive_detection')['denominator']));negative=int(one(data['new_summary'],cohort=cohort)['generated_assay_negative']);title(ax,letter,label,pad=23)
 ax.text(0,1.035,f"n={meta['n']}; P={positive}; N={negative}; blocks={meta['blocks']}",transform=ax.transAxes,fontsize=6,ha='left',va='bottom')
 for i,model in enumerate(MODELS):
  for metric,color,shift in [('end_to_end_positive_detection','#247b89',-.18),('assay_FPR_supported','#c27136',.18)]:
   r=one(data['new_display'],cohort=cohort,model=model,metric=metric);den=float(r['denominator'])
   if den==0:continue
   value=float(r['value']);assert abs(value-float(r['numerator'])/den)<1e-8;low=num(r['ci_low']);high=num(r['ci_high']);valid=math.isfinite(low) and math.isfinite(high)
   ci_point(ax,value*100,i+shift,low*100,high*100,color,hollow=not valid,ms=3.4);ax.text(1.035,i+shift,f"{float(r['numerator']):.0f}/{den:.0f}",transform=ax.get_yaxis_transform(),fontsize=5.5,color=color,ha='left',va='center')
 ax.set_yticks(range(6),[name(m) for m in MODELS]);ax.set_ylim(5.5,-.5);ax.set_xlim(-3,103);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Positive-call proportion (%)')

def fig6(data,args):
 D=data['core'];studies=['APEX_selected','APEX_scoring_selected','AMPGen'];labels=['APEX: model-selected (P=41)','APEX: scoring-selected (P=12)','AMPGen (P=31)'];colors=['#497f98','#698f68','#9275a5'];markers=['o','s','^']
 fig,axes=plt.subplots(2,2,figsize=(183/25.4,172/25.4));fig.subplots_adjust(left=.12,right=.91,bottom=.10,top=.81,wspace=.65,hspace=.75)
 fig.legend(handles=[Line2D([],[],color=c,marker=m,ls='none',ms=3.5,label=l) for c,m,l in zip(colors,markers,labels)],loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,fontsize=5.8,columnspacing=1.2)
 fig.legend(handles=[Line2D([],[],color=c,marker='o',ls='none',ms=3.5,label=l) for c,l in [('#247b89','Activity-positive sequences'),('#c27136','Assay-negative sequences')]],loc='upper center',bbox_to_anchor=(.52,.95),ncol=2,fontsize=6,columnspacing=1.7)
 a,b=axes[0]
 for i,model in enumerate(MODELS):
  for j,(study,color,marker) in enumerate(zip(studies,colors,markers)):
   r=one(D['external'],study=study,tool=model);offset=(j-1)*.25;coverage=100*float(r['supported_positive_n'])/float(r['positive_evidence_n']);a.plot(coverage,i+offset,marker=marker,ls='none',ms=3.1,color=color);lo=num(r['ci95_low']);hi=num(r['ci95_high']);ci_point(b,100*float(r['recall']),i+offset,100*lo,100*hi,color,marker,hollow=not math.isfinite(lo),ms=3.1)
 for ax in [a,b]:ax.set_yticks(range(6),[name(m) for m in MODELS]);ax.set_ylim(5.6,-.6);ax.set_xlim(-3,103);ax.set_xticks([0,25,50,75,100])
 title(a,'a','Positive-input coverage (%)',pad=23);title(b,'b','Recall on supported positives (%)',pad=23)
 for ax in [a,b]:ax.text(0,1.035,'Original APEX and AMPGen panels',transform=ax.transAxes,fontsize=6,va='bottom')
 a.set_xlabel('Supported / activity-positive (%)');b.set_xlabel('Detected / supported positives (%)')
 activity_panel(axes[1,0],data,'ApexGO_template_based','c','ApexGO: template-based');activity_panel(axes[1,1],data,'ApexGO_template_free','d','ApexGO: template-free')
 export(fig,list(axes.flat),'Fig06_experimental_validation',args);write(args.output/'source_data/Fig06_original_panel_metrics.tsv',D['external']);write(args.output/'source_data/Fig06_new_panel_metrics.tsv',[r for r in data['new_display'] if r['cohort'].startswith('ApexGO')])
 fig,axes=plt.subplots(1,2,figsize=(183/25.4,92/25.4));fig.subplots_adjust(left=.12,right=.91,bottom=.20,top=.72,wspace=.65)
 fig.legend(handles=[Line2D([],[],color=c,marker='o',ls='none',ms=3.5,label=l) for c,l in [('#247b89','Activity-positive sequences'),('#c27136','Assay-negative sequences')]],loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,fontsize=6)
 activity_panel(axes[0],data,'HydrAMP_auxiliary_preselected','a','HydrAMP: auxiliary selection');activity_panel(axes[1],data,'HydrAMP_no_auxiliary_preselection','b','HydrAMP: no auxiliary selection')
 export(fig,list(axes),'Supp_HydrAMP_small_cohorts',args);write(args.output/'source_data/Supplementary_HydrAMP.tsv',[r for r in data['new_display'] if r['cohort'].startswith('HydrAMP')])

def legends(args,data):
 text=(HERE/'Figure_legends_EN.txt').read_text(encoding='utf-8')
 if not args.results_root:text+='\nLocal review provenance: Figure 5 is an unchanged copy of the verified formal vector output because numerical intervals remain on the server. Figure 4 local-review vertical intervals use previously frozen formal 10000-replicate risk-at-80% estimates with matching point values; horizontal intervals are absent. With --results-root, Figure 5 is regenerated numerically and Figure 4 reads the current formal selection intervals.\n'
 (args.output/'Figure_legends_EN.txt').write_text(text,encoding='utf-8')

def finish(args,data):
 import pymupdf
 merged=pymupdf.open();pdfs=sorted((args.output/'PDF').glob('Fig0*.pdf'));assert len(pdfs)==6
 for p in pdfs:
  doc=pymupdf.open(p);merged.insert_pdf(doc);doc.close()
 merged.save(args.output/'All_six_main_figures.pdf');merged.close()
 fig,axes=plt.subplots(2,3,figsize=(15,11))
 for ax,p in zip(axes.flat,sorted((args.output/'preview').glob('Fig0*.png'))):ax.imshow(plt.imread(p));ax.set_title(p.stem,fontsize=10);ax.axis('off')
 fig.tight_layout();fig.savefig(args.output/'Six_figures_overview.png',dpi=300);plt.close(fig)
 assert data['fingerprints']=={p:sha(p) for p in data['fingerprints']}
 (args.output/'QA/input_sha256.json').write_text(json.dumps(data['fingerprints'],indent=2),encoding='utf-8')
 (args.output/'QA/run_receipt.json').write_text(json.dumps(dict(six_main_figures_completed=True,source_data_unchanged=True,model_inference=False,new_bootstrap=False,transfer_mode='numerical formal exact-score results' if args.results_root else 'unchanged verified formal vector',selected_transfer=data.get('transfer_root','frozen transfer vector'),results_root=str(args.results_root) if args.results_root else None),indent=2),encoding='utf-8')
 print('ALL SIX MAIN FIGURES COMPLETE',args.output,flush=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results-root',type=Path);parser.add_argument('--transfer-root',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reuse-frozen-transfer',action='store_true',help='Local review: preserve verified formal Figure 5 vector when numerical intervals remain on server');parser.add_argument('--only',choices=['all','1','2','3','4','5','6'],default='all')
 args=parser.parse_args();args.output=args.output.resolve()
 manifest=json.loads((HERE/'input_manifest.json').read_text(encoding='utf-8'))
 for relative,expected in manifest.items():
  p=(HERE/relative).resolve();p.relative_to(HERE.resolve());assert sha(p)==expected,'Bundle checksum mismatch: '+relative
 print('BUNDLE HASHES PASS',len(manifest),flush=True)
 args.output.mkdir(parents=True,exist_ok=False)
 for d in ['PDF','SVG','PNG_600dpi','TIFF_600dpi','preview','source_data','QA']:(args.output/d).mkdir()
 data=load_inputs(args);functions=[fig1,fig2,fig3,fig4,fig5,fig6]
 if args.only=='all':
  for fn in functions:fn(data,args)
  legends(args,data);finish(args,data)
 else:
  functions[int(args.only)-1](data,args);legends(args,data);print('SELECTED FIGURE COMPLETE',args.only,args.output,flush=True)

if __name__=='__main__':main()
