"""Build all six main and nineteen supplementary figures from frozen v9 tables.
No model inference. Existing bootstrap intervals are read, never fabricated.
"""
from pathlib import Path
import os,sys,runpy,argparse,csv,json,math,shutil,importlib.util
import numpy as np
# Optional local desktop dependency path; a normal pip-installed environment needs none.
for p in os.environ.get('AMP_EXTRA_PYTHONPATH','').split(os.pathsep):
 if p:sys.path.insert(0,p)
if os.environ.get('AMP_EXTRA_PYTHONPATH'):os.environ['PYTHONPATH']=os.environ['AMP_EXTRA_PYTHONPATH']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'frozen'
MODELS=['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP','c_AMPs'];NAMES={'tsAMP_I':'tsAMP-I','HMD_AMP':'HMD-AMP','c_AMPs':'c_AMPs*'}
COLORS=['#0072B2','#00875A','#A66A00','#AA62A0','#407DB1','#D55E00'];COHORTS=['ApexGO_template_based','ApexGO_template_free','HydrAMP_auxiliary_preselected','HydrAMP_no_auxiliary_preselection'];CN=['ApexGO template-based','ApexGO template-free','HydrAMP auxiliary-preselected','HydrAMP without auxiliary preselection']

def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),delimiter='\t');w.writeheader();w.writerows(rows)
def num(x):
 try:return float(x)
 except (ValueError,TypeError):return np.nan
def one(rr,**kw):
 out=[x for x in rr if all(str(x.get(k,''))==str(v) for k,v in kw.items())];assert len(out)==1,kw;return out[0]
def setup(p):
 for n in ['PDF','SVG','PNG_600dpi','TIFF_600dpi','preview','QA','source_data']:(p/n).mkdir(parents=True,exist_ok=True)
def save(fig,axes,stem,out,ids=None,rows=None,cols=None):
 fig.canvas.draw();require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids or list('abcdef')[:len(axes)],row_groups=rows,column_groups=cols,json_out=out/'QA'/(stem+'.alignment.json'),strict=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
 fig.savefig(out/'PDF'/(stem+'.pdf'));fig.savefig(out/'SVG'/(stem+'.svg'));fig.savefig(out/'PNG_600dpi'/(stem+'.png'),dpi=600);fig.savefig(out/'TIFF_600dpi'/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'});fig.savefig(out/'preview'/(stem+'.png'),dpi=300);plt.close(fig);print('DRAW',stem,flush=True)
def title(ax,letter,text,pad=10):ax.set_title(letter+'  '+text,loc='left',fontsize=7,pad=pad)
def point(ax,x,y,lo=np.nan,hi=np.nan,color='#247b89',marker='o',open_missing=True):
 good=np.isfinite(num(lo)) and np.isfinite(num(hi))
 if good:ax.hlines(y,float(lo),float(hi),color=color,lw=.8)
 ax.plot(x,y,marker=marker,ls='none',ms=3.3,color=color,mfc='white' if open_missing and not good else color,mew=.7)
def import_file(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

BUD=F/'membership_analysis/active_design_151';TASK=F/'task_threshold'
BP=read(BUD/'budget_points.tsv');BC=read(BUD/'budget_intervals.tsv');DG=read(BUD/'task_diagnostics.tsv');DC=read(BUD/'diagnostic_intervals.tsv');TM=read(BUD/'task_membership.tsv');TS=read(TASK/'05_六模型原生分类与排序.tsv');TB=read(TASK/'06_固定预算结果.tsv');TC=read(TASK/'07_探索性分组Bootstrap区间.tsv')

def native(cohort,subset='all_generated',model='Macrel'):return one(DG,cohort=cohort,subset=subset,model=model,task='any_eligible_measured_strain')
def nativeci(cohort,model,metric,subset='all_generated'):return one(DC,cohort=cohort,subset=subset,model=model,task='any_eligible_measured_strain',metric=metric)
def main4(out):
 c=COHORTS[0];fig,axs=plt.subplots(1,2,figsize=(183/25.4,103/25.4));fig.subplots_adjust(left=.10,right=.98,bottom=.28,top=.80,wspace=.58)
 for m,col in zip(MODELS,COLORS):
  rr=sorted([r for r in BP if r['cohort']==c and r['task']=='any_eligible_measured_strain' and r['subset']=='all_generated' and r['policy']=='score_topK' and r['model']==m],key=lambda r:float(r['budget']))
  axs[0].plot([100*float(r['selected_fraction']) for r in rr],[100*float(r['positive_recovery']) for r in rr],'-o',lw=.9,ms=3,color=col,label=NAMES.get(m,m))
 axs[0].plot([0,60],[0,60],ls=':',color='#666666',lw=.8);axs[0].set(xlim=(0,55),ylim=(-2,65),xticks=[10,20,30,50],xlabel='Selected candidates (%)',ylabel='Activity positives recovered (%)');title(axs[0],'a','Budget and positive recovery')
 for i,(m,col) in enumerate(zip(MODELS,COLORS)):
  p=one(BP,cohort=c,task='any_eligible_measured_strain',subset='all_generated',policy='score_topK',model=m,budget='0.2');ci=one(BC,cohort=c,task='any_eligible_measured_strain',subset='all_generated',policy='score_topK',model=m,budget='0.2',metric='precision_gain_vs_random')
  point(axs[1],100*float(p['precision_gain_vs_random']),i,100*num(ci['ci_low']),100*num(ci['ci_high']),col)
 axs[1].axvline(0,ls=':',color='#666666',lw=.7);axs[1].set(ylim=(5.5,-.5),yticks=range(6),yticklabels=[NAMES.get(m,m) for m in MODELS],xlim=(-17,35),xlabel='Hit-rate gain over random (pp)');title(axs[1],'b','Gain at 20% budget')
 fig.text(.10,.94,'ApexGO template-based: n=100; P=86; threshold-negative=14; blocks=10',fontsize=7)
 fig.legend(handles=[Line2D([],[],color=c,marker='o',lw=.8,ms=3,label=NAMES.get(m,m)) for m,c in zip(MODELS,COLORS)],loc='lower center',bbox_to_anchor=(.52,.03),ncol=3,fontsize=6)
 save(fig,list(axs),'Fig04_budget_and_recovery',out);write(out/'source_data/Fig04_budget.tsv',[r for r in BP if r['cohort']==c and r['subset']=='all_generated' and r['task']=='any_eligible_measured_strain']);write(out/'source_data/Fig04_intervals.tsv',[r for r in BC if r['cohort']==c and r['subset']=='all_generated' and r['task']=='any_eligible_measured_strain'])

def main6(out,core):
 fig,axs=plt.subplots(2,2,figsize=(183/25.4,167/25.4));fig.subplots_adjust(left=.12,right=.89,bottom=.10,top=.88,wspace=.67,hspace=.70)
 a,b,c,d=axs.flat;old=core['external'];counts=[]
 for i,m in enumerate(MODELS):
  rr=[r for r in old if r['tool']==m];support=sum(int(r['supported_positive_n']) for r in rr) if 'supported_positive_n' in rr[0] else sum(int(r['positive_supported_n']) for r in rr);det=sum(int(r['detected_positive_n']) for r in rr) if 'detected_positive_n' in rr[0] else sum(int(r['predicted_positive_n']) for r in rr)
  counts.append(dict(model=m,eligible=84,supported=support,detected=det))
  for val,off,col,mk in [(support,-.15,'#9BA7AF','s'),(det,.15,'#247b89','o')]:
   point(a,100*val/84,i+off,color=col,marker=mk,open_missing=False);a.text(1.03,i+off,f'{val}/84',transform=a.get_yaxis_transform(),fontsize=5.5,va='center',color=col)
 a.set(xlim=(-3,103),xticks=[0,50,100],xlabel='All eligible positives (%)');title(a,'a','Input support and end-to-end detection')
 c0=COHORTS[0]
 for i,m in enumerate(MODELS):
  r=native(c0,model=m)
  for met,numer,den,off,col,mk in [('native_recall','TP','positive',-.15,'#247b89','o'),('assay_negative_positive_call_rate','FP','negative',.15,'#B77842','s')]:
   ci=nativeci(c0,m,met);v=float(r[met]);point(b,100*v,i+off,100*num(ci['ci_low']),100*num(ci['ci_high']),col,mk);b.text(1.03,i+off,f"{float(r[numer]):.0f}/{float(r[den]):.0f}",transform=b.get_yaxis_transform(),fontsize=5.5,va='center',color=col)
  ci=nativeci(c0,m,'AUROC');point(c,float(r['AUROC']),i,num(ci['ci_low']),num(ci['ci_high']),COLORS[i])
  for task,off,col,mk in [('any_eligible_measured_strain',-.14,'#247b89','o'),('source_reference_Ecoli',.14,'#B77842','s')]:
   r1=one(TB,cohort=c0,subset='all_active_design',task=task,threshold='64',model=m,budget='0.2',policy='score_topK');ci=one(TC,cohort=c0,subset='all_active_design',task=task,threshold='64',model=m,budget='0.2',policy='score_topK',metric='precision_at_budget',analysis='budget')
   point(d,100*float(r1['precision_at_budget']),i+off,100*num(ci['ci_low']),100*num(ci['ci_high']),col,mk)
 b.set(xlim=(-3,103),xticks=[0,50,100],xlabel='Native positive calls (%)');title(b,'b','ApexGO native decisions')
 c.axvline(.5,color='#888888',ls=':',lw=.7);c.set(xlim=(-.04,1.04),xticks=[0,.5,1],xlabel='AUROC');title(c,'c','ApexGO score ranking')
 d.axvline(86,color='#247b89',ls=':',lw=.6);d.axvline(33,color='#B77842',ls=':',lw=.6);d.set(xlim=(-3,103),xticks=[0,50,100],xlabel='Hit rate at 20-candidate budget (%)');title(d,'d','Any strain versus E. coli 11775')
 for ax in axs.flat:ax.set_yticks(range(6),[NAMES.get(m,m) for m in MODELS]);ax.set_ylim(5.5,-.5);ax.tick_params(axis='y',length=0)
 fig.legend(handles=[Line2D([],[],color='#9BA7AF',marker='s',ls='none',label='Supported (a)'),Line2D([],[],color='#247b89',marker='o',ls='none',label='Detected / any-strain positive'),Line2D([],[],color='#B77842',marker='s',ls='none',label='Threshold-negative (b) / E. coli (d)')],loc='upper center',bbox_to_anchor=(.53,.995),ncol=1,fontsize=6)
 save(fig,list(axs.flat),'Fig06_coverage_activity_and_task',out);write(out/'source_data/Fig06_original84_counts.tsv',counts);write(out/'source_data/Fig06_ApexGO_diagnostics.tsv',[r for r in DG if r['cohort']==c0 and r['subset']=='all_generated' and r['task']=='any_eligible_measured_strain']);write(out/'source_data/Fig06_task_budget.tsv',[r for r in TB if r['cohort']==c0 and r['subset']=='all_active_design' and r['threshold']=='64' and r['budget']=='0.2' and r['policy']=='score_topK'])

def s12(out):
 fig,axs=plt.subplots(1,2,figsize=(183/25.4,100/25.4));fig.subplots_adjust(left=.12,right=.90,bottom=.20,top=.77,wspace=.60)
 for ax,cohort,letter,text in zip(axs,COHORTS[2:],'ab',['Auxiliary preselection','No auxiliary preselection']):
  r0=native(cohort);title(ax,letter,text,pad=24);ax.text(0,1.035,f"n={r0['n']}; P={r0['positive']}; N={r0['negative']}; blocks={r0['blocks']}",transform=ax.transAxes,fontsize=6,va='bottom')
  for i,m in enumerate(MODELS):
   r=native(cohort,model=m)
   for field,col,off in [('native_recall','#247b89',-.15),('assay_negative_positive_call_rate','#B77842',.15)]:point(ax,100*float(r[field]),i+off,color=col)
  ax.set(xlim=(-3,103),xticks=[0,50,100],yticks=range(6),yticklabels=[NAMES.get(m,m) for m in MODELS],ylim=(5.5,-.5),xlabel='Native positive calls (%)')
 fig.legend(handles=[Line2D([],[],color=c,marker='o',ls='none',label=l) for c,l in [('#247b89','Activity-positive'),('#B77842','Threshold-negative')]],loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,fontsize=6)
 save(fig,list(axs),'FigS12_HydrAMP_small_cohorts',out);write(out/'source_data/FigS12_HydrAMP_v9.tsv',[r for r in DG if r['cohort'].startswith('HydrAMP') and r['subset']=='all_generated' and r['task']=='any_eligible_measured_strain'])

def s14(out):
 fig,axs=plt.subplots(2,2,figsize=(183/25.4,155/25.4));fig.subplots_adjust(left=.12,right=.97,bottom=.11,top=.86,wspace=.50,hspace=.78)
 avail=[]
 for ax,cohort,letter,label in zip(axs.flat,COHORTS,'abcd',CN):
  title(ax,letter,label.replace('without auxiliary preselection','no auxiliary selection'),pad=22)
  for sub,col,mk,off in [('all_generated','#247b89','o',-.14),('five_inventory_pass','#B77842','s',.14)]:
   rr=[r for r in DG if r['cohort']==cohort and r['subset']==sub and r['task']=='any_eligible_measured_strain']
   if not rr:
    n0,p0=0,0
   else:n0,p0=int(rr[0]['n']),int(rr[0]['positive'])
   avail.append(dict(cohort=cohort,subset=sub,n=n0,positive=p0,status='empty' if not n0 else ('no_positive' if not p0 else 'estimable')))
   if not p0:continue
   for i,m in enumerate(MODELS):
    r=one(rr,model=m);ci=nativeci(cohort,m,'native_recall',sub);point(ax,100*float(r['native_recall']),i+off,100*num(ci['ci_low']),100*num(ci['ci_high']),col,mk)
  allr=one(avail,cohort=cohort,subset='all_generated');pas=one(avail,cohort=cohort,subset='five_inventory_pass');ax.text(0,1.04,f"All n/P={allr['n']}/{allr['positive']}; pass={pas['n']}/{pas['positive']}",transform=ax.transAxes,fontsize=5.8,va='bottom')
  ax.set(xlim=(-3,103),xticks=[0,50,100],yticks=range(6),yticklabels=[NAMES.get(m,m) for m in MODELS],ylim=(5.5,-.5),xlabel='Detected / activity-positive (%)')
 fig.legend(handles=[Line2D([],[],color=c,marker=m,ls='none',label=l) for c,m,l in [('#247b89','o','Active-design candidates'),('#B77842','s','Five-inventory-pass subset')]],loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,fontsize=6)
 save(fig,list(axs.flat),'FigS14_inventory_sensitivity',out);write(out/'source_data/FigS14_availability_v9.tsv',avail)

def s15(out):
 fig,axs=plt.subplots(2,2,figsize=(183/25.4,155/25.4));fig.subplots_adjust(left=.12,right=.98,bottom=.15,top=.86,wspace=.42,hspace=.76)
 for ax,cohort,letter,label in zip(axs.flat,COHORTS,'abcd',CN):
  meta=native(cohort);title(ax,letter,label.replace('without auxiliary preselection','no auxiliary selection'),pad=22);ax.text(0,1.035,f"n={meta['n']}; P={meta['positive']}; N={meta['negative']}; blocks={meta['blocks']}",transform=ax.transAxes,fontsize=5.8,va='bottom')
  for m,col in zip(MODELS,COLORS):
   rr=sorted([r for r in BP if r['cohort']==cohort and r['task']=='any_eligible_measured_strain' and r['subset']=='all_generated' and r['policy']=='score_topK' and r['model']==m],key=lambda r:float(r['budget']));ax.plot([100*float(r['selected_fraction']) for r in rr],[100*float(r['positive_recovery']) for r in rr],'-o',lw=.8,ms=2.7,color=col)
  ax.plot([0,60],[0,60],ls=':',color='#777777',lw=.7);ax.set(xlim=(0,60),ylim=(-3,104),xticks=[10,20,30,50],xlabel='Selected candidates (%)',ylabel='Activity positives recovered (%)')
 fig.legend(handles=[Line2D([],[],color=c,marker='o',lw=.8,ms=3,label=NAMES.get(m,m)) for m,c in zip(MODELS,COLORS)],loc='lower center',bbox_to_anchor=(.52,.005),ncol=3,fontsize=6)
 save(fig,list(axs.flat),'FigS15_budget_all_cohorts',out);write(out/'source_data/FigS15_budget.tsv',[r for r in BP if r['task']=='any_eligible_measured_strain' and r['subset']=='all_generated'])

def s16(out):
 fig,axs=plt.subplots(2,2,figsize=(183/25.4,160/25.4));fig.subplots_adjust(left=.12,right=.86,bottom=.16,top=.87,wspace=.58,hspace=.85)
 for ax,cohort,letter,label in zip(axs.flat,COHORTS,'abcd',CN):
  tt=[16,32,64] if cohort.startswith('ApexGO') else [32,64,128];unit='uM' if cohort.startswith('ApexGO') else 'ug/mL';meta=native(cohort);z=np.full((6,6),np.nan)
  for i,m in enumerate(MODELS):
   for j,(task,t) in enumerate([(task,t) for task in ['any_eligible_measured_strain','source_reference_Ecoli'] for t in tt]):z[i,j]=num(one(TS,cohort=cohort,subset='all_active_design',task=task,threshold=str(t),model=m)['AUROC'])
  cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#EEEEEE');im=ax.imshow(np.ma.masked_invalid(z),vmin=0,vmax=1,cmap=cmap,aspect='auto')
  for i in range(6):
   for j in range(6):ax.text(j,i,f'{z[i,j]:.2f}' if np.isfinite(z[i,j]) else '\u2014',ha='center',va='center',fontsize=5.5,color='white' if z[i,j]>.65 else '#253B46')
  ax.set_yticks(range(6),[NAMES.get(m,m) for m in MODELS]);ax.set_xticks(range(6),[str(t) for t in tt]*2);ax.tick_params(length=0,pad=3);ax.text(.23,-.17,'Any strain',transform=ax.transAxes,ha='center',fontsize=5.8);ax.text(.76,-.17,'E. coli reference',transform=ax.transAxes,ha='center',fontsize=5.8);ax.set_xlabel('MIC threshold ('+unit+')',labelpad=21);title(ax,letter,label.replace('without auxiliary preselection','no auxiliary selection'),pad=22);ax.text(0,1.045,f"n={meta['n']}; blocks={meta['blocks']}",transform=ax.transAxes,fontsize=5.8,va='bottom')
  for sp in ax.spines.values():sp.set_visible(False)
 cax=fig.add_axes([.91,.30,.015,.42]);cb=fig.colorbar(im,cax=cax,ticks=[0,.5,1]);cb.ax.tick_params(labelsize=6);cb.ax.set_title('AUROC',fontsize=6,pad=7);cb.outline.set_visible(False)
 save(fig,list(axs.flat),'FigS16_activity_task_threshold',out);write(out/'source_data/FigS16_task_diagnostics.tsv',[r for r in TS if r['subset']=='all_active_design'])

def s19(out):
 scores=read(BUD/'sequence_score_diagnostics.tsv');fig,axs=plt.subplots(2,2,figsize=(183/25.4,150/25.4));fig.subplots_adjust(left=.12,right=.98,bottom=.12,top=.84,wspace=.49,hspace=.76)
 for ax,cohort,letter,label in zip(axs.flat,COHORTS,'abcd',CN):
  meta=native(cohort);title(ax,letter,label.replace('without auxiliary preselection','no auxiliary selection'),pad=22);ax.text(0,1.04,f"n={meta['n']}; P={meta['positive']}; N={meta['negative']}; blocks={meta['blocks']}",transform=ax.transAxes,fontsize=5.8,va='bottom')
  for i,m in enumerate(MODELS):
   for lab,col,off in [('1','#247b89',-.15),('0','#B77842',.15)]:
    rr=sorted([r for r in scores if r['cohort']==cohort and r['model']==m and r['label']==lab],key=lambda r:r['sequence_id']);v=[float(r['score']) for r in rr];jit=np.linspace(-.07,.07,len(rr)) if rr else []
    if rr:ax.scatter(v,i+off+np.array(jit),s=7,c=col,alpha=.55,linewidths=0)
  ax.set(xlim=(-.025,1.025),xticks=[0,.5,1],yticks=range(6),yticklabels=[NAMES.get(m,m) for m in MODELS],ylim=(5.5,-.5),xlabel='Saved AMP score / c_AMPs extension')
 fig.legend(handles=[Line2D([],[],color=c,marker='o',ls='none',label=l) for c,l in [('#247b89','Activity-positive'),('#B77842','Threshold-negative')]],loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,fontsize=6)
 save(fig,list(axs.flat),'FigS19_score_distributions',out);write(out/'source_data/FigS19_all906_scores.tsv',scores)


def run(out):
 main=out/'main';supp=out/'supplementary';setup(main);setup(supp)
 # Import the archived plotting implementations using their own frozen data roots.
 sys.path.insert(0,str(ROOT/'code/legacy_main/code'));M=import_file('main_legacy',ROOT/'code/legacy_main/generate_six_figures.py')
 datadir=ROOT/'code/legacy_main/data';D=json.loads((datadir/'frozen_core.json').read_text(encoding='utf-8'));data=dict(core=D,patterns=read(datadir/'exclusive_membership_patterns.tsv'),length=read(datadir/'length_mcc_intervals.tsv'),new_summary=read(datadir/'new_panel_summary.tsv'))
 for r in data['new_summary']:
  if r['cohort']=='HydrAMP_no_auxiliary_preselection':r['generated_candidates']='7';r['generated_positive']='2'
 args=argparse.Namespace(output=main)
 # Figure1 layout is retained; only role-aware member counts and the legend change.
 orig=Path(ROOT/'code/legacy_main/generate_six_figures.py').read_text(encoding='utf-8');fn=orig[orig.index('def fig1('):orig.index('\ndef fig2(')];fn=fn.replace("label='Tested'","label='Evaluated members'");exec(fn,M.__dict__)
 M.fig1(data,args);M.fig2(data,args);M.fig3(data,args);main4(main)
 # Rebuild the verified point-decomposition Figure5 using its bundled frozen table.
 script=ROOT/'code/plot_source_fig5.py';runpy.run_path(str(script),run_name='__main__');temp=ROOT/'rebuild_figures/_source_fig5'
 for ext,folder in [('pdf','PDF'),('svg','SVG'),('png','PNG_600dpi'),('tiff','TIFF_600dpi')]:shutil.copy2(temp/('Fig5_source_decomposition.'+ext),main/folder/('Fig05_cross_source_transfer.'+ext))
 shutil.copy2(temp/'Fig5_source_decomposition.preview.png',main/'preview/Fig05_cross_source_transfer.png')
 shutil.copy2(temp/'QA/Fig5_source_decomposition.alignment.json',main/'QA/Fig05_cross_source_transfer.alignment.json')
 write(main/'source_data/Fig05_source_decomposition.tsv',read(F/'source_decomposition/01_逐折来源分解表_EN.tsv'))
 main6(main,D)
 # Unchanged SI plots retain original frozen estimates; revised plots use v9 sets.
 sys.path.insert(0,str(ROOT/'code/legacy_supp/code'));GS=import_file('supp_legacy',ROOT/'code/legacy_supp/generate_supplement_v2.py');H=GS.H;H.OUT=supp;L=GS.L
 for f in [L.s1,L.s2,GS.s3,lambda:L.ranking('ap',4),lambda:L.ranking('auroc',5),L.s6,L.s7,L.s8,L.s9,L.s10,GS.s11]:f()
 s12(supp)
 formal=[r for r in read(F/'v8_supp/FigS13_complete_policy_results.tsv') if r['model']!='c_AMPs'];settings=json.loads((F/'v8_supp/FigS13_run_settings.json').read_text(encoding='utf-8'));GS.s13(formal,settings,read(F/'v8_supp/FigS13_threshold_stability.tsv'))
 s14(supp);s15(supp);s16(supp)
 # Preserve original selector-refit bootstrap geometry; do not reinterpret its bars.
 origfig=F/'original_transfer_figure'
 for ext,folder in [('pdf','PDF'),('svg','SVG'),('png','PNG_600dpi'),('tiff','TIFF_600dpi')]:shutil.copy2(origfig/('Fig5_transfer_valid_only_v3.'+ext),supp/folder/('FigS17_transfer_original_intervals.'+ext))
 shutil.copy2(origfig/'Fig5_transfer_valid_only_v3.png',supp/'preview/FigS17_transfer_original_intervals.png')
 # Former main Figure4 is retained as the full historical selection-cost SI display.
 data.update(selection=read(datadir/'selection_points.tsv'),curves=read(datadir/'selection_curves.tsv'),selection_ci=read(datadir/'legacy_selection_intervals.tsv'))
 M.fig4(data,argparse.Namespace(output=supp))
 for folder,ext in [('PDF','pdf'),('SVG','svg'),('PNG_600dpi','png'),('TIFF_600dpi','tiff'),('preview','png')]:
  (supp/folder/('Fig04_selection_error_positive_loss.'+ext)).replace(supp/folder/('FigS18_selection_error_positive_loss.'+ext))
 for p in list((supp/'QA').glob('Fig04_selection_error_positive_loss*')):p.replace(p.with_name(p.name.replace('Fig04_selection_error_positive_loss','FigS18_selection_error_positive_loss')))
 for n in ['curves','top80']:
  (supp/'source_data'/('Fig04_'+n+'.tsv')).replace(supp/'source_data'/('FigS18_'+n+'.tsv'))
 s19(supp)
 write(main/'source_data/Fig01_active_design_members.tsv',read(F/'evidence_adjudication/08_新增151活性设计序列_排除定向阴性对照.tsv'))
 print('ALL SIX MAIN AND NINETEEN SI FIGURES BUILT',out,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'rebuild_figures');a=ap.parse_args();run(a.output)
