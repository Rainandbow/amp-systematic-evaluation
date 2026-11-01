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

def native(cohort,subset='all_generated',model='Macrel'):return one(DG,cohort=cohort,subset=subset,model=model,task='any_eligible_measured_strain')

def nativeci(cohort,model,metric,subset='all_generated'):return one(DC,cohort=cohort,subset=subset,model=model,task='any_eligible_measured_strain',metric=metric)

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
 b.set(xlim=(-3,103),xticks=[0,50,100],xlabel='Fixed-rule positive calls (%)');title(b,'b','ApexGO fixed-rule decisions')
 c.axvline(.5,color='#888888',ls=':',lw=.7);c.set(xlim=(-.04,1.04),xticks=[0,.5,1],xlabel='AUROC');title(c,'c','ApexGO score ranking')
 d.axvline(86,color='#247b89',ls=':',lw=.6);d.axvline(33,color='#B77842',ls=':',lw=.6);d.set(xlim=(-3,103),xticks=[0,50,100],xlabel='Hit rate at 20-candidate budget (%)');title(d,'d','Any strain versus E. coli 11775')
 for ax in axs.flat:ax.set_yticks(range(6),[NAMES.get(m,m) for m in MODELS]);ax.set_ylim(5.5,-.5);ax.tick_params(axis='y',length=0)
 fig.legend(handles=[Line2D([],[],color='#9BA7AF',marker='s',ls='none',label='Supported (a)'),Line2D([],[],color='#247b89',marker='o',ls='none',label='Detected / any-strain positive'),Line2D([],[],color='#B77842',marker='s',ls='none',label='Threshold-negative (b) / E. coli (d)')],loc='upper center',bbox_to_anchor=(.53,.995),ncol=1,fontsize=6)
 save(fig,list(axs.flat),'Fig06_coverage_activity_and_task',out);write(out/'source_data/Fig06_original84_counts.tsv',counts);write(out/'source_data/Fig06_ApexGO_diagnostics.tsv',[r for r in DG if r['cohort']==c0 and r['subset']=='all_generated' and r['task']=='any_eligible_measured_strain']);write(out/'source_data/Fig06_task_budget.tsv',[r for r in TB if r['cohort']==c0 and r['subset']=='all_active_design' and r['threshold']=='64' and r['budget']=='0.2' and r['policy']=='score_topK'])
F=ROOT/'inputs/fig6'
DG=read(F/'task_diagnostics.tsv');DC=read(F/'diagnostic_intervals.tsv');TB=read(F/'task_budget.tsv');TC=read(F/'task_intervals.tsv')
