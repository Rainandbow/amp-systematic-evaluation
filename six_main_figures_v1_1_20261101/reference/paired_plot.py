"""Frozen-output figures: absolute recovery, paired hit gain, and family discrimination."""
from pathlib import Path
import csv,sys,math,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'tools'))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'legend.frameon':False,'savefig.facecolor':'white'})
MODELS=['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP','c_AMPs']
BASELINES=['length_high','length_low','KR_minus_DE_high','KR_minus_DE_low','mean_Kyte_Doolittle_high','mean_Kyte_Doolittle_low']
LABELS={'tsAMP_I':'tsAMP-I','HMD_AMP':'HMD-AMP','c_AMPs':'c_AMPs*','length_high':'Length (high)','length_low':'Length (low)','KR_minus_DE_high':'KR-DE (high)','KR_minus_DE_low':'KR-DE (low)','mean_Kyte_Doolittle_high':'Hydropathy (high)','mean_Kyte_Doolittle_low':'Hydropathy (low)'}
TEAL='#257E91';GREY='#7C858D';ORANGE='#AD6A32'
def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
pairs=read(OUT/'tables/Table_S12_all_48_paired_hit_differences.tsv')
intervals=read(OUT/'inputs/baseline_and_model_intervals.tsv')
points=read(OUT/'inputs/baseline_and_model_points.tsv')
PRIMARY='ApexGO_template_based';ANY='any_eligible_measured_strain';REF='source_reference_Ecoli'

def title(ax,letter,label,y=1.065):
 ax.text(0,y,letter,transform=ax.transAxes,fontweight='bold',fontsize=8,ha='left',va='bottom',gid='panel-label')
 ax.text(.065,y,label,transform=ax.transAxes,fontsize=7.5,ha='left',va='bottom')

def forest(ax,rows,methods,value='estimate',colors=True):
 for i,m in enumerate(methods):
  r=next(r for r in rows if r.get('method',r.get('model'))==m)
  x=float(r[value]);lo=float(r['ci_low']);hi=float(r['ci_high'])
  color=TEAL if m in MODELS else GREY
  ok=math.isfinite(lo) and math.isfinite(hi)
  if ok:ax.hlines(i,lo,hi,color=color,linewidth=.9,zorder=2)
  ax.plot(x,i,'o',ms=3.8,mfc=color if ok else 'white',mec=color,mew=.8,zorder=3)
 ax.set_yticks(range(len(methods)),[LABELS.get(m,m) for m in methods])
 ax.set_ylim(len(methods)-.35,-.65)
 ax.tick_params(axis='y',length=0,pad=5)
 ax.spines['left'].set_visible(False)

def save(fig,stem,axes,letters):
 target=OUT/'figures'/stem
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,axes=list(axes),panel_ids=letters,json_out=str(target)+'.alignment.json',overlay_svg=str(target)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
 fig.savefig(str(target)+'.pdf')
 fig.savefig(str(target)+'.svg')
 fig.savefig(str(target)+'.png',dpi=600)
 fig.savefig(str(target)+'.tiff',dpi=600)
 plt.close(fig)

# Main Figure 4 retains absolute any-strain recovery and both discrimination panels;
# panel b directly displays the reference-task paired gain rather than marginal bars.
fig,axs=plt.subplots(2,2,figsize=(183/25.4,178/25.4))
fig.subplots_adjust(left=.18,right=.985,bottom=.075,top=.94,wspace=.59,hspace=.49)
methods=MODELS+BASELINES
rr=[r for r in intervals if r['cohort']==PRIMARY and r['task']==ANY and r['metric']=='positive_hits']
forest(axs[0,0],rr,methods)
axs[0,0].axvline(17.2,color=ORANGE,ls=(0,(3,2)),lw=.8,zorder=0)
axs[0,0].set_xlim(-.4,20.5);axs[0,0].set_xticks([0,5,10,15,20]);axs[0,0].set_xlabel('Positive hits at K = 20')
title(axs[0,0],'a','Any measured strain')
rr=[r for r in pairs if r['cohort']==PRIMARY and r['task']==REF]
forest(axs[0,1],rr,MODELS,value='difference')
axs[0,1].axvline(0,color=GREY,ls=(0,(3,2)),lw=.8,zorder=0)
axs[0,1].set_xlim(-16,11);axs[0,1].set_xticks([-15,-10,-5,0,5,10]);axs[0,1].set_xlabel('Hits relative to shortest-first')
title(axs[0,1],'b','Reference E. coli gain')
for ax,metric,letter,label in [(axs[1,0],'pooled_AUROC','c','Between all peptides'),(axs[1,1],'macro_within_family_AUROC','d','Within comparable families')]:
 rr=[r for r in intervals if r['cohort']==PRIMARY and r['task']==REF and r['metric']==metric]
 forest(ax,rr,methods)
 ax.axvline(.5,color='#A9AFB3',ls=(0,(3,2)),lw=.8,zorder=0)
 ax.set_xlim(-.025,1.025);ax.set_xticks(np.arange(0,1.01,.2));ax.set_xlabel('Pooled AUROC' if letter=='c' else 'Mean within-family AUROC')
 title(ax,letter,label)
save(fig,'Figure_4_updated_paired_gain',axs.flat,list('abcd'))

# Full named-baseline display retains all four cohorts and both activity tasks.
cohorts=[PRIMARY,'ApexGO_template_free','HydrAMP_auxiliary_preselected','HydrAMP_no_auxiliary_preselection']
clabel=['ApexGO template-based','ApexGO template-free','HydrAMP preselected','HydrAMP unselected']
lims=[(-16,14),(-3,4),(-3,3),(-.3,2.3)]
fig,axs=plt.subplots(4,2,figsize=(183/25.4,194/25.4))
fig.subplots_adjust(left=.155,right=.985,bottom=.065,top=.915,wspace=.49,hspace=.95)
for i,cohort in enumerate(cohorts):
 for j,task in enumerate([ANY,REF]):
  ax=axs[i,j];rr=[r for r in pairs if r['cohort']==cohort and r['task']==task]
  assert len(rr)==6
  forest(ax,rr,MODELS,value='difference')
  ax.axvline(0,color=GREY,ls=(0,(3,2)),lw=.8,zorder=0)
  ax.set_xlim(*lims[i]);ax.xaxis.set_major_locator(MultipleLocator(5 if i==0 else 1))
  title(ax,'abcdefgh'[i*2+j],clabel[i]+(' | Any strain' if j==0 else ' | E. coli'),y=1.20)
  r=rr[0];note=f"n={r['n']}; P={r['positive']}; K={int(float(r['budget_slots']))}; blocks={r['blocks']}"
  ax.text(0,1.005,note,transform=ax.transAxes,ha='left',va='bottom',fontsize=6)
  if i==3:ax.set_xlabel('Hits relative to shortest-first')
save(fig,'Figure_S26_all_paired_hit_differences',axs.flat,list('abcdefgh'))
print('PLOTS COMPLETE',str(OUT/'figures'))
