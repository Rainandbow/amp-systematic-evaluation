"""Figure5: observed source risk decomposition at frozen exact-v3 thresholds.
All segments in panel a are five-source ranges, not confidence intervals.
"""
from pathlib import Path
import csv,json,math,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import transforms
from audit_panel_alignment import require_matplotlib_panel_alignment

PACKAGE=Path(__file__).resolve().parents[1];ROOT=PACKAGE/'frozen/source_decomposition';OUT=PACKAGE/'rebuild_figures/_source_fig5';QA=OUT/'QA';OUT.mkdir(parents=True,exist_ok=True);QA.mkdir(exist_ok=True)

def read(name):
 with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
rows=read('06_图5来源数据.tsv');source=read('01_逐折来源分解表_EN.tsv')
MODEL_ORDER=['Macrel','AMPlify','ampir','tsAMP_I'];SOURCE_ORDER=['AMPlify','tsAMP_I','c_AMPs_benchmark_resplit','HMD_AMP','AmPEP_raw_source','ampir_raw_mature90']
NAMES={'Macrel':'Macrel','AMPlify':'AMPlify','ampir':'ampir','tsAMP_I':'tsAMP-I','HMD_AMP':'HMD-AMP','c_AMPs_benchmark_resplit':'c_AMPs','AmPEP_raw_source':'AmPEP','ampir_raw_mature90':'ampir'}
rows.sort(key=lambda r:(MODEL_ORDER.index(r['model']),SOURCE_ORDER.index(r['held_source'])))
assert len(rows)==23 and len({(r['held_source'],r['model']) for r in rows})==23
ys=[];gaps=[];position=0;last=None
for r in rows:
 if last is not None and r['model']!=last:gaps.append(position-.35);position+=.7
 ys.append(position);position+=1;last=r['model']
ys=np.asarray(ys)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':6.4,'axes.labelsize':7,'axes.titlesize':7.5,'xtick.labelsize':6,'ytick.labelsize':6.3,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'legend.fontsize':6,'savefig.facecolor':'white'})
NAVY='#24566B';GREY='#A1A7AA';DARK='#545D62';ORANGE='#C16D36';BLUE='#477F9A';LIGHT='#D9DDDF'
fig,axs=plt.subplots(1,3,figsize=(183/25.4,150/25.4),sharey=True)
fig.subplots_adjust(left=.225,right=.985,bottom=.18,top=.925,wspace=.24)
for j,ax in enumerate(axs):
 ax.set_ylim(ys[-1]+.65,-.65);ax.tick_params(axis='y',length=0,pad=4);ax.tick_params(axis='x',length=2.5,pad=2)
 ax.set_yticks(ys);ax.grid(False)
 for gap in gaps:ax.axhline(gap+.35,color='#E2E5E6',linewidth=.5,zorder=0)
 if j:ax.tick_params(axis='y',labelleft=False);ax.spines['left'].set_visible(False)
labels=[NAMES[r['model']]+' / '+NAMES[r['held_source']] for r in rows]
axs[0].set_yticklabels(labels)
for i,(r,y) in enumerate(zip(rows,ys)):
 dev=[d for d in source if d['held_source']==r['held_source'] and d['model']==r['model'] and d['role']=='development']
 vals=np.array([100*float(d['conditional_error']) for d in dev if np.isfinite(float(d['conditional_error']))]);assert len(vals)==5
 axs[0].plot([min(vals),max(vals)],[y,y],color=LIGHT,lw=1.25,zorder=1)
 axs[0].scatter(vals,np.full(5,y),s=11,facecolor='white',edgecolor=GREY,linewidth=.55,zorder=2)
 axs[0].plot(100*float(r['development_error']),y,marker='D',ms=3,color=NAVY,linestyle='None',zorder=3)
 held=100*float(r['held_retained_error']);axs[1].plot(held,y,'o',ms=3.5,color=ORANGE if held>5+1e-9 else NAVY,zorder=3)
 axs[2].plot(100*float(r['held_coverage']),y,'o',ms=3.2,color=BLUE,zorder=3)
 axs[2].plot(100*float(r['held_positive_rejection']),y,'^',ms=3.4,color=ORANGE,zorder=3)
for ax in axs[:2]:
 ax.set_xlim(-1,35);ax.set_xticks([0,10,20,30]);ax.axvline(5,color=DARK,ls=(0,(3,2)),lw=.65,zorder=0)
 ax.set_xlabel('Conditional error (%)',labelpad=4)
axs[2].set_xlim(-2,102);axs[2].set_xticks([0,25,50,75,100]);axs[2].set_xlabel('Fraction (%)',labelpad=4)
axs[0].legend(handles=[Line2D([0],[0],color=NAVY,marker='D',linestyle='None',ms=3,label='Weighted mixture'),Line2D([0],[0],color=LIGHT,marker='o',mfc='white',mec=GREY,ms=3,lw=1.25,label='Five sources (range)')],loc='upper center',bbox_to_anchor=(.5,-.105),borderaxespad=0,handletextpad=.5,labelspacing=.5)
axs[1].legend(handles=[Line2D([0],[0],color=NAVY,marker='o',linestyle='None',ms=3.2,label='Error <=5%'),Line2D([0],[0],color=ORANGE,marker='o',linestyle='None',ms=3.2,label='Error >5%')],loc='upper center',bbox_to_anchor=(.5,-.105),borderaxespad=0,handletextpad=.5,labelspacing=.5)
axs[2].legend(handles=[Line2D([0],[0],color=BLUE,marker='o',linestyle='None',ms=3.2,label='Coverage'),Line2D([0],[0],color=ORANGE,marker='^',linestyle='None',ms=3.4,label='Positives discarded')],loc='upper center',bbox_to_anchor=(.5,-.105),borderaxespad=0,handletextpad=.5,labelspacing=.5)
for ax,letter,title in zip(axs,['a','b','c'],['Development sources','Held-source error','Selection cost']):
 offset=transforms.ScaledTranslation(0,8/72,fig.dpi_scale_trans)
 ax.text(0,1,letter,transform=ax.transAxes+offset,fontsize=8,fontweight='bold',va='bottom',ha='left')
 ax.text(.1,1,title,transform=ax.transAxes+offset,fontsize=7.2,va='bottom',ha='left')
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=QA/'Fig5_source_decomposition.alignment.json',overlay_svg=QA/'Fig5_source_decomposition.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(OUT/'Fig5_source_decomposition.pdf')
fig.savefig(OUT/'Fig5_source_decomposition.svg')
fig.savefig(OUT/'Fig5_source_decomposition.png',dpi=600)
fig.savefig(OUT/'Fig5_source_decomposition.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(OUT/'Fig5_source_decomposition.preview.png',dpi=300)
plt.close(fig)
(QA/'render_receipt.json').write_text(json.dumps(dict(backend='python_matplotlib',matplotlib=matplotlib.__version__,numpy=np.__version__,width_mm=183,height_mm=150,rows=23,point_only=True,development_segments='observed range of five source errors, not CI',original_bootstrap_intervals_replaced=False,layout='equal-width equal-height comparable row of three panels',excluded_from_main={'infeasible':7,'score_ineligible':6},exported=['PDF','SVG','PNG600dpi','TIFF600dpi']),indent=2),encoding='utf-8')
print('PLOTTED',OUT/'Fig5_source_decomposition.pdf')
