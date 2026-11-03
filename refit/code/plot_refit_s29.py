"""Rebuild S29 from verified saved estimates; no fitting or resampling."""
from pathlib import Path
import sys,csv,json,subprocess,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
SKILL=ROOT/'tools'
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def main():
 data=rows(ROOT/'data/refit_intervals_80.tsv')
 models=['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP']
 sources=['AMPlify','tsAMP_I','c_AMPs_benchmark_resplit','HMD_AMP','AmPEP_raw_source','ampir_raw_mature90']
 disp=dict(zip(models,['Macrel','AMPlify','ampir','tsAMP-I','HMD-AMP']))
 src=dict(zip(sources,['AMPlify','tsAMP-I','c_AMPs','HMD-AMP','AmPEP','ampir']))
 key={(r['held_source'],r['model'],r['metric'],r['uncertainty_scope']):r for r in data}
 assert len(key)==270
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
 fig,axs=plt.subplots(1,3,figsize=(7.2,8.0))
 fig.subplots_adjust(left=.235,right=.985,bottom=.095,top=.895,wspace=.28)
 labels=[f'{disp[m]} / {src[s]}' for m in models for s in sources]
 plotrows=[]
 for j,(metric,title) in enumerate([('conditional_error','Conditional error'),('positive_rejection','Positive rejection'),('end_to_end_detection','End-to-end detection')]):
  ax=axs[j];ext=[]
  for i,(m,s) in enumerate((m,s) for m in models for s in sources):
   for scope,color,offset,lw,ms in [('conditional','#969da3',-.14,.65,2.3),('joint','#267d96',.14,1.05,3.0)]:
    r=key[s,m,metric,scope];v=100*float(r['estimate']);lo=100*float(r['ci_low']);hi=100*float(r['ci_high']);y=29-i+offset
    valid=np.isfinite(lo) and np.isfinite(hi)
    if valid:ax.plot([lo,hi],[y,y],color=color,lw=lw,zorder=2);ext.extend([lo,hi])
    ax.plot(v,y,'o',ms=ms,mec=color,mfc=color if valid else 'white',mew=.7,zorder=3);ext.append(v)
    plotrows.append(dict(held_source=s,model=m,metric=metric,scope=scope,estimate_pp=v,ci_low_pp=lo,ci_high_pp=hi,status=r['interval_status']))
  ax.axvline(0,color='#c5cbd0',lw=.55,ls='--',zorder=0)
  for y in [5.5,11.5,17.5,23.5]:ax.axhline(y,color='#e7eaed',lw=.5,zorder=0)
  ax.set_ylim(-.8,29.8);ax.set_yticks(range(29,-1,-1),labels if j==0 else ['']*30);ax.tick_params(axis='y',length=0,pad=4)
  low=min(ext+[0]);high=max(ext+[0]);span=max(high-low,2);ax.set_xlim(low-.07*span,high+.07*span)
  ax.set_xlabel('Paired change (pp)');ax.set_title(f'{chr(97+j)}  {title}',loc='left',fontsize=7,pad=10)
 fig.text(.235,.978,'Reference-distance addition at matched 80% retention',fontsize=8,weight='bold')
 handles=[Line2D([0],[0],color=c,marker='o',markersize=3,lw=1,label=l) for c,l in [('#267d96','Development refit + held-source resampling'),('#969da3','Fixed fit + held-source resampling')]]
 fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.222,.957),ncol=1,fontsize=6,handlelength=2,labelspacing=.5)
 fig.text(.235,.045,'C + L + I minus C + L. Pointwise 95% intervals from 2,000 shared draws; no multiplicity adjustment.',fontsize=5.8)
 fig.text(.235,.022,'C, decision confidence; L, length; I, available-inventory distance. Open points: interval omitted.',fontsize=5.8)
 prefix=ROOT/'figures/Figure_S29_refit_updated'
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,axes=list(axs),json_out=str(prefix)+'.alignment.json',overlay_svg=str(ROOT/'qa/S29_alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
 for ext in ['pdf','svg']:fig.savefig(str(prefix)+'.'+ext)
 fig.savefig(str(prefix)+'.png',dpi=600)
 fig.savefig(str(prefix)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 plt.close(fig)
 with (ROOT/'data/Figure_S29_plotted_values.tsv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(plotrows[0]),delimiter='\t');w.writeheader();w.writerows(plotrows)
 for script,args in [('validate_figure.py',[str(Path(__file__))]),('audit_pdf_text.py',[str(prefix)+'.pdf','--min-pt','5']),('audit_figure_collisions.py',[str(prefix)+'.pdf','--json-out',str(ROOT/'qa/S29_collision.json')])]:
  p=subprocess.run([sys.executable,str(SKILL/script)]+args,capture_output=True,text=True,encoding='utf8')
  (ROOT/'qa'/f'S29_{script}.log').write_text(p.stdout+p.stderr,encoding='utf8')
  if p.returncode:raise RuntimeError(script+' failed; inspect QA log')
 print('S29 FIGURE PASS; 180 plotted interval states, 30 comparisons, three metrics')
if __name__=='__main__':main()
