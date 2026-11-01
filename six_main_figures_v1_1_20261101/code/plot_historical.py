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
HERE=Path(__file__).resolve().parents[1]/'inputs/historical'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
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
 b.legend(handles=[Patch(fc='#dce4e9',label='Evaluated members'),Patch(fc='#4b8498',label='Activity-positive')],loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,fontsize=5.5,handlelength=1);title(b,'b','Experimentally tested sequences',pad=27)
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
 write(args.output/'source_data/Fig01_cohorts.tsv',cohorts);write(args.output/'source_data/Fig01_membership_patterns.tsv',patterns);write(args.output/'source_data/Supplementary_inventory_retention.tsv',read(HERE/'inventory_retention.tsv'))

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

def pdf_checks(*args): pass
