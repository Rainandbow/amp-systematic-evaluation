"""Render formal saved estimates; no inference, refitting, or resampling."""
import argparse, csv, hashlib, json, math, os, subprocess, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'formal_analysis/figures'
import numpy as np
sys.path.insert(0, str(ROOT/'qa_tools'))

MODELS = ['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP','c_AMPs']
BASES = ['length_high','length_low','KR_minus_DE_high','KR_minus_DE_low',
         'mean_Kyte_Doolittle_high','mean_Kyte_Doolittle_low']
METHODS = MODELS + BASES
LABEL = dict(zip(METHODS, ['Macrel','AMPlify','ampir','tsAMP-I','HMD-AMP','c_AMPs*',
                          'Length (high)','Length (low)','KR-DE (high)','KR-DE (low)',
                          'Hydropathy (high)','Hydropathy (low)']))
COHORTS = ['ApexGO_template_based','ApexGO_template_free',
           'HydrAMP_auxiliary_preselected','HydrAMP_no_auxiliary_preselection']
SHORT_COHORT = ['ApexGO, template-based','ApexGO, template-free',
                'HydrAMP, preselected','HydrAMP, unselected']
TASKS = ['any_eligible_measured_strain','source_reference_Ecoli']
SOURCES = ['AMPlify','tsAMP_I','c_AMPs_benchmark_resplit','HMD_AMP','AmPEP_raw_source','ampir_raw_mature90']
SL = ['AMPlify','tsAMP-I','c_AMPs','HMD-AMP','AmPEP','ampir']
POLICIES = ['mixture_original','mixture_source_support','per_source_supported']
PL = ['Pooled','Pooled + support','Per source + support']
BLUE, GREY, ORANGE = '#267E93', '#7B828A', '#B66C36'

def read(rel):
    with (ROOT/'inputs'/rel).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))

def write(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)

def num(r,k):
    try: return float(r[k])
    except (KeyError, ValueError, TypeError): return float('nan')

def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Arial','DejaVu Sans'], 'font.size':7,
        'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':7,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.frameon':False,
        'savefig.facecolor':'white','figure.facecolor':'white'})
    return plt

def audit(fig, name, out):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    qa=out/'QA';qa.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(fig,json_out=qa/(name+'.alignment.json'),strict=True)
    for folder in ['PDF','SVG','PNG','TIFF']:(out/folder).mkdir(exist_ok=True)
    fig.savefig(out/'PDF'/(name+'.pdf'))
    fig.savefig(out/'SVG'/(name+'.svg'))
    fig.savefig(out/'PNG'/(name+'.png'),dpi=600)
    fig.savefig(out/'TIFF'/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    for script, args in [
        ('audit_pdf_text.py',['--min-pt','5','--json']),
        ('audit_figure_collisions.py',['--json-out',str(qa/(name+'.collisions.json'))])]:
        p=subprocess.run([sys.executable,str(ROOT/'qa_tools'/script),str(out/'PDF'/(name+'.pdf'))]+args,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        (qa/(name+'.'+script+'.log')).write_text(p.stdout,encoding='utf-8')
        if p.returncode:raise RuntimeError('Figure QA failed: '+name+'; see '+str(qa))
    print('FIGURE PASS',name,flush=True)

def grid(plt, rows=2, cols=2, height=7.0):
    fig,axs=plt.subplots(rows,cols,figsize=(183/25.4,height),squeeze=False)
    fig.subplots_adjust(left=.20,right=.975,bottom=.075,top=.94,wspace=.94,hspace=.49)
    return fig,axs

def heading(ax, letter, title):
    ax.set_title(letter+'  '+title,loc='left',pad=17)

def pairs(ev):
    lookup={(r['held_source'],r['model'],r['repeat'],r['policy'],r['phase']):r for r in ev}
    items=[]
    for key,b in lookup.items():
        h,m,rep,p,phase=key
        if p!='per_source_supported' or phase!='held_source' or b['policy_status']!='feasible':continue
        a=lookup[h,m,rep,'mixture_source_support',phase]
        assert a['policy_status']=='feasible'
        row={'held_source':h,'model':m,'repeat':rep}
        for metric in ['conditional_error_balanced','coverage_balanced','positive_discarded_fraction_balanced']:
            row[metric+'_a']=num(a,metric);row[metric+'_b']=num(b,metric)
            row[metric+'_difference_pp']=100*(num(b,metric)-num(a,metric))
        items.append(row)
    assert len(items)==201
    return items

def delta_panel(ax, data, metric, xlabel, limits):
    groups=[(m,h) for m in ['AMPlify','tsAMP_I'] for h in SOURCES]
    for y,(m,h) in enumerate(groups):
        rs=sorted([r for r in data if r['model']==m and r['held_source']==h],key=lambda r:int(r['repeat']))
        vals=[r[metric+'_difference_pp'] for r in rs]
        offsets=np.linspace(-.18,.18,len(rs))
        ax.scatter(vals,y+offsets,s=9,c=BLUE if m=='AMPlify' else ORANGE,alpha=.45,edgecolors='none')
        if vals:ax.plot(float(np.median(vals)),y,marker='|',ms=9,mew=1.25,color='#252B32')
    labels=[LABEL[m]+' / '+SL[SOURCES.index(h)] for m,h in groups]
    ax.set_yticks(range(12),labels);ax.set_ylim(11.7,-.7);ax.set_xlim(*limits)
    ax.tick_params(axis='y',length=0,pad=5);ax.spines['left'].set_visible(False)
    ax.axvline(0,color='#C5CCD2',lw=.65,zorder=0);ax.set_xlabel(xlabel)

def render(out, rules, ev):
    plt=style();out.mkdir(parents=True,exist_ok=False)
    (out/'source_data').mkdir()
    paired=pairs(ev);write(out/'source_data/Fig05_paired_201.tsv',paired)
    metric_limits={}
    for metric in ['conditional_error_balanced','coverage_balanced','positive_discarded_fraction_balanced']:
        values=[r[metric+'_difference_pp'] for r in paired]
        metric_limits[metric]=(min(-1,5*math.floor(min(values)/5)-1),max(1,5*math.ceil(max(values)/5)+1))
    availability=[]
    matrix=np.zeros((5,3),dtype=int)
    for i,m in enumerate(MODELS[:5]):
        for j,p in enumerate(POLICIES):
            n=sum(r['model']==m and r['policy']==p and r['policy_status']=='feasible' for r in rules)
            matrix[i,j]=n
            availability.append({'model':m,'policy':p,'feasible':n,'planned':120})
    write(out/'source_data/Fig05_availability.tsv',availability)
    fig,axs=grid(plt,height=7.25)
    ax=axs[0,0]
    ax.imshow(matrix,aspect='auto',vmin=0,vmax=120,cmap='Blues',interpolation='nearest')
    for i in range(5):
        for j in range(3):
            ax.text(j,i,f'{matrix[i,j]}/120',ha='center',va='center',fontsize=7,
                    color='white' if matrix[i,j]>65 else '#252B32')
    ax.set_yticks(range(5),[LABEL[m] for m in MODELS[:5]])
    ax.set_xticks(range(3),['Pooled','Support\nmatched','Per\nsource'])
    ax.tick_params(length=0,pad=5)
    for spine in ax.spines.values():spine.set_visible(False)
    heading(ax,'a','Rule feasibility')
    for letter,ax,metric,xlabel,lim,title in [
        ('b',axs[0,1],'conditional_error_balanced','Paired change (pp)',metric_limits['conditional_error_balanced'],'Held-source error'),
        ('c',axs[1,0],'coverage_balanced','Paired change (pp)',metric_limits['coverage_balanced'],'Held-source coverage'),
        ('d',axs[1,1],'positive_discarded_fraction_balanced','Paired change (pp)',metric_limits['positive_discarded_fraction_balanced'],'Source positives discarded')]:
        delta_panel(ax,paired,metric,xlabel,lim);heading(ax,letter,title)
    # Verify limits include every pair; plot limits must not silently hide observations.
    for metric,lim in metric_limits.items():
        vals=[r[metric+'_difference_pp'] for r in paired]
        assert min(vals)>=lim[0] and max(vals)<=lim[1], (metric,min(vals),max(vals))
    audit(fig,'Fig05_honest_source_constraints',out);plt.close(fig)
