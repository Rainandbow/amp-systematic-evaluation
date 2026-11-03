"""Python-only plotting of the frozen linked reliability results; no new statistics."""
from pathlib import Path
import sys,json,math,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import legacy_common as C

ROOT=Path(__file__).resolve().parents[1]
SKILL=ROOT/'tools'
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
DISPLAY={'Macrel':'Macrel','AMPlify':'AMPlify','ampir':'ampir','tsAMP_I':'tsAMP-I','HMD_AMP':'HMD-AMP'}
SOURCE={'AMPlify':'AMPlify','tsAMP_I':'tsAMP-I','c_AMPs_benchmark_resplit':'c_AMPs','HMD_AMP':'HMD-AMP','AmPEP_raw_source':'AmPEP','ampir_raw_mature90':'ampir'}
COLORS={'Decision_confidence':'#267d96','Inventory_similarity':'#479b83','Submodel_SD':'#bd7c46','Entropy_gap':'#9275a8','Random_expected':'#8f969b'}
MDIS={'Decision_confidence':'Decision confidence','Inventory_similarity':'Reference distance','Submodel_SD':'Submodel SD','Entropy_gap':'Entropy gap','Random_expected':'Random expectation'}

def f(r,k):return float(r[k])

def save(fig,axes,name):
    path=ROOT/'figures'/name
    require_matplotlib_panel_alignment(fig,axes=axes,json_out=str(path)+'.alignment.json',overlay_svg=str(ROOT/'qa'/f'{name}.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(path)+'.pdf')
    fig.savefig(str(path)+'.svg')
    fig.savefig(str(path)+'.png',dpi=600)
    fig.savefig(str(path)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    for script,args in [('audit_pdf_text.py',['--min-pt','5']),('audit_figure_collisions.py',['--json-out',str(ROOT/'qa'/f'{name}.collision.json')])]:
        cmd=[sys.executable,str(SKILL/script),str(path)+'.pdf']+args
        p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf8')
        (ROOT/'qa'/f'{name}.{script}.log').write_text(p.stdout+p.stderr,encoding='utf8')
        if p.returncode:raise RuntimeError(f'{name}: {script} failed, see QA log')
    print('FIGURE PASS',name,flush=True)

def diagnostic():
    rows=C.read(ROOT/'results/diagnostic_cells.tsv')
    fig,axs=plt.subplots(2,5,figsize=(7.2,4.6))
    fig.subplots_adjust(left=.135,right=.99,bottom=.13,top=.89,wspace=.22,hspace=.65)
    for row,role in enumerate(['inventory','selection_development']):
        for col,model in enumerate(C.MODELS[:-1]):
            ax=axs[row,col];arr=np.full((6,3),np.nan);nn=np.zeros((6,3),int)
            for s,src in enumerate(C.SOURCES):
                rr=[r for r in rows if r['held_source']==src and r['model']==model and r['reference_role']==role and r['confidence_tier']=='2']
                assert len(rr)==3
                for r in rr:
                    d=int(r['distance_tier']);arr[s,d]=f(r,'error_rate');nn[s,d]=int(r['n'])
            im=ax.imshow(np.ma.masked_invalid(arr),vmin=0,vmax=1,cmap='Blues',aspect='auto',interpolation='nearest')
            for i in range(6):
                for j in range(3):
                    if np.isfinite(arr[i,j]):
                        ax.text(j,i,f'{100*arr[i,j]:.1f}\n{nn[i,j]:,}',ha='center',va='center',fontsize=5.4,color='white' if arr[i,j]>.58 else '#203344',linespacing=1.1)
                    else:ax.text(j,i,'—',ha='center',va='center',fontsize=6,color='#777')
            ax.set_xticks([0,1,2],['T1','T2','T3']);ax.tick_params(length=0,pad=3)
            ax.set_yticks(range(6),[SOURCE[s] for s in C.SOURCES] if col==0 else ['']*6)
            ax.set_title(f'{chr(97+row*5+col)}  {DISPLAY[model]}',loc='left',fontsize=7,pad=7)
            for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.018,.68,'Available inventory',rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=7)
    fig.text(.018,.285,'Selection-only reference',rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=7)
    fig.text(.135,.967,'High-confidence decisions (confidence ≥0.95)',fontsize=8,weight='bold')
    fig.text(.135,.055,'Cells: error (%) / record count. T1–T3: selection-defined distance tiers. Dash: empty cell.',fontsize=6)
    save(fig,list(axs.flat),'A_high_confidence_error_map')

def paired():
    rr=[r for r in C.read(ROOT/'results/historical_paired_intervals_80.tsv') if r['method_a']=='Logistic_C_L_I' and r['method_b']=='Logistic_C_L']
    fig,axs=plt.subplots(1,3,figsize=(7.2,8.0));fig.subplots_adjust(left=.235,right=.985,bottom=.095,top=.925,wspace=.28)
    labels=[f'{DISPLAY[m]} / {SOURCE[s]}' for m in C.MODELS[:-1] for s in C.SOURCES]
    for j,(metric,title) in enumerate([('conditional_error','Conditional error'),('positive_rejection','Positive rejection'),('end_to_end_detection','End-to-end detection')]):
        ax=axs[j];ext=[]
        for i,(model,src) in enumerate((m,s) for m in C.MODELS[:-1] for s in C.SOURCES):
            row=[r for r in rr if r['model']==model and r['held_source']==src and r['metric']==metric];assert len(row)==1;r=row[0]
            v=100*f(r,'estimate');lo=100*f(r,'ci_low');hi=100*f(r,'ci_high');y=29-i
            if np.isfinite(lo) and np.isfinite(hi):ax.plot([lo,hi],[y,y],color='#4c93a5',lw=.8);ext.extend([lo,hi]);face='#267d96'
            else:face='white'
            ax.plot(v,y,'o',ms=3,mec='#267d96',mfc=face,mew=.8,zorder=3);ext.append(v)
        ax.axvline(0,color='#b8c2c8',lw=.6,ls='--',zorder=0)
        for y in [5.5,11.5,17.5,23.5]:ax.axhline(y,color='#e7eaed',lw=.5,zorder=0)
        ax.set_ylim(-.8,29.8);ax.set_yticks(range(29,-1,-1),labels if j==0 else ['']*30);ax.tick_params(axis='y',length=0,pad=4)
        lo=min(ext+[0]);hi=max(ext+[0]);span=max(hi-lo,2);ax.set_xlim(lo-.07*span,hi+.07*span)
        ax.set_xlabel('Paired change (pp)');ax.set_title(f'{chr(97+j)}  {title}',loc='left',fontsize=7,pad=10)
        ax.tick_params(axis='x',labelsize=6)
    fig.text(.235,.976,'Added reference-distance information at 80% retention',fontsize=8,weight='bold')
    fig.text(.235,.045,'C + L + I minus C + L; primary development partition. Lines: conditional 95% block-bootstrap intervals.',fontsize=5.8)
    fig.text(.235,.022,'C, decision confidence; L, length; I, available-inventory distance. Open points: interval omitted.',fontsize=5.8)
    save(fig,list(axs),'B_paired_added_distance_value')

def experimental():
    cohort='ApexGO_template_based';task='any_eligible_measured_strain'
    rows=[r for r in C.read(ROOT/'results/experimental_retention_points.tsv') if r['cohort']==cohort and r['task']==task]
    ci=[r for r in C.read(ROOT/'results/experimental_paired_intervals_80.tsv') if r['cohort']==cohort and r['task']==task and r['metric']=='conditional_error']
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.8));fig.subplots_adjust(left=.075,right=.98,bottom=.25,top=.71,wspace=.42)
    for j,(metric,title) in enumerate([('conditional_error','Conditional error'),('positive_rejection','Positive rejection')]):
        ax=axs[j]
        for method in ['Decision_confidence','Inventory_similarity','Submodel_SD','Entropy_gap','Random_expected']:
            rr=sorted([r for r in rows if r['method']==method],key=lambda r:f(r,'coverage_target'))
            ax.plot([100*f(r,'coverage_target') for r in rr],[100*f(r,metric) for r in rr],color=COLORS[method],marker='o' if method!='Random_expected' else None,markersize=2.8,lw=1.1 if method=='Decision_confidence' else .8,ls='--' if method=='Random_expected' else '-')
        ax.set_xticks([20,40,60,80,100]);ax.set_xlabel('Retention (%)');ax.set_ylabel('(%)');ax.set_title(f'{chr(97+j)}  {title}',loc='left',fontsize=7,pad=8)
        ax.set_ylim(bottom=0)
    ax=axs[2];ext=[]
    for y,method in zip([2,1,0],['Inventory_similarity','Submodel_SD','Entropy_gap']):
        r=[r for r in ci if r['method_a']==method][0];v=100*f(r,'estimate');lo=100*f(r,'ci_low');hi=100*f(r,'ci_high')
        ax.plot([lo,hi],[y,y],color=COLORS[method],lw=1);ax.plot(v,y,'o',color=COLORS[method],ms=4);ext.extend([lo,hi])
    ax.axvline(0,color='#bbc5ca',lw=.6,ls='--');ax.set_yticks([2,1,0],['Reference','SD','Entropy gap']);ax.tick_params(axis='y',length=0);ax.set_ylim(-.5,2.5)
    span=max(ext)-min(ext);ax.set_xlim(min(ext)-.12*span,max(ext)+.12*span);ax.set_xlabel('Change vs confidence (pp)');ax.set_title('c  Error change at 80%',loc='left',fontsize=7,pad=8)
    handles=[Line2D([0],[0],color=COLORS[m],lw=1,ls='--' if m=='Random_expected' else '-',label=MDIS[m]) for m in COLORS]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,.95),fontsize=6,columnspacing=1.2,handlelength=1.8)
    fig.text(.075,.06,'AMPlify; ApexGO template-based cohort: 100 candidates, 86 activity positives, 10 dependence blocks.',fontsize=6)
    save(fig,list(axs),'C_AMPlify_experimental_selection_utility')

if __name__=='__main__':
    diagnostic();paired();experimental()
