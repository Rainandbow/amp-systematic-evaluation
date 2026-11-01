"""Ten revised supplementary figures, using only frozen data and intervals."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,hashlib
import supplement_helpers as H
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

ROOT=H.ROOT
D,DS,MS,C=H.D,H.DS,H.MS,H.COLORS
name,one,num,style,heading,save=H.name,H.one,H.num,H.style,H.heading,H.save
SETS=['A_declared_all','B_bacterial_assay_clue','C_bacterial_quant_clue']
PERF=H.read_tsv(ROOT/'source_data/historical_metrics.tsv')
EXT=H.read_tsv(ROOT/'source_data/positive_recall_metrics.tsv')
JOIN=H.read_tsv(ROOT/'source_data/evidence_prediction_joined.tsv')
STEMS=[]
def export(fig,stem,axes,**kw):
    save(fig,stem,axes,**kw);STEMS.append(stem)
def dot(ax,x,y,i,lo=None,hi=None,marker=None,filled=None):
    if lo is not None:ax.hlines(y,lo,hi,color=C[i],lw=1)
    if filled is None:filled=lo is not None
    ax.plot(x,y,marker=marker or H.MARKERS[i],ms=3.5,mec=C[i],mfc=C[i] if filled else 'white',mew=.8)
def select(scope='full',rule='paper_rule',tool=None):
    return one(EXT,rule=rule,scope=scope,study='ALL',tool=tool)

def s1():
    fig,axs=plt.subplots(3,2,figsize=(183/25.4,185/25.4))
    cmap=LinearSegmentedColormap.from_list('source_fpr',['#FBFCFD','#DFB79C','#A93836'])
    for ax,ds,letter in zip(axs.flat,DS,'abcdef'):
        bins=[b for b in ['10_29','30_59','60_99','100_126'] if int(one(D['length'],dataset=ds,tool=MS[0],length_bin=b)['negative'])]
        values=np.array([[float(one(D['length'],dataset=ds,tool=m,length_bin=b)['fpr']) for b in bins]for m in MS])
        im=ax.imshow(values,cmap=cmap,vmin=0,vmax=1,aspect='auto')
        for i,m in enumerate(MS):
            for j,b in enumerate(bins):
                r=one(D['length'],dataset=ds,tool=m,length_bin=b);v=values[i,j]
                ax.text(j,i,f'{v*100:.1f}%'+('*' if int(r['negative'])<20 else ''),ha='center',va='center',fontsize=6.3,color='white'if v>.7 else '#203B4E')
        ax.set_yticks(range(6),[name(m)for m in MS]);ax.set_xticks(range(len(bins)),[b.replace('_','–')+'\nN='+one(D['length'],dataset=ds,tool=MS[0],length_bin=b)['negative']for b in bins],fontsize=6.3)
        ax.tick_params(length=0,pad=3)
        for spine in ax.spines.values():spine.set_visible(False)
        heading(ax,letter,name(ds))
    fig.subplots_adjust(left=.12,right=.90,top=.91,bottom=.12,hspace=.68,wspace=.40)
    cax=fig.add_axes([.94,.25,.013,.48]);cb=fig.colorbar(im,cax=cax,ticks=[0,.5,1]);cb.ax.set_yticklabels(['0%','50%','100%']);cb.ax.tick_params(labelsize=6);cb.ax.set_title('FPR',fontsize=6.5,pad=9);cb.outline.set_visible(False)
    export(fig,'FigS01_length_negative_calls',list(axs.flat))

def s2():
    fig,axs=plt.subplots(3,2,figsize=(183/25.4,185/25.4))
    for ax,ds,letter in zip(axs.flat,DS,'abcdef'):
        upper=[]
        for i,m in enumerate(MS[:5]):
            cv=one(D['reliability'],dataset=ds,tool=m,metric='high_confidence_coverage_095')
            er=one(D['reliability'],dataset=ds,tool=m,metric='high_confidence_error_095')
            x,y=float(cv['estimate'])*100,float(er['estimate'])*100
            upper.append(max(y, float(er['ci_high'])*100 if num(er['ci_high']) is not None else y))
            if num(cv['ci_low'])is not None:ax.hlines(y,float(cv['ci_low'])*100,float(cv['ci_high'])*100,color=C[i],lw=.9)
            if num(er['ci_low'])is not None:ax.vlines(x,float(er['ci_low'])*100,float(er['ci_high'])*100,color=C[i],lw=.9)
            ax.plot(x,y,marker=H.MARKERS[i],ms=3.4,color=C[i])
        limit=max(10,10*np.ceil(max(upper)/10));ax.set_xlim(-3,103);ax.set_ylim(-.03*limit,1.03*limit);ax.set_xticks([0,50,100]);ax.set_yticks(np.linspace(0,limit,3))
        ax.set_xlabel('Coverage (%)');ax.set_ylabel('Selected error (%)',fontsize=6.4);style(ax,'both');heading(ax,letter,name(ds))
    H.legend(fig,n=5)
    fig.subplots_adjust(left=.11,right=.98,top=.85,bottom=.11,hspace=.70,wspace=.53)
    export(fig,'FigS02_extreme_score_tradeoff',list(axs.flat))

def s3():
    fig=plt.figure(figsize=(180/25.4,112/25.4))
    a=fig.add_axes([.12,.19,.43,.62]);b=fig.add_axes([.62,.19,.35,.62])
    codes=[['Y','Y','Y','S']for _ in range(5)]+[['Y','E','R','T']]
    colors={'Y':'#DAEAF2','S':'#F3E6C7','E':'#F3E6C7','R':'#ECEFF2','T':'#ECEFF2'}
    for i,row in enumerate(codes):
        for j,code in enumerate(row):
            a.add_patch(Rectangle((j-.5,i-.5),1,1,fc=colors[code],ec='white',lw=1.4))
            a.text(j,i,code,ha='center',va='center',fontsize=8,color='#253F50',fontweight='bold')
    a.set_xlim(-.5,3.5);a.set_ylim(5.5,-.5);a.set_yticks(range(6),[name(m)for m in MS])
    a.set_xticks(range(4),['Decision\nmetrics','AP /\nAUROC','Score-based\nselection','Inventory\nscreen'],fontsize=6.2);a.tick_params(length=0)
    for sp in a.spines.values():sp.set_visible(False)
    heading(a,'a','Analysis-specific availability')
    b.axis('off');heading(b,'b','Recovered c_AMPs evidence')
    lines=[('Available',True),('Published PR coordinates',False),('Internal training IDs',False),('Native strict-AND decisions',False),
           ('Unresolved',True),('Complete ID–sequence mapping',False),('Native ensemble probability rule',False),
           ('Analysis treatment',True),('AP / AUROC: min-score extension',False),('Score retention: not substituted',False)]
    for i,(text,bold)in enumerate(lines):b.text(0,.98-i*.098,text,transform=b.transAxes,va='top',fontsize=6.3,fontweight='bold'if bold else 'normal')
    export(fig,'FigS03_analysis_availability',[a,b],rows=[['a','b']],ids=['a','b'])

def ranking(metric,index):
    fig,axs=plt.subplots(3,2,figsize=(183/25.4,185/25.4))
    for ax,ds,letter in zip(axs.flat,DS,'abcdef'):
        for i,m in enumerate(MS):
            r=one(PERF,dataset=ds,tool=m,metric=metric,scope='global_primary')
            if m=='c_AMPs':assert r['score_rule']=='researcher_min_component_extension'
            dot(ax,float(r['estimate']),i,i,num(r['ci_low']),num(r['ci_high']))
        ax.set_yticks(range(6),[name(m)+(' (extension)'if m=='c_AMPs'else '')for m in MS],fontsize=6.3)
        ax.set_ylim(5.6,-.6);ax.set_xlim(-.03,1.04);ax.set_xticks([0,.5,1]);ax.set_xlabel('AP'if metric=='ap'else'AUROC');style(ax);heading(ax,letter,name(ds))
    fig.subplots_adjust(left=.16,right=.98,top=.91,bottom=.11,hspace=.64,wspace=.55)
    export(fig,f'FigS0{index}_'+('average_precision'if metric=='ap'else'auroc'),list(axs.flat))

def s6():
    composition=H.read_tsv(ROOT/'source_data/evidence_composition.tsv');recall=H.read_tsv(ROOT/'source_data/evidence_recall_ci.tsv')
    fig=plt.figure(figsize=(180/25.4,170/25.4))
    a=fig.add_axes([.16,.67,.33,.19]);b=fig.add_axes([.62,.67,.33,.19]);c=fig.add_axes([.16,.11,.79,.37])
    for ax,dim,cats,cols in [(a,'database',['APD','dbAMP','DRAMP'],['#0072B2','#67B9A7','#B6C3D0']),
                           (b,'length',['10_29','30_59','60_99','100_126'],['#326D92','#6FA8B5','#B3D1CA','#DFE6E7'])]:
        for j,s in enumerate(SETS):
            left=0
            for cat,col in zip(cats,cols):
                r=one(composition,cohort=s,dimension=dim,category=cat);v=float(r['fraction'])*100
                ax.barh(j,v,left=left,height=.5,color=col,edgecolor='none')
                if v>=10:ax.text(left+v/2,j,f'{v:.1f}%',ha='center',va='center',fontsize=6.3,color='white'if cat in ['APD','10_29']else'#254252')
                left+=v
            assert abs(left-100)<1e-8
        ax.set_yticks(range(3),['A (3,176)','B (1,386)','C (1,081)'],fontsize=6.3);ax.set_ylim(2.6,-.6);ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.set_xlabel('Share of records (%)',fontsize=6.4);style(ax,grid=None)
        ax.legend([Patch(fc=col)for col in cols],[cat.replace('_','–')for cat in cats],loc='lower left',bbox_to_anchor=(-.03,1.015),ncol=2,fontsize=6.2,columnspacing=1,handlelength=1)
    # Separate the section titles from the two-line legends by fixed points.
    for ax,letter,title in [(a,'a','Database composition'),(b,'b','Length composition')]:
        ax.text(0,1,letter,transform=ax.transAxes+H.ScaledTranslation(0,28/72,fig.dpi_scale_trans),fontsize=9,fontweight='bold',va='bottom')
        ax.text(.11,1,title,transform=ax.transAxes+H.ScaledTranslation(0,28/72,fig.dpi_scale_trans),fontsize=7.5,fontweight='bold',va='bottom')
    for i,m in enumerate(MS):
        for j,s in enumerate(SETS):
            r=one(recall,cohort=s,tool=m)
            dot(c,float(r['estimate'])*100,i+(j-1)*.22,i,num(r['ci_low'])*100 if num(r['ci_low'])is not None else None,num(r['ci_high'])*100 if num(r['ci_high'])is not None else None,marker=['o','s','^'][j])
    c.set_yticks(range(6),[name(m)for m in MS]);c.set_ylim(5.7,-.7);c.set_xlim(0,103);c.set_xticks([0,25,50,75,100]);c.set_xlabel('Positive-call rate (%)');style(c);heading(c,'c','Positive calls across evidence strata')
    c.legend([Line2D([],[],marker=mk,color='#506779',lw=0,ms=3)for mk in ['o','s','^']],['Stratum A','Stratum B','Stratum C'],loc='lower right',bbox_to_anchor=(1,1.02),ncol=3,fontsize=6)
    export(fig,'FigS06_evidence_strata',[a,b,c],ids=['a','b','c'],rows=[['a','b']])

def s7():
    rows=H.read_tsv(ROOT/'source_data/matched_PU.tsv')
    fig,axs=plt.subplots(2,3,figsize=(180/25.4,155/25.4))
    for j,s in enumerate(SETS):
        a,b=axs[0,j],axs[1,j]
        for i,m in enumerate(MS):
            r=one(rows,cohort=s,length_bin='all',tool=m)
            for key,offset,mk,filled in [('P_recall',-.14,'o',True),('U_positive_call_rate',.14,'s',False)]:
                dot(a,float(r[key])*100,i+offset,i,num(r[key+'_ci_low'])*100 if num(r[key+'_ci_low'])is not None else None,num(r[key+'_ci_high'])*100 if num(r[key+'_ci_high'])is not None else None,marker=mk,filled=filled)
            dot(b,float(r['P_minus_U'])*100,i,i,num(r['P_minus_U_ci_low'])*100 if num(r['P_minus_U_ci_low'])is not None else None,num(r['P_minus_U_ci_high'])*100 if num(r['P_minus_U_ci_high'])is not None else None)
        n=one(rows,cohort=s,length_bin='all',tool=MS[0])['pair_n']
        for ax in [a,b]:ax.set_yticks(range(6),[name(m)for m in MS]);ax.set_ylim(5.6,-.6);style(ax)
        a.set_xlim(-3,103);a.set_xticks([0,50,100]);a.set_xlabel('Positive-call rate (%)',fontsize=6.3)
        b.set_xlim(-25,103);b.set_xticks([-25,0,50,100]);b.axvline(0,color='#99A7B2',lw=.6);b.set_xlabel('P − U (percentage points)',fontsize=6.2)
        heading(a,'abc'[j],f'Stratum {"ABC"[j]}: {n} pairs');heading(b,'def'[j],f'Stratum {"ABC"[j]}: paired contrast')
    fig.legend([Line2D([],[],marker='o',color='#506779',lw=0,ms=3),Line2D([],[],marker='s',mfc='white',mec='#506779',lw=0,ms=3)],['Declared-positive P','Unannotated U'],loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,fontsize=6.5)
    fig.subplots_adjust(left=.11,right=.98,top=.86,bottom=.10,hspace=.65,wspace=.57)
    export(fig,'FigS07_matched_PU',list(axs.flat))

def s8():
    fig=plt.figure(figsize=(180/25.4,150/25.4));a=fig.add_axes([.12,.13,.31,.73]);b=fig.add_axes([.70,.13,.27,.73])
    for i,m in enumerate(MS):
        r=select(scope='common_six_support',tool=m)
        dot(a,float(r['recall'])*100,i,i,num(r['ci95_low'])*100 if num(r['ci95_low'])is not None else None,num(r['ci95_high'])*100 if num(r['ci95_high'])is not None else None)
        a.text(114,i,f'{r["predicted_positive_n"]}/76',fontsize=6.2,ha='center',va='center')
    a.set_yticks(range(6),[name(m)for m in MS]);a.set_ylim(5.6,-.6);a.set_xlim(-3,133);a.set_xticks([0,50,100]);a.set_xlabel('Recall (%)');style(a);heading(a,'a','Same 76 supported positives')
    pairs=[r for r in H.read_tsv(ROOT/'source_data/paired_recall_differences.tsv')if(r['rule'],r['scope'],r['study'])==('paper_rule','common_six_support','ALL')]
    assert len(pairs)==15
    for i,r in enumerate(pairs):
        if num(r['ci95_low'])is not None:b.hlines(i,float(r['ci95_low'])*100,float(r['ci95_high'])*100,color='#506E85',lw=1)
        b.plot(float(r['recall_difference'])*100,i,'o',ms=3,mfc='#506E85'if num(r['ci95_low'])is not None else'white',mec='#506E85')
    b.set_yticks(range(15),[name(r['tool_a'])+' − '+name(r['tool_b'])for r in pairs],fontsize=6.3);b.set_ylim(14.7,-.7);b.set_xlim(-65,65);b.set_xticks([-60,0,60]);b.axvline(0,color='#99A7B2',lw=.6);b.set_xlabel('Recall difference (pp)',fontsize=6.3);style(b);heading(b,'b','Paired model contrasts')
    export(fig,'FigS08_common_members',[a,b],ids=['a','b'],rows=[['a','b']])

def compare(ax,scope0,scope1,rule1='paper_rule'):
    for i,m in enumerate(MS):
        for j,(scope,rule)in enumerate([(scope0,'paper_rule'),(scope1,rule1)]):
            r=select(scope=scope,rule=rule,tool=m)
            if num(r['recall'])is None:continue
            dot(ax,float(r['recall'])*100,i+(j-.5)*.25,i,num(r['ci95_low'])*100 if num(r['ci95_low'])is not None else None,num(r['ci95_high'])*100 if num(r['ci95_high'])is not None else None,marker=['o','s'][j],filled=(j==0))
    ax.set_yticks(range(6),[name(m)for m in MS]);ax.set_ylim(5.6,-.6);ax.set_xlim(-3,103);ax.set_xticks([0,50,100]);ax.set_xlabel('Recall (%)');style(ax)

def s9():
    fig=plt.figure(figsize=(180/25.4,163/25.4))
    a=fig.add_axes([.18,.59,.31,.25]);b=fig.add_axes([.68,.59,.29,.25]);c=fig.add_axes([.18,.10,.31,.27]);d=fig.add_axes([.68,.10,.29,.27])
    idx={(r['record_id'],r['tool']):r for r in JOIN};ids={r['record_id']for r in JOIN}
    p5={rid for rid in ids if all(idx[rid,m]['screen_status']=='not_detected_in_available_inventory'for m in MS[:5])}
    sup={rid for rid in p5 if all(idx[rid,m]['supported']=='1'for m in MS)}
    pos={rid for rid in sup if idx[rid,MS[0]]['paper_rule_positive']=='1'}
    sizes=[len(ids),len(p5),len(sup),len(pos)];assert sizes==[156,45,24,18]
    a.barh(range(4),sizes,height=.53,color=['#A6BCCB','#6D9EB7','#4385A8','#245C83'],edgecolor='none')
    for i,n in enumerate(sizes):a.text(n+4,i,str(n),va='center',fontsize=6.5)
    a.set_yticks(range(4),['Tested','Five-inventory pass','Six-model support','Activity-positive'],fontsize=6);a.set_ylim(3.6,-.6);a.set_xlim(0,180);a.set_xticks([0,75,150]);a.set_xlabel('Sequences');style(a,grid=None);heading(a,'a','Sequential membership')
    compare(b,'common_six_support','common_five_inventory_pass')
    b.annotate('18/18',xy=(100,2+.125),xytext=(-9,9),textcoords='offset points',ha='right',va='bottom',fontsize=6.5,color=C[2])
    compare(d,'common_six_support','common_six_support',rule1='unified64')
    for ax,letter,title,labels in [(b,'b','Common-support sensitivity',['76 shared positives','18 inventory-pass positives']),
                                        (d,'d','MIC-rule sensitivity',['Original rules: 76','Unified ≤64 μM: 75'])]:
        trans=ax.transAxes+H.ScaledTranslation(0,29/72,fig.dpi_scale_trans)
        ax.text(0,1,letter,transform=trans,fontsize=9,fontweight='bold',va='bottom');ax.text(.11,1,title,transform=trans,fontsize=7,fontweight='bold',va='bottom')
        ax.legend([Line2D([],[],marker='o',color='#506779',lw=0,ms=3),Line2D([],[],marker='s',mfc='white',mec='#506779',lw=0,ms=3)],labels,loc='lower left',bbox_to_anchor=(-.04,1.01),ncol=1,fontsize=6.3,handletextpad=.3)
    for i,m in enumerate(MS[:5]):
        r=select(scope='per_tool_inventory_pass',tool=m)
        dot(c,float(r['recall'])*100,i,i,num(r['ci95_low'])*100 if num(r['ci95_low'])is not None else None,num(r['ci95_high'])*100 if num(r['ci95_high'])is not None else None)
        c.text(114,i,f'{r["predicted_positive_n"]}/{r["supported_positive_n"]}',ha='center',va='center',fontsize=6.3)
    c.set_yticks(range(5),[name(m)for m in MS[:5]]);c.set_ylim(4.6,-.6);c.set_xlim(-3,135);c.set_xticks([0,50,100]);c.set_xlabel('Recall (%)');style(c);heading(c,'c','Model-specific screened sets')
    for ax in [a,c]:
        for text in ax.texts:
            if text.get_text()in ['a','c','Sequential membership','Model-specific screened sets']:
                text.set_transform(ax.transAxes+H.ScaledTranslation(0,29/72,fig.dpi_scale_trans))
    export(fig,'FigS09_screening_activity_sensitivity',[a,b,c,d],ids=['a','b','c','d'],rows=[['a','b'],['c','d']],cols=[['a','c']])

def s10():
    rows=defaultdict(dict)
    for r in JOIN:rows[r['record_id']][r['tool']]=r
    selected={rid:rr for rid,rr in rows.items()if rr[MS[0]]['paper_rule_positive']=='1'and rr[MS[0]]['common_six_support']=='1'}
    assert len(selected)==76 and all(len(rr)==6 for rr in selected.values())
    patterns=Counter(tuple(int(rr[m]['prediction'])for m in MS)for rr in selected.values())
    ordered=sorted(patterns,key=lambda k:(-patterns[k],-sum(k),tuple(-v for v in k)))
    assert len(patterns)==19 and patterns[(1,1,1,1,1,1)]==18 and patterns.get((0,0,0,0,0,0),0)==0
    totals=[sum(k[i]*n for k,n in patterns.items())for i in range(6)];assert totals==[47,65,73,73,59,34]
    fig=plt.figure(figsize=(180/25.4,115/25.4))
    bar=fig.add_axes([.26,.55,.60,.28]);matrix=fig.add_axes([.26,.19,.60,.29]);sets=fig.add_axes([.07,.19,.14,.29])
    xx=np.arange(len(ordered));counts=[patterns[k]for k in ordered];cols=['#00875A'if sum(k)==6 else'#536F86'for k in ordered]
    bar.bar(xx,counts,width=.65,color=cols,edgecolor='none')
    for x,n in zip(xx,counts):bar.text(x,n+.45,str(n),ha='center',va='bottom',fontsize=6)
    bar.set_xlim(-.7,len(ordered)-.3);bar.set_ylim(0,max(counts)+4);bar.set_ylabel('Exact pattern count',fontsize=6.5);bar.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=4));bar.set_xticks([]);style(bar,'y');bar.spines['bottom'].set_visible(False);heading(bar,'a','Exact six-model positive-call patterns')
    for i in range(6):
        if i%2==0:matrix.axhspan(i-.5,i+.5,color='#F1F4F6');sets.axhspan(i-.5,i+.5,color='#F1F4F6')
        matrix.scatter(xx,np.full(len(xx),i),s=6,color='#D4DDE3',edgecolors='none')
    for j,k in enumerate(ordered):
        yy=np.flatnonzero(k);matrix.plot([j,j],[min(yy),max(yy)],lw=.9,color=cols[j]);matrix.scatter(np.full(len(yy),j),yy,s=13,color=cols[j],edgecolors='none')
    matrix.set_xlim(bar.get_xlim());matrix.set_ylim(5.6,-.6);matrix.set_xticks([]);matrix.set_yticks(range(6),[name(m)for m in MS]);matrix.yaxis.tick_right();matrix.tick_params(length=0,pad=5);matrix.set_xlabel('Exact decision membership patterns',fontsize=6.5,labelpad=8)
    for sp in matrix.spines.values():sp.set_visible(False)
    sets.barh(range(6),totals,height=.5,color=C,edgecolor='none')
    for i,n in enumerate(totals):sets.text(n+2,i,str(n),ha='right',va='center',fontsize=6)
    sets.set_xlim(85,0);sets.set_ylim(matrix.get_ylim());sets.set_yticks([]);sets.set_xticks([0,40,80]);sets.set_xlabel('Detected / 76',fontsize=6.2);style(sets,grid=None);sets.spines['left'].set_visible(False)
    fig.text(.07,.78,'76 positives',fontsize=7.5,fontweight='bold');fig.text(.07,.70,'Six fixed\nmodels',fontsize=6.5,linespacing=1.5)
    export(fig,'FigS10_positive_decision_patterns',[bar,matrix,sets],ids=['count','membership','marginal'],rows=[['membership','marginal']],cols=[['count','membership']])
    with (H.OUT/'source_data/FigS10_decision_patterns.tsv').open('w',encoding='utf-8',newline='')as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['exact_count',*MS]);[w.writerow([patterns[k],*k])for k in ordered]
