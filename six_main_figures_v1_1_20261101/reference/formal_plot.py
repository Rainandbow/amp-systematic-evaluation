"""Render formal saved estimates; no inference, refitting, or resampling."""
import argparse, csv, hashlib, json, math, os, subprocess, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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

def verify():
    if (ROOT/'package_manifest.json').exists():
        manifest=json.loads((ROOT/'package_manifest.json').read_text())
        for rel,sha in manifest['files'].items():
            assert (ROOT/rel).is_file() and hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha, 'Package mismatch: '+rel
    for rel, sha in json.loads((ROOT/'input_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha, 'Changed input: '+rel
    c=json.loads((ROOT/'inputs/COMPLETE.json').read_text())
    assert c['smoke_only'] is False and c['bootstrap']==10000 and c['repeats']==20
    pts=read('baseline_family/baseline_and_model_points.tsv')
    rules=read('honest_transfer/selected_rules.tsv')
    ev=read('honest_transfer/evaluation_points.tsv')
    assert (len(pts),len(rules),len(ev))==(96,1800,5400)
    assert all(r['smoke_only']=='False' for r in pts+rules+ev)
    assert Counter(r['policy'] for r in rules)==dict.fromkeys(POLICIES,600)
    counts=[sum(r['policy']==p and r['policy_status']=='feasible' for r in rules) for p in POLICIES]
    assert counts==[458,312,201], counts
    assert len({(r['cohort'],r['task'],r['method']) for r in pts})==96
    print('FORMAL INPUTS PASS; 96 points, 1800 rules, 5400 evaluations')
    return pts,rules,ev

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

def estimates(ax, points, intervals, cohort, task, metric, limit, xlabel):
    rows=[next(r for r in points if (r['cohort'],r['task'],r['method'])==(cohort,task,m)) for m in METHODS]
    ints={(r['method'],r['metric']):r for r in intervals if (r['cohort'],r['task'])==(cohort,task)}
    any_estimate=any(math.isfinite(num(r,metric)) for r in rows)
    for y,r in enumerate(rows):
        v=num(r,metric); ci=ints.get((r['method'],metric),{})
        lo,hi=num(ci,'ci_low'),num(ci,'ci_high')
        color=BLUE if y<6 else GREY
        if math.isfinite(v):
            hasci=math.isfinite(lo) and math.isfinite(hi)
            if hasci:ax.hlines(y,lo,hi,color=color,lw=.85,zorder=2)
            ax.scatter([v],[y],s=15,facecolor=color if hasci else 'white',edgecolor=color,lw=.9,zorder=3)
        elif any_estimate:
            ax.text(.52,y,'Not estimable',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=6)
    if not any_estimate:
        ax.text(.5,.5,'No paired-class families',transform=ax.transAxes,ha='center',va='center',fontsize=6)
    ax.set_yticks(range(12),[LABEL[m] for m in METHODS]);ax.set_ylim(11.65,-.65)
    ax.set_xlim(*limit);ax.set_xlabel(xlabel);ax.tick_params(axis='y',length=0,pad=5)
    ax.spines['left'].set_visible(False)
    if metric=='positive_hits':
        ax.axvline(num(rows[0],'random_expected_hits'),color=ORANGE,ls=(0,(3,2)),lw=.75,zorder=0)
    elif any_estimate:ax.axvline(.5,color='#D1D5D9',ls=(0,(3,2)),lw=.75,zorder=0)
    return rows

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

def render(out, pts, rules, ev):
    plt=style();out.mkdir(parents=True,exist_ok=False)
    (out/'source_data').mkdir()
    ints=read('baseline_family/baseline_and_model_intervals.tsv')
    fig,axs=grid(plt)
    configs=[(TASKS[0],'positive_hits',(0,21),'Tie-weighted positive hits at K = 20','Any measured strain'),
             (TASKS[1],'positive_hits',(0,21),'Tie-weighted positive hits at K = 20','Reference E. coli'),
             (TASKS[1],'pooled_AUROC',(-.03,1.03),'Pooled AUROC','Between all peptides'),
             (TASKS[1],'macro_within_family_AUROC',(-.03,1.03),'Mean within-family AUROC','Within comparable families')]
    for i,(ax,c) in enumerate(zip(axs.flat,configs)):
        task,metric,lim,xlabel,title=c
        estimates(ax,pts,ints,COHORTS[0],task,metric,lim,xlabel)
        heading(ax,chr(97+i),title)
    audit(fig,'Fig04_budget_baselines_and_family',out);plt.close(fig)
    write(out/'source_data/Fig04_points.tsv',[r for r in pts if r['cohort']==COHORTS[0]])
    write(out/'source_data/Fig04_intervals.tsv',[r for r in ints if r['cohort']==COHORTS[0]])
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
    for t,task in enumerate(TASKS):
        for metric,kind,lim,xlabel in [
            ('positive_hits','budget',(0,21),'Tie-weighted positive hits at fixed budget'),
            ('macro_within_family_AUROC','family',(-.03,1.03),'Mean within-family AUROC')]:
            fig,axs=grid(plt,height=7.25)
            for i,ax in enumerate(axs.flat):
                rows=estimates(ax,pts,ints,COHORTS[i],task,metric,lim,xlabel)
                if kind=='budget':
                    bound=math.ceil(num(rows[0],'budget_slots'))+.75
                    ax.set_xlim(-.035*bound,bound)
                title=SHORT_COHORT[i]
                heading(ax,chr(97+i),title)
            name=f'Sup_{kind}_{"any_strain" if t==0 else "reference_Ecoli"}'
            audit(fig,name,out);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(183/25.4,3.15))
    fig.subplots_adjust(left=.13,right=.975,bottom=.32,top=.81,wspace=.47)
    full=[]
    for j,(p,ax) in enumerate(zip(POLICIES,axs)):
        counts=np.zeros((5,6),dtype=int)
        for i,m in enumerate(MODELS[:5]):
            for k,h in enumerate(SOURCES):
                n=sum(r['held_source']==h and r['model']==m and r['policy']==p and r['policy_status']=='feasible' for r in rules)
                counts[i,k]=n
                full.append({'policy':p,'model':m,'held_source':h,'feasible':n,'planned':20})
        ax.imshow(counts,aspect='auto',vmin=0,vmax=20,cmap='Blues',interpolation='nearest')
        for i in range(5):
            for k in range(6):ax.text(k,i,str(counts[i,k]),ha='center',va='center',fontsize=6,
                color='white' if counts[i,k]>10 else '#252B32')
        ax.set_yticks(range(5),[LABEL[m] for m in MODELS[:5]])
        ax.set_xticks(range(6),SL,rotation=65,ha='right',rotation_mode='anchor')
        ax.tick_params(length=0,pad=4)
        for spine in ax.spines.values():spine.set_visible(False)
        heading(ax,chr(97+j),PL[j]);ax.set_xlabel('Held-out source')
    audit(fig,'Sup_honest_full_source_availability',out);plt.close(fig)
    write(out/'source_data/Sup_honest_full_availability.tsv',full)
    for p in (ROOT/'inputs').rglob('*.tsv'):
        # Full tables retain all 1,440 paired contrasts and every unavailable state.
        dst=out/'source_data'/p.parent.name/p.name;dst.parent.mkdir(exist_ok=True);dst.write_bytes(p.read_bytes())
    report(out, pts, paired, rules, ev)
    checks={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob('*') if p.is_file()}
    (out/'output_manifest.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps({'figures':7,'new_inference':False,
        'new_bootstrap':False,'formal_bootstrap_used':10000,'source_table_hashes_verified':True,
        'split_points_are_not_independent_replicates':True},indent=2),encoding='utf-8')
    print('COMPLETE',out)

def report(out, pts, paired, rules, ev):
    error=np.median([r['conditional_error_balanced_difference_pp'] for r in paired])
    cov=np.median([r['coverage_balanced_difference_pp'] for r in paired])
    loss=np.median([r['positive_discarded_fraction_balanced_difference_pp'] for r in paired])
    risks=[sum(r['conditional_error_balanced_'+s]>.05 for r in paired) for s in ['a','b']]
    s=f'''# 图文更新核对单

本目录仅从已保存的正式统计表生成图件，没有重新训练、推理或重采样，也没有自动修改 Word。

## 执行顺序

1. 将 Fig04_budget_baselines_and_family 作为图4修订候选。原预算曲线仍保留在补充材料，避免遗漏其他预算点。
2. 将 Fig05_honest_source_constraints 作为图5修订候选。原六折来源分解与选择器重拟合区间移至补充材料，说明旧分析与本次组划分验证的差别。
3. 将五张 Sup 图纳入补充材料；与现有补充图逐一核对后再确定编号，不能直接把文件名当成既有 S20–S24 的正式编号。
4. 保留图1–3、图6的现有数据；更新图1流程中的新增分析名称，以及正文和图注对图4/5的交叉引用。
5. 同步摘要、预算结果、跨来源结果、Discussion、Methods 与源数据索引。保留主要实验面板151条、121阳性/30阈值阴性；不与原始156条实验集的84阳性混用。
6. 最后再进行整篇 Word 高亮修订、参考文献和投稿格式检查；作者事实仍由作者填写。

## 数值与结论边界

- Fig4 主体为 ApexGO template-based 的100条：任一菌株任务86阳性、参考 E. coli 任务33阳性。两者预算均为20条。
- 任一菌株任务：六模型均命中20，随机期望17.2，最多可提升2.8。参考 E. coli 任务：六模型命中3.5–11，短序列优先命中12。分数并列时使用标签无关的分数权重分配，因此命中数可能为小数。
- 六模型与短序列优先的命中差异95%配对区间均包含0。不能称短序列优先显著更好，也不能称六模型稳定优于简单排序。全部六个基线方向必须报告。
- E. coli 的家族内结果包含9个同时具有两类标签的家族；任一菌株任务仅4个，不报告该家族内区间。家族内和总体AUROC是不同估计目标。
- 三规则分别458/600、312/600、201/600可行；每类600=6个留出来源×5模型×20组划分，不是600次独立实验。
- 在同一201个可配对实例中，逐来源约束相对支持匹配的混合规则：留出来源错误变化中位数{error:.2f}个百分点，覆盖变化{cov:.2f}个百分点，阳性丢弃变化{loss:.2f}个百分点。
- 在同一201个实例中，留出来源错误超过5%的次数为{risks[0]}→{risks[1]}；不能把不同可行子集的169和46直接比较。
- 这是更严格筛选的风险—代价关系，不是相同覆盖下排序能力提高，也不是分布外错误保证。
- c_AMPs*仅用最小分支分数作补充排序扩展，不纳入q规则；不应补造连续集成概率。

## 图4图注（英文草稿）

Fig. 4 | Task-dependent recovery relative to simple sequence baselines. All panels use the same 100 ApexGO template-based peptides. a,b, Expected active hits at a fixed budget of 20 peptides for activity against any eligible measured strain (86 positives) and reference E. coli ATCC 11775 (33 positives), respectively. The initial MIC threshold is 64 μM. Dashed orange lines indicate random-selection expectations. c,d, Pooled AUROC and equal-family mean within-family AUROC for the reference-strain task; nine families contribute to d. Six unfitted controls rank length, KR−DE residue counts and mean Kyte–Doolittle hydropathy in both directions. Filled points and horizontal lines show estimates with available pointwise exploratory 95% intervals from 10,000 paired dependence-block bootstrap resamples; open points indicate an omitted interval under the fixed reporting rules. Intervals are not multiplicity-adjusted. Tied scores receive label-blind fractional budget allocation. c_AMPs* denotes the supplementary minimum-branch ranking extension. Full contrasts and interval-availability states are supplied as source data.

## 图5图注（英文草稿）

Fig. 5 | Error reduction under source-specific constraints incurs a screening cost. a, Feasible rules among 120 planned instances per model and policy (six held-out sources × 20 group partitions). The pooled rule constrains source-balanced error; the support-matched rule adds minimum support in every development source; the source-specific rule additionally constrains each development-source error. b–d, Paired changes in held-source error, coverage and activity-positive rejection for the 201 instances in which both support-matched and source-specific rules are feasible. Differences are source-specific minus support-matched pooled rules. Each point is one group partition; black marks indicate within-model, within-held-source medians. Source labels after the slash identify the held-out source. The selection and validation sets use disjoint dependence groups, and groups shared with the held-out source are purged before partitioning. All thresholds are selected without validation or held-source labels. Repeated partitions reuse held-source records; their spread describes split sensitivity and is not a confidence interval. c_AMPs is ineligible because a verified native continuous ensemble probability is unavailable. Complete feasibility counts and unavailable states are supplied in the supplementary figures and source data.

## Methods 必须增加的信息

报告六个未拟合基线的定义和两个方向；固定预算ceil(0.2n)与并列规则；等家族平均AUROC及可评估家族条件；组重采样及区间缺失规则。三规则须逐项给出选择约束、最低支持、阈值搜索、分组隔离、五个开发来源与一个留出来源、70/30组划分和20次重复。重复分组范围不能写成95%置信区间。原分析与新增分析均为探索性，不声称事前注册。

## 可继续做、但不是本次结果已经证明的内容

如需主张“替代规则在相同实验代价下更好”，另需预先固定覆盖水平，在选择集确定阈值并冻结，然后在验证/留出来源报告覆盖、错误和阳性损失；这会成为新分析。当前不要为了增加图数再加多种事后阈值。当前也不能由序列相关性推断模型失效的因果机制。
'''
    (out/'manuscript_update_plan_zh.md').write_text(s,encoding='utf-8')
    supp='''# Supplementary figure definitions

Sup_budget_any_strain and Sup_budget_reference_Ecoli: Four cohorts, all six released models and all six unfitted baseline directions, at ceil(0.2n) slots. Dashed orange lines are random expected hits. ApexGO uses initial MIC ≤64 μM; HydrAMP uses ≤32 μg/mL. The reference strains are E. coli ATCC 11775 and ATCC 25922, respectively; units and strains are not pooled. Cohort sizes are 100, 20, 24 and 7. The source tables retain positive counts, threshold-negative counts, budgets and complete paired contrasts. Tied selections are fractional expectations. Error bars follow the fixed 10,000-block-bootstrap reporting rules. Open symbols have no reportable interval; consult status fields for the reason. c_AMPs* is a supplementary ranking extension.

Sup_family_any_strain and Sup_family_reference_Ecoli: Equal-family means of within-family AUROC among families containing both task-label classes. Informative-family counts are 4, 0, 3, 1 for the any-strain task and 9, 0, 3, 1 for reference E. coli. No within-family estimate exists for template-free singleton families. Available pointwise exploratory 95% intervals use dependence-block resampling; fewer than five contributing blocks or a degenerate distribution yields no interval. The absence of an interval is distinct from the absence of an estimate. Source data retain every state; open points are estimates with omitted intervals.

Sup_honest_full_source_availability: Feasible rules out of 20 repeated group partitions for every eligible model, held-out source and policy. Zero means no feasible rule under fixed constraints; it is not a missing measurement. Each policy plans 600 instances. Counts do not constitute independent replications. All unavailable and score-ineligible states remain in source data. c_AMPs is ineligible for these score-extremeness rules.
'''
    (out/'supplementary_legends_en.md').write_text(supp,encoding='utf-8')
    (out/'figure_contracts.json').write_text(json.dumps({
        'backend':'Python; matplotlib','archetype':'quantitative grid',
        'Fig04':'Fixed-budget recovery is task-dependent and must be interpreted against all simple baselines and family structure.',
        'Fig05':'Sourcewise constraints lower held-source error at a substantial coverage and positive-rejection cost, with limited feasibility.',
        'export':'183 mm; editable PDF/SVG; 600 dpi PNG/TIFF; minimum PDF glyph 5 pt',
        'exclusions':'No formal source-table rows removed. Main panels specify cohort/endpoint; complete cohort/task/method tables and five supplementary figures retain full results. Group-partition assignment file is omitted from this plotting-only package; the formal analysis archive remains authoritative.',
        'intervals':'Saved pointwise exploratory block-bootstrap intervals; repeated-partition scatter is NOT a CI.',
        'symbol_key':'Filled point: reportable CI; open point: omitted CI. c_AMPs*: supplementary ranking extension.',
        'analysis_boundary':'No new training, inference, threshold selection or bootstrap.'},ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','all'])
    ap.add_argument('--output',type=Path);a=ap.parse_args()
    pts,rules,ev=verify()
    if a.action=='all':
        out=a.output or ROOT/('output_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        render(out,pts,rules,ev)

if __name__=='__main__':main()
