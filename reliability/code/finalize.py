"""Audit denominators and frozen results, then prepare a factual bilingual report."""
from pathlib import Path
import sys,json,math,hashlib,platform
import numpy as np
import pandas as pd
import legacy_common as C

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_PROTOCOL='1fe201f7eeb1248eadc52b420ad67940206466121af2521dad72a2b11ccc1c46'
DISPLAY={'Macrel':'Macrel','AMPlify':'AMPlify','ampir':'ampir','tsAMP_I':'tsAMP-I','HMD_AMP':'HMD-AMP'}
SOURCE={'AMPlify':'AMPlify','tsAMP_I':'tsAMP-I','c_AMPs_benchmark_resplit':'c_AMPs','HMD_AMP':'HMD-AMP','AmPEP_raw_source':'AmPEP','ampir_raw_mature90':'ampir'}

def read(name,where='results'):return pd.read_csv(ROOT/where/name,sep='\t',keep_default_na=True)
def save(name,df):df.to_csv(ROOT/'results'/name,sep='\t',index=False,encoding='utf-8-sig',lineterminator='\n')
def pp(x):return f'{100*x:+.2f}'
def interval_text(r):
    if pd.notna(r.ci_low) and pd.notna(r.ci_high):return f'{pp(r.estimate)} [{100*r.ci_low:.2f}, {100*r.ci_high:.2f}]'
    return f'{pp(r.estimate)} [区间未报告：{r.interval_status}]'

def main():
    import analyze as A
    A.selftest();C.verify_inputs();assert C.sha(ROOT/'protocol.json')==EXPECTED_PROTOCOL
    members=C.load_historical();panel,_,_=C.load_panel()
    assert len(members)==29104 and len({r['sequence_id'] for r in members})==28481
    ids={}
    for r in members:
        ids.setdefault(r['sequence_id'],set()).add(r['global_group'])
    assert all(len(g)==1 for g in ids.values())
    points=read('historical_retention_points.tsv');ints=read('historical_paired_intervals_80.tsv');cells=read('diagnostic_cells.tsv');fits=read('selector_coefficients.tsv');sens=read('repeated_partition_sensitivity_80_NOT_CI.tsv');rep=read('repeated_partition_contrasts.tsv');records=read('historical_record_diagnostics_repeat1.tsv.gz');auc=read('error_detection_AUROC.tsv')
    assert len(points)==144000 and len(ints)==450 and len(cells)==540 and len(fits)==2400 and len(sens)==450 and len(rep)==45000 and len(records)==145520 and len(auc)==210
    assert set(fits.status)<=set(['converged','single_error_class_constant'])
    split=read('analysis_split_audit.tsv','qa');assert len(split)==120
    for key in ['original_partition_reproduced','all_reference_neighbors_selection_only','no_query_group_reference']:assert split[key].astype(str).str.lower().eq('true').all()
    assert split.group_or_exact_overlap.astype(str).str.lower().eq('false').all()
    refs=read('inventory_nearest_reference.tsv.gz');assert len(refs)==143160
    assert refs.normalized_edit_distance.between(0,1).all()
    assert records.saved_score.between(0,1).all() and records.native_decision_confidence.between(0,1).all()
    assert records.inventory_distance.between(0,1).all() and records.selection_development_distance.between(0,1).all()
    assert np.allclose(points.actual_coverage,np.ceil(points.coverage_target*points.n-1e-10)/points.n,atol=1e-12)
    roles=C.read(ROOT/'inputs/inventory.tsv')
    role_by_model={r['tool']:r for r in roles}
    C.write(ROOT/'results/model_analysis_availability.tsv',[dict(model=m,native_classification_available=True,confidence_selection_eligible=m in A.MODELS,inventory_role=role_by_model[m]['inventory_role'],note='native ensemble probability unavailable' if m=='c_AMPs' else 'fixed saved AMP score; not assumed calibrated') for m in C.MODELS])
    for key in A.METRICS:
        v=points[key].dropna();assert (v>=-1e-10).all() and (v<=1+1e-10).all()
        full=points[points.coverage_target.eq(1)].groupby(['held_source','repeat','model','phase','dataset'])[key].agg(['min','max'])
        assert (full['max']-full['min']).abs().max()<1e-10
    curves=points.sort_values('coverage_target').groupby(['held_source','repeat','model','phase','dataset','method'])
    for _,g in curves:
        assert (np.diff(g.positive_rejection)<=1e-10).all()
        assert (np.diff(g.end_to_end_detection)>=-1e-10).all()
    for (source,model,role),g in cells.groupby(['held_source','model','reference_role']):assert g.n.sum()==sum(r['dataset']==source for r in members)
    assert ints.bootstrap.eq(10000).all() and ints.selector_refit.eq(False).all()
    reported=ints.interval_status.eq('pointwise_exploratory_percentile');assert ints.loc[reported,'valid_fraction'].ge(.95).all();assert ints.loc[reported,'contributing_groups'].ge(5).all()
    exp=read('experimental_paired_intervals_80.tsv');counts=read('experimental_task_denominators.tsv');er=read('experimental_record_diagnostics.tsv');assert len(exp)==72 and len(counts)==8 and len(er)==302
    assert counts[counts.task.eq('any_eligible_measured_strain')].n.sum()==151
    assert counts[counts.task.eq('any_eligible_measured_strain')].positive.sum()==121
    assert counts.groupby('cohort').n.first().to_dict()==dict(zip(C.COHORTS,[100,20,24,7]))
    assert exp.loc[exp.contributing_groups.lt(5),'ci_low'].isna().all()
    hist_summary=[]
    for (source,model),g in records.groupby(['dataset','model'],sort=False):
        high=g[g.native_decision_confidence.ge(.95)]
        for role,col in [('all','inventory_distance'),('inventory_exact','inventory_distance'),('inventory_nonexact','inventory_distance')]:
            h=high if role=='all' else high[high[col].eq(0) if role=='inventory_exact' else high[col].gt(0)]
            hist_summary.append(dict(held_source=source,model=model,reference_subset=role,high_confidence_n=len(h),errors=int(h.error.sum()),error_rate=h.error.mean(),positive=int(h.label.sum()),negative=len(h)-int(h.label.sum()),groups=h.global_group.nunique()))
    high=pd.DataFrame(hist_summary);save('high_confidence_counts_by_source.tsv',high)
    primary=ints[ints.method_a.eq('Logistic_C_L_I')&ints.method_b.eq('Logistic_C_L')].copy();assert len(primary)==90
    rows=[]
    for (source,model),g in primary.groupby(['held_source','model'],sort=False):
        row=dict(held_source=source,model=model,n=int(g.n.iloc[0]),positive=int(g.positive.iloc[0]))
        for metric in A.METRICS:
            r=g[g.metric.eq(metric)].iloc[0]
            row.update({metric+'_delta_pp':100*r.estimate,metric+'_low_pp':100*r.ci_low,metric+'_high_pp':100*r.ci_high,metric+'_interval_status':r.interval_status})
        rows.append(row)
    primary_wide=pd.DataFrame(rows);save('primary_80_paired_summary.tsv',primary_wide)
    reduced=primary_wide[primary_wide.conditional_error_delta_pp.lt(-1e-10)]
    vpoints=points[points['repeat'].eq(1)&points.coverage_target.eq(.8)]
    keys=['held_source','repeat','phase','dataset','model','n','positive','groups','actual_coverage']
    transfer=vpoints[vpoints.method.eq('Logistic_C_L_I')].merge(vpoints[vpoints.method.eq('Logistic_C_L')],on=keys,suffixes=('_with_inventory','_confidence_length'),validate='one_to_one')
    assert len(transfer)==180
    for metric in A.METRICS:transfer[metric+'_delta_pp']=100*(transfer[metric+'_with_inventory']-transfer[metric+'_confidence_length'])
    save('validation_to_transfer_primary80.tsv',transfer)
    validation=transfer[transfer.phase.eq('same_source_validation')]
    agg=[]
    for (method,baseline,metric),g in ints.groupby(['method_a','method_b','metric'],sort=False):
        ss=sens[sens.method_a.eq(method)&sens.method_b.eq(baseline)&sens.metric.eq(metric)]
        agg.append(dict(method_a=method,method_b=baseline,metric=metric,source_model_combinations=len(g),lower_point=int(g.estimate.lt(-1e-12).sum()),equal_point=int(g.estimate.abs().le(1e-12).sum()),higher_point=int(g.estimate.gt(1e-12).sum()),interval_below_zero=int(g.ci_high.lt(0).sum()),interval_above_zero=int(g.ci_low.gt(0).sum()),interval_includes_zero=int((g.ci_low.le(0)&g.ci_high.ge(0)).sum()),omitted_interval=int(g.ci_low.isna().sum()),all_20_partitions_lower=int(ss.negative_count.eq(20).sum()),all_20_partitions_higher=int(ss.positive_count.eq(20).sum()),descriptive_median_delta_pp=100*g.estimate.median(),aggregation_warning='source/model combinations are dependent; counts are descriptive, not replications or a pooled effect'))
    summary=pd.DataFrame(agg);save('contrast_summary_counts.tsv',summary)
    effect=primary[primary.metric.eq('conditional_error')];s=summary[summary.method_a.eq('Logistic_C_L_I')&summary.metric.eq('conditional_error')].iloc[0]
    all_high=high[high.reference_subset.eq('all')]
    # The report includes all 30 planned cases in fixed model/source order.
    table=['| 模型 | 留出来源 | 错误率差（百分点，95%区间） |','|---|---|---|']
    for model in A.MODELS:
        for source in C.SOURCES:
            r=effect[effect.model.eq(model)&effect.held_source.eq(source)].iloc[0]
            table.append(f'| {DISPLAY[model]} | {SOURCE[source]} | {interval_text(r)} |')
    apex=exp[exp.cohort.eq('ApexGO_template_based')&exp.task.eq('any_eligible_measured_strain')&exp.metric.eq('conditional_error')]
    sd=apex[apex.method_a.eq('Submodel_SD')].iloc[0];gap=apex[apex.method_a.eq('Entropy_gap')].iloc[0];iv=apex[apex.method_a.eq('Inventory_similarity')].iloc[0]
    q=read('native_confidence_vs_q.tsv','qa')
    report=f'''# 高置信错误诊断与同覆盖筛选增益：结果说明

本次已经将两步连起来完成：先定位高置信错误，再在相同保留量下检验参考距离的额外筛选价值。未运行新模型推理或湿实验，未改动正文、补充材料、GitHub和Zenodo。

## 最关键的结果

主比较是“置信分数＋长度＋模型对应库存距离”减去“置信分数＋长度”，两者均只用开发选择组拟合错误风险；在留出来源保留约80%的相同数量。30个来源—模型组合中，错误率点估计下降{s.lower_point}个、上升{s.higher_point}个、不变{s.equal_point}个。探索性逐点区间完全低于零{s.interval_below_zero}个、完全高于零{s.interval_above_zero}个、包含零{s.interval_includes_zero}个，另有{s.omitted_interval}个区间未报告。负差值表示错误减少；这些依赖的组合不能作为30次独立试验或成功率。

在错误率点估计下降的{len(reduced)}个组合中，{int(reduced.positive_rejection_delta_pp.gt(1e-10).sum())}个同时增加了阳性拒绝，{int(reduced.end_to_end_detection_delta_pp.lt(-1e-10).sum())}个降低了端到端检出。即使保留总量完全相同，留下哪些阳性也会改变，因此不能只看错误率一列。

20次开发组划分中始终下降的组合{s.all_20_partitions_lower}个、始终上升{s.all_20_partitions_higher}个。该分布是划分敏感性，不能当作置信区间。全部改善、恶化和区间未报告状态均保留。

第1划分的150个逐开发来源验证组合中，加入库存距离的错误率点估计下降{int(validation.conditional_error_delta_pp.lt(-1e-10).sum())}个、上升{int(validation.conditional_error_delta_pp.gt(1e-10).sum())}个。它们按来源分别保存于`results/validation_to_transfer_primary80.tsv`，与30个真正留出来源组合明确区分，不合并成更大的独立样本量。

因此，这项新增分析检验的是参考距离的适用边界，并不能支持“加入距离普遍改善预测”或“远离参考序列导致模型失败”的结论。历史来源标签及库存角色会影响这种关系。序列距离属于可观测的分布特征，不能拆分知识不确定性与偶然不确定性，也不构成因果机制解释。

## 高置信错误诊断说明了什么

在原生决策置信≥0.95的记录中，30个组合的来源标签错误率介于{100*all_high.error_rate.min():.1f}%与{100*all_high.error_rate.max():.1f}%之间，显示这一分数阈值不能直接解释为不超过5%的实际错误。距离关系也不是统一单调增加：例如Macrel在tsAMP-I来源中的选择参考距离T1–T3高置信错误率为69.7%、30.0%和13.1%；AMPlify在同一来源对应为5.7%、10.0%和22.2%。这些分层的类别组成和覆盖成员不同，因此它们是诊断，额外筛选价值必须由后续同覆盖、隔离来源比较确认。全分层分母和错误类型见图A源表，未单独挑选极端格作为总体机制结论。

## 实验队列中的检验

AMPlify的四个实验队列共151条候选，任一合格初始菌株任务下121条阳性；按原始MIC定义逐队列计算，参考大肠杆菌任务另外报告。ApexGO模板引导的100条候选中，保留80%时，相对原决策置信：子模型标准差的错误率差为{interval_text(sd)}个百分点，熵差为{interval_text(gap)}，库存距离为{interval_text(iv)}。子模型分歧尚未显示区间排除零的增益。HydrAMP的24条和7条队列只有4个和2个依赖块，保留描述性结果，不补造区间。20条ApexGO无模板队列也单独报告，不与100条队列合并。

## 三张图怎样用

- 图A：高置信决策（≥0.95）的错误率与参考距离分层。上排为各模型可用库存，下排为仅来自开发选择组的参考。每格同时写错误率和样本量；横线表示空格。距离分界在选择数据内固定，相同距离不会被强行切开。用于定位错误出现的条件，不能解释为校准或因果证据。
- 图B：30个组合在相同80%保留量下的配对差，依次为错误率、阳性拒绝和端到端检出。主比较只增加库存距离，排除把长度信息当成新增贡献的混淆。区间来自第1个预先指定划分的10,000次留出依赖组重采样，条件是开发拟合和参考固定。
- 图C：AMPlify在100条ApexGO模板引导候选中的实验任务检验。展示错误率—保留率、阳性拒绝—保留率以及80%保留时的配对错误率差。所有方法均按固定规则使用，未在实验标签上择优。

其他覆盖目标、开发参考距离、两种距离组合、每个来源的同来源验证、全部实验队列和两种任务均在源表中，不因结果不利而删除。

## 全部主比较

差值均为加入库存距离后的规则减去置信＋长度规则；单位为百分点。每个组合的阳性拒绝和端到端检出见`results/primary_80_paired_summary.tsv`。

{chr(10).join(table)}

## 边界与可复算性

1. 六模型中五个具备已核实的原生连续AMP分数，故该分析有五个模型。c_AMPs原生分类可用于论文既有分类比较；其分支最小值不是已验证的原生合成概率，本次不据此编造置信度。
2. 五份参考库存角色分别为Macrel来源代理、AMPlify提供训练集、ampir恢复训练集、tsAMP-I部分训练集和HMD-AMP训练来源池；不是五套已完全核定的拟合训练集。222,792条去重参考中排除620条非标准氨基酸序列后计算距离，逐库存分母保留。
3. 距离为全局单位代价Levenshtein距离除以两序列最大长度；它与原论文MMseqs相似度/90%和80%覆盖筛选不同。开发距离缓存为float32，完整参考搜索距离和最近序列编号保留。
4. 原生决策置信为c=d·p+(1−d)·(1−p)，不是将c≥0.95解释成95%正确。与q=max(p,1−p)不同的历史记录计{int(q.q_vs_native_confidence_different_n.sum())}个模型—来源记录，具体见`qa/native_confidence_vs_q.tsv`。
5. 跨来源按全局依赖组隔离；验证和留出序列不进入开发参考池。选择记录计算开发距离时还排除自身整个依赖组。所有划分复现已有70:30方案，标签只用于指定选择组拟合。
6. 本次为看到早期结果后追加、在新计算前固定的探索性方案；不是预注册确认性检验。区间是逐点95%百分位区间，未作多重比较校正，且未包含辅助模型重新拟合的不确定性。抽样区间和20次划分敏感性分开保存。
7. 结果包包含输入、校验和、方案、代码和图件；大距离矩阵缓存不进入压缩包，可按说明重建。输入含第三方序列，公开上传前沿用原始来源的许可范围，本次未重新授予许可。

建议先把本结果作为补充证据使用。若最终需要写入正文，核心增量应是“在相同保留量、组隔离验证和长度控制下检验可靠性信息的实际收益及其边界”，并完整报告改善与恶化；不能写成提出了一种已经普遍有效的新不确定性算法。
'''
    (ROOT/'RESULTS_zh.md').write_text(report,encoding='utf8')
    # Ready-to-use prose, with no reference markers added to the main manuscript.
    text=f'''# 本次新增分析的中英文表述（尚未写入Word）

## Methods — English

We linked high-confidence error diagnostics to matched-coverage retention experiments for five predictors with verified native AMP-class scores. Confidence in a saved decision d was c=dp+(1−d)(1−p), where p was the saved AMP score. We measured nearest-reference distance using global unit-cost Levenshtein distance normalized by the maximum sequence length. Model-specific available inventories retained their documented provenance roles. A separate development reference contained selection-group sequences only; held-source groups were purged and a selection query's own dependence group was excluded from its reference search. Reference distance was not interpreted as complete training independence.

Across six leave-one-source-out folds and 20 fixed group partitions, auxiliary ridge-logistic models predicted native decision errors from confidence and log length, with inventory distance, development distance or both added as fixed feature sets. Source-balanced fitting used selection labels only, without tuning on validation or held-source labels. Ranking by estimated correctness retained the same label-blind fractional-tie capacity at 20%, 40%, 60%, 80% and 100% coverage. The primary contrast added inventory distance to confidence and length at 80% retention. Paired 95% percentile intervals used 10,000 held-source dependence-group resamples for the prespecified first partition, conditional on the fixed development fit and references. The 20-partition spread was reported separately as sensitivity, not as a confidence interval. Experimental analyses compared fixed AMPlify confidence, reference distance and submodel-disagreement rankings within each cohort and activity task, without fitting an auxiliary selector to the experimental labels.

## 方法——中文

对五个具备已核实原生连续AMP分数的预测器，将高置信错误诊断与同覆盖筛选检验连接。已保存决策d的置信分数定义为c=dp+(1−d)(1−p)，p为AMP分数。参考距离为全局单位代价Levenshtein距离除以两序列最大长度。模型对应库存按其实际角色报告；另外构建只含选择组序列的开发参考，清除留出来源依赖组，并在选择记录的参考搜索中排除其自身整个依赖组。参考距离不解释为完整的训练独立性。

在六折逐来源留出、20次固定依赖组划分中，辅助岭逻辑回归以置信和对数长度预测原生决策错误，并按固定方案加入库存距离、开发参考距离或二者。来源平衡的拟合只使用选择组标签，验证及留出标签不参与调参。按预测正确性排序，在20%、40%、60%、80%和100%目标覆盖率保留相同数量，边界并列采用与标签无关的分数权重。主比较为80%保留时，在置信与长度之外加入库存距离。逐点95%百分位区间来自预先指定第1划分的10,000次留出依赖组重采样，条件是开发拟合与参考固定。20次划分另作敏感性报告。实验队列仅比较固定AMPlify置信、参考距离及子模型分歧规则，未用实验标签拟合辅助筛选器。

## Results — English

At matched 80% retention, adding model-specific inventory distance to confidence and length reduced held-source error point estimates in {s.lower_point} of 30 source–model combinations and increased them in {s.higher_point}; {s.equal_point} were unchanged. Pointwise conditional intervals excluded zero in the direction of lower error for {s.interval_below_zero} combinations and higher error for {s.interval_above_zero}. These counts summarize dependent comparisons rather than independent replications. Reference information therefore had source- and model-dependent utility. In the 100-candidate ApexGO template-based cohort, the paired error difference for AMPlify submodel SD versus decision confidence was {100*sd.estimate:.2f} percentage points (95% interval, {100*sd.ci_low:.2f} to {100*sd.ci_high:.2f}), providing no clear evidence of improved retention under the evaluated activity task.

## 结果——中文

在相同80%保留量下，加入模型对应库存距离后，30个来源—模型组合中{s.lower_point}个的留出错误率点估计下降、{s.higher_point}个上升、{s.equal_point}个不变。逐点条件区间在{s.interval_below_zero}个组合中完全位于错误减少方向，在{s.interval_above_zero}个组合中完全位于错误增加方向。这些计数来自相互依赖的比较，不代表独立重复。参考信息的筛选价值因此依赖来源和模型。100条ApexGO模板引导候选中，AMPlify子模型标准差相对原决策置信的配对错误率差为{100*sd.estimate:.2f}个百分点（95%区间{100*sd.ci_low:.2f}至{100*sd.ci_high:.2f}），尚未显示在该活动任务下明确改善保留决策的证据。
'''
    (ROOT/'MANUSCRIPT_TEXT_bilingual.md').write_text(text,encoding='utf8')
    qa=dict(passed=True,historical_records=29104,historical_unique_sequences=28481,historical_score_models=5,experimental_candidates=151,experimental_any_strain_positives=121,reference_rows=len(refs),partitions=120,auxiliary_fits=len(fits),all_fits_converged_or_explicit_constant=True,all_methods_full_coverage_equal=True,retained_positive_and_detection_monotone=True,primary_contrasts=30,bootstrap=10000,selector_refit=False,protocol_sha256=EXPECTED_PROTOCOL,word_modified=False,github_zenodo_modified=False)
    C.dump(ROOT/'qa/final_statistical_audit.json',qa)
    C.dump(ROOT/'results/ANALYSIS_COMPLETE.json',qa)
    print('FINAL STATISTICAL AUDIT PASS',json.dumps(qa,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
