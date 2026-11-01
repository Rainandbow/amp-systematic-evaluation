"""Same-cohort panels; no fitting or test-driven threshold selection."""
from pathlib import Path
import core as c


def metrics(records, tool, extension=False):
    import numpy as np
    from sklearn import metrics as m
    n = len(records)
    y = np.asarray([int(r["label"]) for r in records],dtype=int)
    d = np.asarray([int(r["prediction"]) for r in records],dtype=int)
    p = int(sum(y)); negative = n-p
    tp = int(sum((y==1)&(d==1))); tn = int(sum((y==0)&(d==0)))
    fp = negative-tn; fn = p-tp
    ratio = lambda a,b: float(a/b) if b else None
    denominator = float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    row = {"n":n,"positive":p,"negative":negative,"positive_prevalence":ratio(p,n),
           "tp":tp,"tn":tn,"fp":fp,"fn":fn,"accuracy":ratio(tp+tn,n),
           "recall":ratio(tp,p),"specificity":ratio(tn,negative),"precision":ratio(tp,tp+fp),
           "f1":ratio(2*tp,2*tp+fp+fn),"mcc":float((tp*tn-fp*fn)/denominator**.5) if denominator else (0.0 if n else None),
           "balanced_accuracy":((tp/p+tn/negative)/2.0) if p and negative else None,
           "auprc":None,"auroc":None,
           "ranking_score":"min of 3 submodel probabilities; recovery extension, NOT native ensemble probability"
                             if extension else ("not provided: native hard AND" if tool=="c_AMPs" else "released continuous AMP score")}
    if (tool != "c_AMPs" or extension) and p and negative:
        key = "extension_min_score" if extension else "score"
        score = np.asarray([float(r[key]) for r in records],dtype=float)
        c.require(np.isfinite(score).all() and ((0<=score)&(score<=1)).all(),"Invalid ranking scores")
        row["auprc"] = float(m.average_precision_score(y,score))
        row["auroc"] = float(m.roc_auc_score(y,score))
    return row


def source_matches(plan, cohorts):
    root = Path(plan["source_work"])
    stamp = c.read_json(root / "ingest_complete.json")
    c.verify_files(root,stamp["files"])
    provenance = root / "01_sources/record_provenance.tsv"
    index = {}
    mapping = {"AMPlify":("AMPlify","train","supplied original TRAIN"),
               "tsAMP_I":("tsAMP_I","train","supplied training subset; release membership incomplete"),
               "HMD_AMP":("HMD_AMP","training_source","supplied training pool; exact release membership not fully proved"),
               "Macrel":("AmPEP_raw_source","raw_source","raw AmPEP pool; NOT confirmed final Macrel TRAIN"),
               "ampir":("ampir_raw_mature90","raw_source","raw mature90 pool; NOT exact ampir v1.1 TRAIN")}
    for r in c.read_tsv(provenance):
        index.setdefault(r["sequence_id"],set()).add((r["dataset"],r["original_split"]))
    result = []
    for dataset in c.DATASETS:
        for tool in c.TOOLS:
            rows = [r for r in cohorts[dataset] if c.support(tool,r)]
            if tool in mapping:
                name,split,note = mapping[tool]
                n = sum((name,split) in index.get(r["sequence_id"],set()) for r in rows)
            else:
                n,note = "","original c_AMPs TRAIN membership not verified; test pool is not training evidence"
            result.append({"dataset":dataset,"tool":tool,"supported_test_n":len(rows),
                           "exact_matches_to_supplied_source_scope":n,"scope":note,
                           "independent_test_proven":False,"homology_membership_audit":"not performed here"})
    return result,{"ingest_manifest_sha256":c.sha(root / "ingest_complete.json"),"provenance_sha256":c.sha(provenance)}


def summarize(plan, cohorts, jobs):
    work = Path(plan["work"])
    c.prepared_checks(work)
    components = {name:c.collect_component(work,name,jobs) for name in c.COMPONENTS}
    all_results = {}; cell_stamps = []
    pred_fields = c.FINAL_FIELDS + ["lstm","attention","bert","extension_min_score"]
    for dataset in c.DATASETS:
        for tool in c.TOOLS:
            rows = []
            for r in cohorts[dataset]:
                if not c.support(tool,r):
                    continue
                uid = r["sequence_id"]
                row = dict(r,score="",prediction="",score_definition="",lstm="",attention="",bert="",extension_min_score="")
                if tool == "c_AMPs":
                    lstm = float(components["camps_keras"][uid]["lstm"])
                    attention = float(components["camps_keras"][uid]["attention"])
                    bert = float(components["camps_bert"][uid]["probability"])
                    row.update(lstm=lstm,attention=attention,bert=bert,
                               prediction=int(lstm>0.5 and attention>0.5 and bert>0.5),
                               extension_min_score=min(lstm,attention,bert),
                               score_definition="native hard AND; continuous ensemble probability undefined")
                else:
                    saved = components[tool][uid]
                    row.update(score=float(saved["probability"]),prediction=int(saved["prediction"]),
                               score_definition="native released score (Macrel retains native output rounding)")
                rows.append(row)
            all_results[dataset,tool] = rows
            directory = work / "04_predictions" / dataset / tool
            path = directory / "predictions.tsv"
            c.write_tsv(path,rows,pred_fields)
            c.dump(directory / "complete.json",{"version":c.VERSION,"dataset":dataset,"tool":tool,
                                                "n":len(rows),"files":{"predictions.tsv":c.sha(path)},
                                                "prepared_sha256":c.sha(work / "prepared.json")})
            cell_stamps.append(directory / "complete.json")
    output,extension_rows = [],[]
    for dataset in c.DATASETS:
        common = {r["sequence_id"] for r in cohorts[dataset] if all(c.support(tool,r) for tool in c.TOOLS)}
        short = {r["sequence_id"] for r in cohorts[dataset] if r["sequence_id"] in common and 10<=len(r["sequence"])<=49}
        for tool in c.TOOLS:
            full = all_results[dataset,tool]
            for panel,subset in [("tool_supported",full),
                                 ("common_six_tools",[r for r in full if r["sequence_id"] in common]),
                                 ("common_short_10_49",[r for r in full if r["sequence_id"] in short])]:
                base = {"version":c.VERSION,"dataset":dataset,"tool":tool,"panel":panel,
                        "original_test_n":len(cohorts[dataset]),"excluded_n":len(cohorts[dataset])-len(subset)}
                output.append(dict(base,**metrics(subset,tool)))
                if tool == "c_AMPs":
                    extension_rows.append(dict(base,method="c_AMPs_min_score_extension",**metrics(subset,tool,True)))
    directory = work / "05_reports"
    c.write_tsv(directory / "final_results.tsv",output,list(output[0]))
    c.write_tsv(directory / "camps_min_score_extension.tsv",extension_rows,list(extension_rows[0]))
    for panel in ("tool_supported","common_six_tools","common_short_10_49"):
        for metric in ("mcc","f1","recall","auprc","auroc","accuracy"):
            table = []
            for tool in c.TOOLS:
                row = {"tool":tool}
                for dataset in c.DATASETS:
                    selected = next(r for r in output if r["tool"]==tool and r["dataset"]==dataset and r["panel"]==panel)
                    row[dataset] = selected[metric]
                table.append(row)
            c.write_tsv(directory / (panel+"__"+metric+"_matrix.tsv"),table,["tool"]+c.DATASETS)
    matches,provenance = source_matches(plan,cohorts)
    c.write_tsv(directory / "training_source_match_summary.tsv",matches,list(matches[0]))
    files = {p.relative_to(work).as_posix():c.sha(p) for p in directory.iterdir() if p.is_file()}
    files.update({p.relative_to(work).as_posix():c.sha(p) for p in cell_stamps})
    if (work / "recovery_complete.json").exists():
        files["recovery_complete.json"] = c.sha(work / "recovery_complete.json")
    c.dump(work / "experiment_complete.json",{"version":c.VERSION,"complete":True,"expected_cells":36,
                                              "completed_cells":len(cell_stamps),"files":files,
                                              "prepared_sha256":c.sha(work / "prepared.json"),
                                              "source_provenance":provenance,"independent_test_proven":False})
    print("ALL 36 RELEASED-TOOL / DATASET CELLS COMPLETE:",directory / "final_results.tsv",flush=True)
