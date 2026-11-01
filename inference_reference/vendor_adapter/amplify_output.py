"""Validate native AMPlify TSV without altering any score or decision (Python 3.6+)."""
MEAN_ATOL = 5e-7
OLD_MEAN_ATOL = 2e-7


def amplify_frame(path, expected):
    import numpy as np
    import pandas as pd
    import math
    frame = pd.read_csv(path,sep="\t",dtype={"Sequence_ID":str,"Sequence":str})
    columns = ["Sub_model_{}_probability_score".format(i) for i in range(1,6)]
    required = {"Sequence_ID","Sequence","Probability_score","Prediction"}.union(columns)
    if not required.issubset(frame.columns):
        raise ValueError("Missing native AMPlify columns")
    if frame.Sequence_ID.isna().any() or not frame.Sequence_ID.is_unique:
        raise ValueError("Missing/duplicate AMPlify IDs")
    if expected.seq_uid.isna().any() or not expected.seq_uid.is_unique:
        raise ValueError("Invalid expected IDs")
    if len(frame)!=len(expected) or set(frame.Sequence_ID)!=set(expected.seq_uid.astype(str)):
        raise ValueError("AMPlify coverage mismatch; no missing prediction filled")
    aligned = frame.set_index("Sequence_ID").loc[expected.seq_uid.astype(str)].reset_index()
    if not np.array_equal(aligned.Sequence.to_numpy(),expected.sequence.to_numpy()):
        raise ValueError("AMPlify ID-matched sequences differ")
    scores = aligned.Probability_score.to_numpy(dtype=float)
    subs = aligned[columns].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not np.isfinite(subs).all() or (
            (scores<0)|(scores>1)).any() or ((subs<0)|(subs>1)).any():
        raise ValueError("Invalid native AMPlify probabilities")
    mean = np.asarray([math.fsum(row)/5 for row in subs],dtype=float)
    delta = np.abs(scores-mean)
    if (delta>MEAN_ATOL).any():
        index = int(delta.argmax())
        raise ValueError("AMPlify mean mismatch exceeds FP32/TSV tolerance: {} delta={} bound={}".format(
            aligned.Sequence_ID.iloc[index],float(delta[index]),MEAN_ATOL))
    decisions = aligned.Prediction.map({"AMP":1,"non-AMP":0})
    if decisions.isna().any():
        raise ValueError("Invalid native AMPlify classes")
    if ((decisions.to_numpy()!=(scores>0.5).astype(int))&(np.abs(scores-0.5)>1.1e-7)).any():
        raise ValueError("AMPlify classes contradict unchanged native >0.5 rule")
    result = expected[["seq_uid","sequence","label"]].copy().reset_index(drop=True)
    result["probability"] = scores
    result["prediction"] = decisions.to_numpy(dtype=int)
    for i,column in enumerate(columns):
        result[column] = subs[:,i]
    audit = {"n":len(scores),"mean_absolute_tolerance":MEAN_ATOL,
             "old_mean_absolute_tolerance":OLD_MEAN_ATOL,
             "max_absolute_delta":float(delta.max()) if len(delta) else None,
             "above_old_bound":int((delta>OLD_MEAN_ATOL).sum()),
             "scores_replaced_or_recomputed":False,"native_decisions_unchanged":True,
             "reason":"FP32 ensemble computation and independently rounded/serialized TSV values"}
    return result,audit
