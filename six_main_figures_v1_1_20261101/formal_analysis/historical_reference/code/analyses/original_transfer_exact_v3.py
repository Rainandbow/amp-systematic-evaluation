#!/usr/bin/env python3
"""Exact observed-score threshold transfer; saved predictions only, unchanged risk constraints."""
import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
SEED = 2026102302


def prepare(dev, model):
    groups, gi = np.unique([r['global_group'] for r in dev], return_inverse=True)
    sources, si = np.unique([r['dataset'] for r in dev], return_inverse=True)
    assert len(sources) == 5
    p = np.array([float(r[model+'__score']) for r in dev])
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    q = np.maximum(p, 1-p)
    error = np.array([int(r[model+'__prediction']) != int(r['label']) for r in dev])
    order = np.argsort(-q, kind='stable')
    q = q[order]
    ends = np.r_[np.flatnonzero(q[1:] != q[:-1]), len(q)-1]
    gi, si = gi[order], si[order]
    _, first = np.unique(gi, return_index=True)
    first_record = np.zeros(len(dev), dtype=bool)
    first_record[first] = True
    return dict(G=len(groups), gi=gi, si=si, error=error[order], ends=ends,
                thresholds=q[ends], first=first_record, sources=sources)


def select_exact(D, prep, curve=False):
    counts = D[:, prep['gi']].astype(float)
    totals = np.column_stack([counts[:, prep['si'] == s].sum(axis=1) for s in range(5)])
    denominator = totals[:, prep['si']]
    weights = np.divide(counts, 5*denominator, out=np.zeros_like(counts), where=denominator > 0)
    ends = prep['ends']
    coverage = np.cumsum(weights, axis=1)[:, ends]
    errors = np.cumsum(weights * prep['error'], axis=1)[:, ends]
    risk = np.divide(errors, coverage, out=np.full_like(errors, np.nan), where=coverage > 0)
    retained = np.cumsum(counts, axis=1)[:, ends]
    group_n = np.cumsum((counts > 0) & prep['first'], axis=1)[:, ends]
    ok = ((risk <= .05+1e-12) & (coverage >= .2-1e-12) & (retained >= 100)
          & (group_n >= 30) & np.all(totals > 0, axis=1)[:, None])
    starts = np.r_[0, ends[:-1]+1]
    ok &= np.add.reduceat(counts, starts, axis=1) > 0
    feasible = ok.any(axis=1)
    last = ok.shape[1]-1-np.argmax(ok[:, ::-1], axis=1)
    idx = np.where(feasible, last, -1)
    thresholds = np.where(feasible, prep['thresholds'][last], np.inf)
    result = dict(index=idx, threshold=thresholds, feasible=feasible)
    for name, values in [('coverage', coverage), ('error', risk), ('retained_records', retained), ('groups', group_n)]:
        result[name] = np.where(feasible, values[np.arange(len(D)), last], np.nan)
    if curve:
        result['curve'] = dict(threshold=prep['thresholds'], coverage=coverage[0], error=risk[0], retained_records=retained[0], contributing_groups=group_n[0], selected=np.arange(len(ends)) == idx[0])
    return result


def target_metrics(W, q, y, prediction, selection, metrics):
    kept = W*(q[None, :] >= selection['threshold'][:, None])
    values = metrics(W, kept, y, prediction)
    for v in values.values():
        v[~selection['feasible']] = np.nan
    return values


def run(args):
    sys.path.insert(0, str(args.root/'code'))
    from selective import TOOLS, DATASETS, load, write, metrics, interval
    if args.bootstrap < 10000 and not args.smoke:
        raise ValueError('Formal analysis requires at least 10000 replicates; --smoke is QA only')
    args.output.mkdir(parents=True, exist_ok=False)
    fingerprint = {name: hashlib.sha256((args.root/'inputs'/name).read_bytes()).hexdigest() for name in ['frozen_members_and_groups.tsv.gz', 'historical_scores.tsv.gz']}
    rows = load()
    points, intervals, folds, grid, threshold_summary = [], [], [], [], []
    begun = time.monotonic()
    with gzip.open(args.output/'bootstrap_thresholds.tsv.gz', 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['held_source', 'model', 'replicate', 'feasible', 'threshold'])
        for di, held in enumerate(DATASETS):
            target = [r for r in rows if r['dataset'] == held]
            forbidden = {r['global_group'] for r in target}
            before = [r for r in rows if r['dataset'] != held]
            dev = [r for r in before if r['global_group'] not in forbidden]
            assert not forbidden.intersection(r['global_group'] for r in dev)
            assert not {r['sequence_id'] for r in target}.intersection(r['sequence_id'] for r in dev)
            groups_t, tgi = np.unique([r['global_group'] for r in target], return_inverse=True)
            GT = len(groups_t)
            y = np.array([int(r['label']) for r in target])
            for model in TOOLS:
                prep = prepare(dev, model)
                G = prep['G']
                score = np.array([float(r[model+'__score']) for r in target])
                q = np.maximum(score, 1-score)
                prediction = np.array([int(r[model+'__prediction']) for r in target])
                selection = select_exact(np.ones((1, G)), prep, curve=True)
                values = target_metrics(np.ones((1, len(target))), q, y, prediction, selection, metrics)
                good = bool(selection['feasible'][0])
                row = dict(held_source=held, model=model, policy_status='feasible' if good else 'no_feasible_policy', threshold=float(selection['threshold'][0]) if good else float('nan'), development_records=len(dev), development_groups=G, purged_development_records=len(before)-len(dev), test_records=len(target), test_groups=GT, development_coverage=float(selection['coverage'][0]), development_error=float(selection['error'][0]))
                row.update({name: float(v[0]) for name, v in values.items()})
                c = selection['curve']
                for j in range(len(c['threshold'])):
                    grid.append(dict(held_source=held, model=model, **{name: values[j].item() for name, values in c.items()}))
                folds.append(dict(held_source=held, model=model, development_sources=[s for s in DATASETS if s != held], development_n=len(dev), test_n=len(target), development_groups=G, test_groups=GT, group_overlap=0, exact_overlap=0, observed_unique_thresholds=len(c['threshold'])))
                rng = np.random.default_rng(SEED+di*100)
                boot = {name: [] for name in values}
                boot_thresholds = []
                feasible_count = 0
                for start in range(0, args.bootstrap, args.batch):
                    b = min(args.batch, args.bootstrap-start)
                    D = rng.multinomial(G, np.ones(G)/G, size=b)
                    W = rng.multinomial(GT, np.ones(GT)/GT, size=b)[:, tgi].astype(float)
                    selected = select_exact(D, prep)
                    feasible_count += int(selected['feasible'].sum())
                    selected_q = np.where(selected['feasible'], selected['threshold'], np.nan)
                    boot_thresholds.append(selected_q)
                    data = target_metrics(W, q, y, prediction, selected, metrics)
                    for name, v in data.items():
                        boot[name].append(v)
                    for j, threshold in enumerate(selected_q):
                        writer.writerow([held, model, start+j+1, bool(selected['feasible'][j]), format(float(threshold), '.17g') if np.isfinite(threshold) else ''])
                fraction = feasible_count/args.bootstrap
                row['bootstrap_policy_feasible_fraction'] = fraction
                for name, chunks in boot.items():
                    low, high, status, valid = interval(np.concatenate(chunks), args.bootstrap)
                    if not good:
                        low = high = float('nan')
                        status = 'point_policy_infeasible'
                    intervals.append(dict(held_source=held, model=model, metric=name, ci_low=low, ci_high=high, status=status, valid_fraction=valid, policy_feasible_fraction=fraction, bootstrap=args.bootstrap))
                low, high, status, valid = interval(np.concatenate(boot_thresholds), args.bootstrap)
                if not good:
                    low = high = float('nan')
                    status = 'point_policy_infeasible'
                threshold_summary.append(dict(held_source=held, model=model, threshold=row['threshold'], ci_low=low, ci_high=high, status=status, valid_fraction=valid, bootstrap_policy_feasible_fraction=fraction, bootstrap=args.bootstrap))
                points.append(row)
                print('EXACT TRANSFER', held, model, row['policy_status'], 'q', row['threshold'], 'feasible resamples', round(fraction,4), 'elapsed_s', round(time.monotonic()-begun,1), flush=True)
    write(args.output/'transfer_points.tsv', points)
    write(args.output/'transfer_intervals.tsv', intervals)
    write(args.output/'development_threshold_grid.tsv', grid)
    write(args.output/'threshold_stability.tsv', threshold_summary)
    missing = [dict(held_source=r['held_source'], model=r['model'], reason='No observed-score threshold satisfied all fixed development constraints', bootstrap_policy_feasible_fraction=r['bootstrap_policy_feasible_fraction']) for r in points if r['policy_status'] != 'feasible']
    missing += [dict(held_source=s, model='c_AMPs', reason='Ineligible: no verified native continuous ensemble probability', bootstrap_policy_feasible_fraction='') for s in DATASETS]
    write(args.output/'Supplementary_policy_unavailable.tsv', missing)
    (args.output/'fold_audit.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
    settings = dict(version='transfer_exact_v3', bootstrap=args.bootstrap, seed=SEED, batch=args.batch, smoke_only=args.bootstrap < 10000, target_error=.05, min_coverage=.2, min_retained_records=100, min_retained_groups=30, threshold_search='All unique development score-extremeness values; descending cumulative scan; include q equal to threshold; full precision retained; candidate breakpoints determined without held-source labels; zero-mass score tiers excluded in each bootstrap resample', selection='Maximize source-balanced development coverage among feasible candidates; lowest observed threshold among equal-coverage feasible candidates', amendment='Post hoc correction of coarse 0.005 grid following discovery of missed feasible thresholds; original outputs preserved; fixed constraints and model weights unchanged; not preregistered', source_balance='Equal original development-source mass; conditional retained error uses retained weighted mass', group_exclusion='Global components present in the held source removed from development', shared_draws_across_models=True, intervals='Development selector re-fitted and target groups resampled each replicate; pointwise exploratory percentile; infeasible replicates retained as NA with feasibility fraction reported separately', uncertainty_boundary='No claim of guaranteed future error; no guarantee of model-training independence', input_sha256=fingerprint, elapsed_s=time.monotonic()-begun)
    assert fingerprint == {name: hashlib.sha256((args.root/'inputs'/name).read_bytes()).hexdigest() for name in fingerprint}
    (args.output/'run_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    print('EXACT TRANSFER COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
