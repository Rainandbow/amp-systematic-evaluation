# Linked diagnostics and matched-retention analysis

Run from repository root using release.py. This public adaptation replaces historical letters with verified hashes and lengths and supplies the frozen distance/selection feature inputs needed for statistical recalculation. Seeds, labels, estimators and protocols are unchanged. See docs/RELIABILITY_DATA_ACCESS.md for exact scope and external raw-distance inputs.

python release.py check
python release.py reliability-statistics --output ../fresh_reliability
python release.py reliability-figures --output ../fresh_figures

The reliability module's conditional-only comparison is retained as supporting evidence. The manuscript's updated S29 and S16 use refit/, including joint development-group refitting. Do not substitute the older conditional-only plot for updated S29.
