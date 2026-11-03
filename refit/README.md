# Joint development refitting and held-source resampling

Shared inputs are in ../reliability, including its six required development feature files. The protocol uses 2,000 paired draws, five models, six held sources and three metrics. Original AMP models, the first frozen partition and references remain fixed.

python release.py refit-smoke --output ../fresh_smoke
python release.py refit-statistics --output ../fresh_refit
python release.py refit-figure --output ../fresh_s29

Run these from the repository root. Fresh outputs never silently replace archived formal arrays. A 20-draw smoke test checks execution only; it is not formal evidence. The formal calculation retains 2,000 draws. Cache identities include adapted code, inputs, protocol and runtime and reject stale results.
