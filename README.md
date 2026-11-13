# Driftwatch

### Spot changes in incoming ML features before they're hidden inside a dashboard.

I wanted a lightweight way to explain *why* a model-monitoring alert fired. Driftwatch compares reference and current data windows, runs population stability index (PSI) on numerical features and Jensen–Shannon divergence on categorical ones, and emits an inspection-friendly JSON report with per-feature thresholds.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
python -m driftwatch --report reports/generated/drift.json
pytest -q
```

The synthetic demo deliberately shifts basket values and web/store channel mix, so you can see an alert. The same functions can be used with your own `pandas.DataFrame` pairs, as long as the schemas match.

## Design choices
- Numeric bins are fitted exclusively on the reference window. Recomputing bins with live data would conceal some movement.
- Smoothed proportions prevent `log(0)` and support unseen categorical values.
- Tests cover unchanged windows, distribution shifts, constant features, and bad schemas.
- Reports separate metric, measured value, configured threshold and alert flag.

## What this does not tell you
PSI/JS describe feature distribution change, **not causal model degradation**. There is no model-performance monitoring, delayed ground truth, seasonal adjustment or alert routing. Production use should also add feature contracts, missingness monitoring, sample-size policies, rolling reference windows, and false-alarm evaluation.

To compare two real CSV files (no identifiers or secrets should be committed):
```bash
python -m driftwatch --reference baseline.csv --current latest.csv \
  --psi-threshold 0.2 --js-threshold 0.1 --report reports/generated/drift.json
```
