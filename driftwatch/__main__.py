import argparse
import json
import pandas as pd
from pathlib import Path
from .metrics import demo,monitor
p=argparse.ArgumentParser(description='Compare current model inputs with a reference CSV')
p.add_argument('--reference',help='Reference CSV (optional; otherwise run synthetic demo)')
p.add_argument('--current',help='Current CSV')
p.add_argument('--psi-threshold',type=float,default=.2)
p.add_argument('--js-threshold',type=float,default=.1)
p.add_argument('--report',default='reports/generated/drift.json')
a=p.parse_args()
if bool(a.reference)!=bool(a.current):p.error('--reference and --current must be supplied together')
if a.psi_threshold<0 or a.js_threshold<0:p.error('thresholds must not be negative')
if a.reference:
    r=monitor(pd.read_csv(a.reference),pd.read_csv(a.current),{'psi':a.psi_threshold,'js':a.js_threshold})
    path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r,indent=2)+'\n')
else:r=demo(a.report)
print(r['status'].upper(),[(x['feature'],x['metric'],x['value']) for x in r['results']])
print('Wrote',a.report)
