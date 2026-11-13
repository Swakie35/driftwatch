"""Reference-vs-current drift; metrics are descriptive, not proof of model harm."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-6

@dataclass
class DriftResult:
    feature: str
    metric: str
    value: float
    threshold: float
    alert: bool
    reference_count: int
    current_count: int

def _probability(counts:np.ndarray)->np.ndarray:
    adjusted=np.asarray(counts,dtype=float)+EPS
    return adjusted/adjusted.sum()

def psi(reference, current, bins:int=10)->float:
    """Population stability index using bins determined only from the reference."""
    a=np.asarray(reference,dtype=float);b=np.asarray(current,dtype=float)
    a=a[np.isfinite(a)];b=b[np.isfinite(b)]
    if len(a)<5 or len(b)<5: raise ValueError('PSI requires at least 5 finite values per window')
    if bins<2:raise ValueError('bins must be >=2')
    edges=np.unique(np.quantile(a,np.linspace(0,1,bins+1)))
    if len(edges)<2:
        # Same-valued reference: represent reference as central bin.
        v=float(a[0]);edges=np.array([-np.inf,np.nextafter(v,-np.inf),np.nextafter(v,np.inf),np.inf])
    else:
        edges[0]=-np.inf;edges[-1]=np.inf
    pa=_probability(np.histogram(a,bins=edges)[0]);pb=_probability(np.histogram(b,bins=edges)[0])
    return float(np.sum((pb-pa)*np.log(pb/pa)))

def jensen_shannon(reference,current)->float:
    """Categorical JS divergence, natural-log units; includes previously unseen labels."""
    a=pd.Series(reference).fillna('<NULL>').astype(str).value_counts()
    b=pd.Series(current).fillna('<NULL>').astype(str).value_counts()
    if not len(reference) or not len(current): raise ValueError('empty window')
    keys=a.index.union(b.index)
    pa=_probability(a.reindex(keys,fill_value=0).to_numpy())
    pb=_probability(b.reindex(keys,fill_value=0).to_numpy())
    mid=.5*(pa+pb)
    return float(.5*np.sum(pa*np.log(pa/mid))+.5*np.sum(pb*np.log(pb/mid)))

def monitor(reference:pd.DataFrame,current:pd.DataFrame,thresholds:dict|None=None)->dict:
    if list(reference.columns)!=list(current.columns):
        raise ValueError('reference/current schema mismatch (including column order)')
    if reference.empty or current.empty:raise ValueError('empty monitoring window')
    t={'psi':.2,'js':.1,**(thresholds or {})}
    rows=[]
    for feature in reference.columns:
        numeric=pd.api.types.is_numeric_dtype(reference[feature]) and pd.api.types.is_numeric_dtype(current[feature])
        metric='psi' if numeric else 'js'
        if numeric:
            v=psi(reference[feature].to_numpy(),current[feature].to_numpy())
            n1=int(np.isfinite(reference[feature]).sum());n2=int(np.isfinite(current[feature]).sum())
        else:
            v=jensen_shannon(reference[feature].tolist(),current[feature].tolist())
            n1=len(reference);n2=len(current)
        rows.append(asdict(DriftResult(feature,metric,round(v,6),t[metric],v>t[metric],n1,n2)))
    return {'status':'alert' if any(r['alert'] for r in rows) else 'ok',
            'method':'reference-fitted PSI and categorical Jensen-Shannon divergence',
            'results':rows}

def example(seed:int=42)->tuple[pd.DataFrame,pd.DataFrame]:
    rng=np.random.default_rng(seed)
    ref=pd.DataFrame({'basket_value':rng.normal(50,8,500),
                      'orders_per_day':rng.poisson(5,500),
                      'channel':rng.choice(['web','store'],500,p=[.45,.55])})
    current=pd.DataFrame({'basket_value':rng.normal(61,9,500),
                          'orders_per_day':rng.poisson(5,500),
                          'channel':rng.choice(['web','store'],500,p=[.75,.25])})
    return ref,current

def demo(path:str|Path):
    result=monitor(*example())
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2)+'\n')
    return result
