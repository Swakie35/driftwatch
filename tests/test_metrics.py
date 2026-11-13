import numpy as np
import pandas as pd
import pytest
from driftwatch.metrics import psi,jensen_shannon,monitor,example

def test_identical_windows_have_zero_drift():
    rng=np.random.default_rng(7);x=rng.normal(size=300)
    assert abs(psi(x,x))<1e-8
    assert abs(jensen_shannon(['a','a','b'],['a','a','b']))<1e-8

def test_shift_is_detected():
    ref,cur=example()
    result=monitor(ref,cur)
    assert result['status']=='alert'
    assert {'psi','js'}=={row['metric'] for row in result['results']}
    assert any(x['alert'] for x in result['results'])

def test_constant_reference_not_crashing():
    assert psi(np.ones(100),np.ones(100))<1e-8
    assert psi(np.ones(100),np.ones(100)*2)>0

def test_schema_guard():
    with pytest.raises(ValueError):monitor(pd.DataFrame({'a':[1,2]}),pd.DataFrame({'b':[1,2]}))
    with pytest.raises(ValueError):psi([1],[2])
