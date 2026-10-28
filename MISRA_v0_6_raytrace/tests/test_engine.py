import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import Config,trace

def test_flat():
    a=trace(Config(n=32,mirror='flat',phase_steps=4))
    assert a['frames'].shape==(4,32,32)
    assert np.all(np.isfinite(a['frames']))
    assert a['valid'].any()

def test_curvature_changes():
    a=trace(Config(n=32,mirror='flat'))
    b=trace(Config(n=32,mirror='spherical',radius_mm=200))
    mask=a['valid']&b['valid']
    assert np.std((a['opd_m']-b['opd_m'])[mask])>1e-10

def test_phase_steps():
    a=trace(Config(n=32,phase_steps=4))
    assert np.allclose(a['phase_shifts_rad'],[0,np.pi/2,np.pi,3*np.pi/2])

def test_reproducible():
    c=Config(n=32,noise_std=.02)
    assert np.array_equal(trace(c)['frames'],trace(c)['frames'])

def test_aperture():
    a=trace(Config(n=32,aperture_mm=.4,fov_mm=3))
    assert not a['valid'].all()
