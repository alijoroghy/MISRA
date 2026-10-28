"""MISRA v0.6: vectorized, detector-pixel backward geometric ray tracing.
Coordinate convention: BS x=y, source (-S,0,0), reference y=Lr,
test x=Lt+sag(y,z), detector y=-D. Units metres.
No paraxial mirror intersection approximation; no diffraction/polarization.
"""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class Config:
    n: int = 256
    wavelength_nm: float = 632.8
    fov_mm: float = 3.0
    reference_mm: float = 50.0
    test_mm: float = 50.0
    detector_mm: float = 50.0
    source_mm: float = 70.0
    radius_mm: float = 1000.0  # signed, positive sag points +x
    mirror: str = 'spherical'  # flat/spherical/parabolic
    tilt_y_mrad: float = 0.0
    tilt_z_mrad: float = 0.0
    aperture_mm: float = 8.0
    noise_std: float = 0.0
    seed: int = 42
    source: str = 'plane'  # plane or point (wavefront phase at source plane)
    phase_steps: int = 4

def norm(v):
    return np.sqrt(np.sum(v*v, axis=-1))

def unit(v):
    return v/np.maximum(norm(v)[...,None],1e-30)

def reflect(d,n):
    return d-2*np.sum(d*n,axis=-1)[...,None]*n

def hit_plane(p,d,axis,value):
    t=(value-p[...,axis])/np.where(np.abs(d[...,axis])<1e-14,np.nan,d[...,axis])
    return p+t[...,None]*d,t

def sag(y,z,c):
    r2=y*y+z*z
    if c.mirror=='flat':
        h=np.zeros_like(y); gy=np.zeros_like(y);gz=np.zeros_like(y); valid=np.ones_like(y,dtype=bool)
    elif c.mirror=='parabolic':
        R=c.radius_mm*1e-3
        h=r2/(2*R);gy=y/R;gz=z/R;valid=np.ones_like(y,dtype=bool)
    else:
        R=c.radius_mm*1e-3
        valid=r2<R*R
        root=np.sqrt(np.maximum(R*R-r2,1e-30))
        h=np.sign(R)*(abs(R)-root)
        gy=np.sign(R)*y/root;gz=np.sign(R)*z/root
    gy=gy+c.tilt_y_mrad*1e-3;gz=gz+c.tilt_z_mrad*1e-3
    h=h+c.tilt_y_mrad*1e-3*y+c.tilt_z_mrad*1e-3*z
    return h,gy,gz,valid

def hit_test(p,d,c):
    L=c.test_mm*1e-3
    t=(L-p[...,0])/np.where(np.abs(d[...,0])<1e-14,np.nan,d[...,0])
    for _ in range(20):
        q=p+t[...,None]*d
        h,gy,gz,valid=sag(q[...,1],q[...,2],c)
        f=q[...,0]-L-h
        df=d[...,0]-gy*d[...,1]-gz*d[...,2]
        delta=f/np.where(np.abs(df)<1e-12,np.nan,df)
        t=t-delta
    q=p+t[...,None]*d
    h,gy,gz,valid=sag(q[...,1],q[...,2],c)
    residual=np.abs(q[...,0]-L-h)
    normal=unit(np.stack([np.ones_like(gy),-gy,-gz],axis=-1))
    return q,t,normal,valid&(residual<1e-8)&np.isfinite(t)

def bs_intersection(p,d):
    den=d[...,0]-d[...,1]
    t=(p[...,1]-p[...,0])/np.where(np.abs(den)<1e-14,np.nan,den)
    return p+t[...,None]*d,t

def trace(c:Config):
    if c.n<16 or c.n>2048: raise ValueError('n must be 16..2048')
    if c.wavelength_nm<=0 or c.aperture_mm<=0: raise ValueError('wavelength/aperture must be positive')
    if c.mirror not in ('flat','spherical','parabolic'): raise ValueError('invalid mirror')
    if c.mirror!='flat' and abs(c.radius_mm)*1e-3 <= c.aperture_mm*5e-4: raise ValueError('mirror radius too small for aperture')
    if c.phase_steps not in (1,4): raise ValueError('phase_steps must be 1 or 4')
    # detector pixel rays reverse-propagate along +y to BS, one ray per pixel per arm
    grid=(np.arange(c.n)-(c.n-1)/2)*c.fov_mm*1e-3/c.n
    xx,zz=np.meshgrid(grid,grid)
    shape=xx.shape
    p=np.stack([xx,np.full(shape,-c.detector_mm*1e-3),zz],axis=-1)
    d=np.broadcast_to(np.array([0.,1.,0.]),p.shape).copy()
    b,tb=bs_intersection(p,d)
    # BS reflection: normal (1,-1,0)/sqrt(2), sends +y -> +x
    bs_n=np.array([1.,-1.,0.])/np.sqrt(2.)
    rays={}; mask_all=np.ones(shape,dtype=bool)
    for arm in ('reference','test'):
        di=d if arm=='reference' else reflect(d,bs_n)
        if arm=='reference':
            m,tm=hit_plane(b,di,1,c.reference_mm*1e-3)
            normal=np.broadcast_to(np.array([0.,1.,0.]),p.shape)
            valid=np.ones(shape,dtype=bool)
            r2=m[...,0]**2+m[...,2]**2
        else:
            m,tm,normal,valid=hit_test(b,di,c)
            r2=m[...,1]**2+m[...,2]**2
        dr=reflect(di,normal)
        b2,tback=bs_intersection(m,dr)
        # Backwards propagation to illumination input (-x) from BS.
        # Reference return -y reflects into -x; test return -x transmits into -x.
        ds=reflect(dr,bs_n) if arm=='reference' else dr
        src,ts=hit_plane(b2,ds,0,-c.source_mm*1e-3)
        path=tb+tm+tback+ts
        # point source eikonal on source plane; plane source has constant phase
        if c.source=='point':
            extra=np.sqrt((src[...,1])**2+(src[...,2])**2+(c.source_mm*1e-3)**2)
            # source is located at x=-2S; source plane at x=-S
        else:
            extra=np.zeros(shape)
        path=path+extra
        good=valid&(tb>0)&(tm>0)&(tback>0)&(ts>0)&(r2<(c.aperture_mm*5e-4)**2)&np.isfinite(path)
        rays[arm]={'path':path,'valid':good,'mirror_xyz':m,'return_bs_xyz':b2,'source_xyz':src,
                   'segment_lengths':np.stack([tb,tm,tback,ts],axis=0)}
        mask_all &= good
    opd=rays['test']['path']-rays['reference']['path']
    phase=np.remainder(2*np.pi*opd/(c.wavelength_nm*1e-9)+np.pi,2*np.pi)-np.pi
    # Equal amplitudes, normalized incoherent offset 0.5 and modulation 0.45
    rng=np.random.default_rng(c.seed)
    shifts=np.arange(c.phase_steps)*np.pi/2
    ideal=np.stack([0.5+0.45*np.cos(phase+s) for s in shifts])
    frames=np.clip(ideal+rng.normal(0,c.noise_std,ideal.shape),0,1)
    frames[:,~mask_all]=0
    return {'frames':frames.astype(np.float32),'ideal_frames':ideal.astype(np.float32),
            'valid':mask_all,'opd_m':opd,'phase_rad':phase,'x_m':grid,'z_m':grid,
            'phase_shifts_rad':shifts,'reference':rays['reference'],'test':rays['test']}
