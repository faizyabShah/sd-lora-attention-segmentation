"""SG-threshold: label-free per-image threshold rules on stored 64x64 maps.

Each rule maps one min-max map y (and, for T5, the same image's R0 map) to a
threshold tau_I. tau_I is snapped to the registered 17-point grid
0.30..0.70, so the existing confusion counts score it exactly (raw and R2).
Rules with one parameter expose it on a 17-point axis that the unchanged
fold selector chooses in place of the global threshold. No rule sees GT,
the class or COCO.

* T0 global: slot j is grid threshold t_j (the registered rule).
* T1 Otsu: 256-bin between-class-variance maximum on [0, 1].
* T2 two-Gaussian: EM from the Otsu split; tau is the crossing of the
  weighted densities between the means (Otsu if none).
* T3 mean + k std: tau = mu + k sigma, k = -1.00..3.00 (step 0.25).
* T4 most stable: the interior grid threshold with the smallest area change
  A(t_{j-1}) - A(t_{j+1}).
* T5 R0 area: slot j matches the area of R0 >= t_j (quantile of y).
* T6 Otsu + offset: tau = tau_Otsu + d, d = -0.20..0.20 (step 0.025).
"""
from __future__ import annotations

import numpy as np

GRID=np.round(np.arange(17)*.025+.3,3)
K_AXIS=np.round(np.arange(17)*.25-1.,3)
D_AXIS=np.round(np.arange(17)*.025-.2,3)
RULES=("T0","T1","T2","T3","T4","T5","T6")
PARAMETER_FREE=("T1","T2","T4")
EDGE=.0125


def minmax(y):
    y=np.asarray(y,np.float64).ravel();lo,hi=y.min(),y.max()
    return (y-lo)/(hi-lo) if hi>lo else np.zeros_like(y)


def snap(tau):
    """Nearest grid index and whether tau lies beyond the grid by > half a step."""
    tau=float(tau)
    return int(np.abs(GRID-tau).argmin()),bool(tau<GRID[0]-EDGE or tau>GRID[-1]+EDGE)


def otsu(y,bins=256):
    hist,edges=np.histogram(y,bins=bins,range=(0.,1.))
    p=hist/max(hist.sum(),1);centres=(edges[:-1]+edges[1:])/2
    w0=np.cumsum(p);m=np.cumsum(p*centres);mt=m[-1]
    with np.errstate(divide="ignore",invalid="ignore"):
        between=(mt*w0-m)**2/(w0*(1-w0))
    between[~np.isfinite(between)]=-1
    return float(edges[int(between.argmax())+1])


def gmm_threshold(y,iterations=200,floor=1e-4):
    t=otsu(y);low,high=y[y<t],y[y>=t]
    if low.size<2 or high.size<2:return t
    pi=np.array([low.size,high.size],float)/y.size
    mu=np.array([low.mean(),high.mean()]);var=np.maximum([low.var(),high.var()],floor)
    for _ in range(iterations):
        dens=pi/np.sqrt(2*np.pi*var)*np.exp(-(y[:,None]-mu)**2/(2*var))
        r=dens/np.maximum(dens.sum(1,keepdims=True),1e-300)
        n=r.sum(0)
        if (n<1e-6).any():return t
        new_mu=(r*y[:,None]).sum(0)/n
        var=np.maximum((r*(y[:,None]-new_mu)**2).sum(0)/n,floor);pi=n/y.size
        done=np.abs(new_mu-mu).max()<1e-7;mu=new_mu
        if done:break
    order=np.argsort(mu);mu,var,pi=mu[order],var[order],pi[order]
    # log pi0 N(x;mu0,v0) = log pi1 N(x;mu1,v1) as a quadratic a x^2 + b x + c = 0.
    a=1/(2*var[1])-1/(2*var[0]);b=mu[0]/var[0]-mu[1]/var[1]
    c=(mu[1]**2/(2*var[1])-mu[0]**2/(2*var[0])
       +np.log(pi[0]/np.sqrt(var[0]))-np.log(pi[1]/np.sqrt(var[1])))
    roots=[-c/b] if abs(a)<1e-12 else [r.real for r in np.roots([a,b,c]) if abs(r.imag)<1e-9]
    inside=[r for r in roots if mu[0]<r<mu[1]]
    return float(min(inside,key=lambda r:abs(r-(mu[0]+mu[1])/2))) if inside else t


def stable_threshold(y):
    area=np.array([(y>=t).mean() for t in GRID])
    change=area[:-2]-area[2:]
    best=np.flatnonzero(change<=change.min()+1e-12)+1
    return float(GRID[min(best,key=lambda j:(abs(GRID[j]-.45),GRID[j]))])


def quantile_for_area(y,area):
    """Threshold whose mask keeps the top ``area`` fraction of y (all if area >= 1)."""
    if area>=1:return float(y.min())
    if area<=0:return float(y.max())+1.
    return float(np.quantile(y,1-area))


def rule_slots(rule,y,r0=None):
    """17 snapped grid indices and edge flags for one rule on one map."""
    y=minmax(y)
    if rule=="T0":taus=GRID
    elif rule=="T1":taus=[otsu(y)]*17
    elif rule=="T2":taus=[gmm_threshold(y)]*17
    elif rule=="T3":taus=y.mean()+K_AXIS*y.std()
    elif rule=="T4":taus=[stable_threshold(y)]*17
    elif rule=="T5":
        if r0 is None:raise ValueError("T5 needs the image's R0 map")
        r0=minmax(r0);taus=[quantile_for_area(y,(r0>=t).mean()) for t in GRID]
    elif rule=="T6":taus=otsu(y)+D_AXIS
    else:raise ValueError(rule)
    snapped=[snap(t) for t in taus]
    return np.array([s for s,_ in snapped]),np.array([e for _,e in snapped])
