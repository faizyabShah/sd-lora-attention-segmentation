"""Independent frozen formulas, region changes, inherited-threshold CV and resume tests."""
from pathlib import Path
import sys
import tempfile
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.segmentation import locked_refinements as lr
from src.segmentation import controlled_cv as cv
from src.segmentation.experiment_records import NumericSidecar
from scripts.select_propagation_cv import run_screen


def test_frozen_selection_trim_and_gate():
    spec=json.loads((ROOT/'configs/sg12_crf_sg14_sg15.json').read_text())
    gen=torch.Generator().manual_seed(77)
    obs=torch.rand(248,16,generator=gen);c=torch.rand(16,generator=gen)
    maps,diag=lr.selected_consensus_maps(obs,c,spec)
    unit=obs/obs.norm(dim=1,keepdim=True)
    matrix=unit@unit.T
    r=torch.stack([torch.cat((matrix[i,:i],matrix[i,i+1:])).mean() for i in range(248)])
    rank=torch.argsort(r,descending=True,stable=True)[:124]
    assert torch.equal(rank,diag['selected_indices'])
    selected=obs[rank];order=selected.sort(0).values
    trim=order[24:100].mean(0) # floor(.2*124)=24 removed FROM EACH tail
    agreement=torch.maximum((selected>c).float().mean(0),(selected<c).float().mean(0))
    expected=torch.where(agreement>=.8,trim,c)
    torch.testing.assert_close(maps[lr.SG14],expected)
    torch.testing.assert_close(maps[lr.SG12],selected.mean(0))
    equal=torch.ones(248,4);maps,diag=lr.selected_consensus_maps(equal,torch.ones(4),spec)
    assert diag['selected_indices'].tolist()==list(range(124))
    assert torch.equal(maps[lr.SG14],torch.ones(4)) # strict up/down, equality abstains


def test_streamed_affinities_and_two_pass():
    g=torch.Generator().manual_seed(33)
    calls=[(torch.randn(2,4,3,generator=g),torch.randn(2,4,3,generator=g)) for _ in range(3)]
    c=torch.rand(4,generator=g)
    obs,one,two,graph=lr.collect_observations(calls,c,need_two=True,need_graph=True)
    a=torch.cat([(q@k.transpose(-1,-2)/3**.5).softmax(-1) for q,k in calls]);z=a@c
    torch.testing.assert_close(obs,z);torch.testing.assert_close(one,z.mean(0))
    torch.testing.assert_close(two,(a@z[...,None])[...,0].mean(0));torch.testing.assert_close(graph,a.mean(0))


def test_region_occupancy_snapping_and_change_accounting():
    labels=np.tile(np.arange(4),(4,1));prior=np.array([[1,1,1,0],[0,1,1,0],[0,0,1,1],[0,0,0,1]],bool)
    b,f,a=lr.refine_regions(labels,prior,'B')
    np.testing.assert_allclose(f[:4],[.25,.5,.75,.5])
    assert not b[:,0].any() and b[:,2].all()
    np.testing.assert_array_equal(b[:,1],prior[:,1]);np.testing.assert_array_equal(b[:,3],prior[:,3])
    truth=labels<2
    for kind in ('A025','A050','A075','B'):
        after,_,_=lr.refine_regions(labels,prior,kind)
        change=lr.changes(prior,after,truth);before=lr.mask_counts(prior,truth);new=lr.mask_counts(after,truth)
        np.testing.assert_array_equal(new-before,[change[0]-change[2],change[3]-change[1],change[2]-change[0],change[1]-change[3]])
        assert change.sum()==np.count_nonzero(prior!=after)
    # Native regions expanded without interpolation of binary boundaries.
    assert lr.refine_regions(labels,prior,'A050')[0][:,1].all()


def test_inherited_threshold_cv_has_no_retuning():
    # Prior is best at .3, candidate is best at .45. A refinement MUST inherit .3.
    prior=np.tile([[8,2,2,8],[5,5,5,5]],(5,1,1))
    refined=np.tile([[7,3,3,7],[10,0,0,10]],(5,1,1))
    results=cv.Results(('p','r'),(.3,.45),tuple(map(str,range(5))),('class',)*5,np.arange(5),np.stack((prior,refined)),np.zeros((2,5)),{})
    default=run_screen(results,'p',['r'],{})
    explicit=run_screen(results,'p',['r'],{},threshold_sources=None)
    assert default==explicit and default['frozen']=='r'
    inherited=run_screen(results,'p',['r'],{},threshold_sources={'r':'p'})
    assert inherited['frozen']=='p'
    assert all(r['chosen_threshold']==.3 for r in inherited['folds'])


def test_atomic_sidecar_resume():
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'diag.npz';w=NumericSidecar(path,{'stage':'a'});w.add('x',{'values':[1,2]})
        w=NumericSidecar(path,{'stage':'a'});w.add('x',{'values':[3,4]})
        assert w.records['x']['values'].tolist()==[3,4]
        try:NumericSidecar(path,{'stage':'b'})
        except ValueError:pass
        else:raise AssertionError('changed settings accepted')

if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):print(name,flush=True);fn()
    print('Locked refinement checks passed',flush=True)
