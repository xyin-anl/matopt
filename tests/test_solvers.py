from math import sqrt

import numpy as np
import pytest
from pyomo.environ import SolverFactory

from matopt.materials.atom import Atom
from matopt.materials.lattices.fcc_lattice import FCCLattice
from matopt.materials.canvas import Canvas
from matopt.aml.expr import SumSites
from matopt.aml.rule import PiecewiseLinear, EqualTo
from matopt.aml.model import MatOptModel, _SOLVER_ALIASES

OPEN_SOURCE_SOLVERS = [
    s
    for s in ("highs", "scip")
    if SolverFactory(_SOLVER_ALIASES[s]).available(exception_flag=False)
]
pytestmark = pytest.mark.skipif(
    not OPEN_SOURCE_SOLVERS, reason="no open-source MILP solver installed"
)


def make_cluster_model(N, nShells=1):
    Lat = FCCLattice(IAD=2.7704443686888935)
    Canv = Canvas()
    Canv.addLocation(np.array([0, 0, 0], dtype=float))
    Canv.addShells(nShells, Lat.getNeighbors)
    Canv.setNeighborsFromFunc(Lat.getNeighbors)
    m = MatOptModel(Canv, [Atom("Pt")])
    m.addSitesDescriptor(
        "CNRi",
        bounds=(0, sqrt(12)),
        integer=False,
        rules=PiecewiseLinear(
            values=[sqrt(CN) for CN in range(13)],
            breakpoints=[CN for CN in range(13)],
            input_desc=m.Ci,
            con_type="UB",
        ),
    )
    m.addGlobalDescriptor(
        "Ecoh", rules=EqualTo(SumSites(desc=m.CNRi, coefs=(1.0 / sqrt(12) / N)))
    )
    m.addGlobalDescriptor("Size", bounds=(N, N), rules=EqualTo(SumSites(desc=m.Yi)))
    return m


def n_atoms(D):
    return sum(1 for c in D.Contents if c is not None)


@pytest.mark.parametrize("solver", OPEN_SOURCE_SOLVERS)
def test_open_source_solver_finds_optimum(solver):
    m = make_cluster_model(N=6)
    D = m.maximize(m.Ecoh, tilim=60, solver=solver, tee=False, disp=0)
    assert D is not None
    assert n_atoms(D) == 6


def test_solvers_agree_on_objective():
    if len(OPEN_SOURCE_SOLVERS) < 2:
        pytest.skip("need two solvers to compare")
    objs = []
    for solver in OPEN_SOURCE_SOLVERS:
        m = make_cluster_model(N=6)
        m.maximize(m.Ecoh, tilim=60, solver=solver, tee=False, disp=0)
        objs.append(m._pyomo_m.obj())
    assert objs[0] == pytest.approx(objs[1], abs=1e-6)


def test_auto_solver():
    m = make_cluster_model(N=6)
    D = m.maximize(m.Ecoh, tilim=60, tee=False, disp=0)
    assert D is not None


@pytest.mark.parametrize("solver", OPEN_SOURCE_SOLVERS)
def test_populate_returns_distinct_ranked_designs(solver):
    m = make_cluster_model(N=6)
    Ds = m.maximize(m.Ecoh, nSolns=3, tilim=60, solver=solver, tee=False, disp=0)
    assert len(Ds) == 3
    occupied = [tuple(c is not None for c in D.Contents) for D in Ds]
    assert len(set(occupied)) == 3


@pytest.mark.parametrize("solver", OPEN_SOURCE_SOLVERS)
def test_infeasible_returns_none(solver):
    # 1 shell of FCC has 13 sites, so 20 atoms cannot fit
    m = make_cluster_model(N=20)
    D = m.maximize(m.Ecoh, tilim=60, solver=solver, tee=False, disp=0)
    assert D is None


def test_unknown_solver_raises():
    m = make_cluster_model(N=6)
    with pytest.raises(Exception):
        m.maximize(m.Ecoh, solver="not_a_solver", tee=False, disp=0)
