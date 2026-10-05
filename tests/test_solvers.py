import itertools
from math import sqrt

import numpy as np
import pytest
from pyomo.environ import Constraint, SolverFactory

from matopt.materials.atom import Atom
from matopt.materials.lattices.fcc_lattice import FCCLattice
from matopt.materials.canvas import Canvas
from matopt.aml.expr import SumSites
from matopt.aml.rule import PiecewiseLinear, EqualTo
from pyomo.common.errors import ApplicationError

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


def occupied(D):
    return frozenset(i for i, c in enumerate(D.Contents) if c is not None)


def ecoh(canv, sites, N):
    """Evaluate the model's objective for a set of occupied sites directly."""
    return sum(
        sqrt(sum(1 for j in canv.NeighborhoodIndexes[i] if j in sites))
        for i in sites
    ) / (sqrt(12) * N)


def brute_force_objectives(N):
    canv = make_cluster_model(N).canv
    return sorted(
        (ecoh(canv, set(occ), N) for occ in itertools.combinations(range(len(canv.Points)), N)),
        reverse=True,
    )


@pytest.mark.parametrize("solver", OPEN_SOURCE_SOLVERS)
def test_open_source_solver_finds_optimum(solver):
    best = brute_force_objectives(6)[0]
    m = make_cluster_model(N=6)
    D = m.maximize(m.Ecoh, tilim=60, solver=solver, tee=False, disp=0)
    assert D is not None
    assert n_atoms(D) == 6
    assert m._pyomo_m.obj() == pytest.approx(best, abs=1e-6)
    assert ecoh(m.canv, occupied(D), 6) == pytest.approx(best, abs=1e-6)


@pytest.mark.parametrize("solver", OPEN_SOURCE_SOLVERS)
def test_pyomo_interface_name_gets_solver_options(solver, monkeypatch):
    import matopt.aml.model as mm

    created = []
    real_factory = mm.SolverFactory

    def spy(name, *args, **kwargs):
        opt = real_factory(name, *args, **kwargs)
        created.append(opt)
        return opt

    monkeypatch.setattr(mm, "SolverFactory", spy)
    m = make_cluster_model(N=6)
    m.maximize(m.Ecoh, tilim=30, solver=_SOLVER_ALIASES[solver], tee=False, disp=0)
    opts = dict(created[-1].options)
    assert 30 in opts.values() or 30.0 in opts.values()
    assert 0.0 in opts.values()


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
def test_populate_returns_ranked_top_designs(solver):
    N = 5
    expected = brute_force_objectives(N)[:4]
    m = make_cluster_model(N)
    Ds = m.maximize(m.Ecoh, nSolns=4, tilim=60, solver=solver, tee=False, disp=0)
    assert len(Ds) == 4
    assert len({occupied(D) for D in Ds}) == 4
    objs = [ecoh(m.canv, occupied(D), N) for D in Ds]
    assert objs == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("solver", OPEN_SOURCE_SOLVERS)
def test_populate_stops_when_exhausted(solver):
    # 13 atoms on 13 sites: exactly one design exists
    m = make_cluster_model(N=13)
    Ds = m.maximize(m.Ecoh, nSolns=3, tilim=60, solver=solver, tee=False, disp=0)
    assert len(Ds) == 1


@pytest.mark.parametrize("solver", OPEN_SOURCE_SOLVERS)
def test_timeout_after_cut_does_not_return_stale_design(solver):
    from matopt.aml.rule import Disallow

    m = make_cluster_model(N=25, nShells=2)
    D1 = m.maximize(m.Ecoh, tilim=5, solver=solver, tee=False, disp=0)
    assert D1 is not None
    # Forbid D1 as populate does, then re-solve with no time for an incumbent.
    # The model still holds D1's values; they must not be returned again.
    m._pyomo_m.Cut = Constraint(expr=Disallow(D1)._pyomo_expr(m.Yi) >= 1)
    D2 = m._MatOptModel__solve_pyomo_model(False, 0, False, 0, None, solver)
    assert D2 is None or occupied(D2) != occupied(D1)


@pytest.mark.parametrize("solver", OPEN_SOURCE_SOLVERS)
def test_infeasible_returns_none(solver):
    # 1 shell of FCC has 13 sites, so 20 atoms cannot fit
    m = make_cluster_model(N=20)
    D = m.maximize(m.Ecoh, tilim=60, solver=solver, tee=False, disp=0)
    assert D is None


def test_unknown_solver_raises():
    m = make_cluster_model(N=6)
    with pytest.raises(ApplicationError, match="not_a_solver"):
        m.maximize(m.Ecoh, solver="not_a_solver", tee=False, disp=0)
