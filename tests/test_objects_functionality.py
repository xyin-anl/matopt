import numpy as np
from math import sqrt
from matopt.materials.util import areEqual
from tests.test_objects_construction import *


def test_functionality_FCCLattice():
    lattice = test_construct_FCCLattice()
    assert lattice.areNeighbors(np.zeros(3, dtype=float), np.array([0.0, -0.5, 0.5]))
    nbs = lattice.getNeighbors(np.zeros(3, dtype=float))
    for p in nbs:
        assert lattice.areNeighbors(np.zeros(3, dtype=float), p)
    assert areEqual(lattice.IAD, sqrt(2) / 2, 1e-4)
    assert areEqual(lattice.FCC111LayerSpacing, 1 / sqrt(3), 1e-4)
    assert areEqual(lattice.FCC100LayerSpacing, 1 / 2, 1e-4)
    assert areEqual(lattice.FCC110LayerSpacing, sqrt(2) / 2, 1e-4)


def test_functionality_CubicLattice():
    lattice = test_construct_CubicLattice()
    assert lattice.areNeighbors(np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0]))
    nbs = lattice.getNeighbors(np.zeros(3, dtype=float))
    for p in nbs:
        assert lattice.areNeighbors(np.zeros(3, dtype=float), p)
    assert areEqual(lattice.IAD, 1.0, 1e-4)


def test_functionality_DiamondLattice():
    lattice = test_construct_DiamondLattice()
    assert lattice.areNeighbors(np.zeros(3, dtype=float), np.array([0.25, 0.25, 0.25]))
    nbs = lattice.getNeighbors(np.zeros(3, dtype=float))
    for p in nbs:
        assert lattice.areNeighbors(np.zeros(3, dtype=float), p)
    assert lattice.isASite(np.zeros(3, dtype=float))
    assert lattice.isBSite(np.array([0.25, 0.25, 0.25]))
    assert areEqual(lattice.IAD, sqrt(3) / 4, 1e-4)
    assert areEqual(lattice.getLayerSpacing("100"), 1 / 4, 1e-4)
    assert areEqual(lattice.getLayerSpacing("110"), sqrt(2) / 4, 1e-4)
    assert areEqual(lattice.getLayerSpacing("111"), sqrt(3) / 3, 1e-4)
    assert areEqual(lattice.getLayerSpacing("112"), sqrt(6) / 12, 1e-4)
    assert areEqual(lattice.getShellSpacing("100"), sqrt(8) / 4, 1e-4)
    assert areEqual(lattice.getShellSpacing("112"), sqrt(2) / 4, 1e-4)
    assert areEqual(lattice.getUniqueLayerCount("100"), 4, 1e-4)
    assert areEqual(lattice.getUniqueLayerCount("110"), 2, 1e-4)
    assert areEqual(lattice.getUniqueLayerCount("111"), 3, 1e-4)
    assert areEqual(lattice.getUniqueLayerCount("112"), 6, 1e-4)


def test_functionality_WurtziteLattice():
    lattice = test_construct_WurtziteLattice()
    assert lattice.areNeighbors(
        np.zeros(3, dtype=float), np.array([0.0, 0.0, -3 / sqrt(24)])
    )
    nbs = lattice.getNeighbors(np.zeros(3, dtype=float))
    for p in nbs:
        assert lattice.areNeighbors(np.zeros(3, dtype=float), p)
    assert lattice.isASite(np.zeros(3, dtype=float))
    assert lattice.isBSite(np.array([0.0, 0.0, -3 / sqrt(24)]))
    assert areEqual(lattice.IAD, sqrt(3 / 8), 1e-4)
    assert areEqual(lattice.getLayerSpacing("0001"), sqrt(2 / 3), 1e-4)
    assert areEqual(lattice.getLayerSpacing("1100"), sqrt(3) / 2, 1e-4)
    assert areEqual(lattice.getLayerSpacing("1120"), 1, 1e-4)
    assert areEqual(lattice.getShellSpacing("0001"), sqrt(3) / 2, 1e-4)
    assert areEqual(lattice.getShellSpacing("1100"), 1, 1e-4)
    assert areEqual(lattice.getShellSpacing("1120"), 1, 1e-4)
    assert areEqual(lattice.getUniqueLayerCount("0001"), 2, 1e-4)
    assert areEqual(lattice.getUniqueLayerCount("1100"), 2, 1e-4)
    assert areEqual(lattice.getUniqueLayerCount("1120"), 1, 1e-4)
