import numpy as np
from copy import deepcopy

from matopt.materials.geometry import Parallelepiped
from matopt.materials.transform_func import ScaleFunc
from matopt.materials.lattices.unit_cell_lattice import UnitCell, UnitCellLattice
from matopt.materials.tiling import CubicTiling
from matopt.materials.util import ListHasPoint


class CubicLattice(UnitCellLattice):
    RefIAD = 1

    # === STANDARD CONSTRUCTOR
    def __init__(self, IAD):
        RefUnitCellShape = Parallelepiped.fromEdgesAndAngles(
            CubicLattice.RefIAD,
            CubicLattice.RefIAD,
            CubicLattice.RefIAD,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.array([0, 0, 0], dtype=float),
        )
        RefUnitCellTiling = CubicTiling(RefUnitCellShape)
        RefFracPositions = [np.array([0.0, 0.0, 0.0])]
        RefUnitCell = UnitCell(RefUnitCellTiling, RefFracPositions)
        UnitCellLattice.__init__(self, RefUnitCell)
        self._IAD = CubicLattice.RefIAD
        self.applyTransF(ScaleFunc(IAD / CubicLattice.RefIAD))
        self._NthNeighbors = [
            [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, -1.0]),
            ]
        ]

    # === MANIPULATION METHODS
    def applyTransF(self, TransF):
        if isinstance(TransF, ScaleFunc):
            if TransF.isIsometric:
                self._IAD *= TransF.Scale[0]
            else:
                raise ValueError(
                    "CubicLattice applyTransF: Can only scale isometrically"
                )
        UnitCellLattice.applyTransF(self, TransF)

    # === PROPERTY EVALUATION METHODS
    # NOTE: This method is inherited from UnitCellLattice
    # def isOnLattice(self,P):

    def areNeighbors(self, P1, P2):
        return np.linalg.norm(P2 - P1) <= self.IAD

    def getNeighbors(self, P, layer=1):
        RefP = self._getConvertToReference(P)
        if layer > len(self._NthNeighbors):
            self._calculateNeighbors(layer)
        NBs = deepcopy(self._NthNeighbors[layer - 1])
        for NeighP in NBs:
            NeighP += RefP
            self._convertFromReference(NeighP)
        return NBs

    def _calculateNeighbors(self, layer):
        NList = [np.array([0, 0, 0], dtype=float)]
        for nb in self._NthNeighbors:
            NList.extend(nb)
        for _ in range(layer - len(self._NthNeighbors)):
            tmp = []
            for P in self._NthNeighbors[len(self._NthNeighbors) - 1]:
                for Q in self._NthNeighbors[0]:
                    N = P + Q
                    if not ListHasPoint(NList, N, 0.001 * CubicLattice.RefIAD):
                        tmp.append(N)
                        NList.append(N)
            self._NthNeighbors.append(tmp)

    def getLatticeVectors(self):
        """Get the three lattice vectors that define the unit cell.
        
        Returns:
            numpy.ndarray: 3x3 array where each row is a lattice vector
        """
        # Simple cubic lattice has cubic unit cell
        # The lattice parameter a equals IAD
        a = self.IAD
        
        return np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])

    # === BASIC QUERY METHODS
    @property
    def IAD(self):
        return self._IAD
