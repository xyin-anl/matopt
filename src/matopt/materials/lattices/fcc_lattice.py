from copy import deepcopy
from math import sqrt

import numpy as np

from matopt.materials.lattices.unit_cell_lattice import UnitCell, UnitCellLattice
from matopt.materials.geometry import Cube
from matopt.materials.tiling import CubicTiling
from matopt.materials.transform_func import ScaleFunc, RotateFunc
from matopt.materials.util import ListHasPoint


class FCCLattice(UnitCellLattice):
    RefIAD = sqrt(2) / 2

    # === STANDARD CONSTRUCTOR
    def __init__(self, IAD):
        RefUnitCellShape = Cube(1, BotBackLeftCorner=np.array([0, 0, 0], dtype=float))
        RefUnitCellTiling = CubicTiling(RefUnitCellShape)
        RefFracPositions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
        ]
        RefUnitCell = UnitCell(RefUnitCellTiling, RefFracPositions)
        UnitCellLattice.__init__(self, RefUnitCell)
        self._IAD = FCCLattice.RefIAD  # IAD is set correctly after calling applyTransF
        self.applyTransF(ScaleFunc(IAD / FCCLattice.RefIAD))
        self._NthNeighbors = [
            [
                np.array([0.0, -0.5, 0.5]),
                np.array([-0.5, -0.5, 0.0]),
                np.array([-0.5, 0.0, 0.5]),
                np.array([0.5, -0.5, 0.0]),
                np.array([0.0, -0.5, -0.5]),
                np.array([-0.5, 0.0, -0.5]),
                np.array([-0.5, 0.5, 0.0]),
                np.array([0.0, 0.5, 0.5]),
                np.array([0.5, 0.0, 0.5]),
                np.array([0.5, 0.0, -0.5]),
                np.array([0.0, 0.5, -0.5]),
                np.array([0.5, 0.5, 0.0]),
            ]
        ]

    # === CONSTRUCTOR - Aligned with FCC {100}
    @classmethod
    def alignedWith100(cls, IAD):
        return cls(IAD)  # Default implementation

    # === CONSTRUCTOR - Aligned with FCC {111}
    @classmethod
    def alignedWith111(cls, IAD, blnTrianglesAlignedWithX=True):
        result = cls(IAD)
        thetaX = -np.pi * 0.25
        thetaY = -np.arctan2(-sqrt(2), 2)
        thetaZ = np.pi * 0.5 if blnTrianglesAlignedWithX else 0
        result.applyTransF(RotateFunc.fromXYZAngles(thetaX, thetaY, thetaZ))
        return result

    # === MANIPULATION METHODS
    def applyTransF(self, TransF):
        if isinstance(TransF, ScaleFunc):
            if TransF.isIsometric:
                self._IAD *= TransF.Scale[0]
            else:
                raise ValueError("FCCLattice applyTransF: Can only scale isometrically")
        UnitCellLattice.applyTransF(self, TransF)

    # === PROPERTY EVALUATION METHODS
    # NOTE: inherited from UnitCellLattice
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
                    if not ListHasPoint(NList, N, 0.001 * FCCLattice.RefIAD):
                        tmp.append(N)
                        NList.append(N)
            self._NthNeighbors.append(tmp)

    def getLatticeVectors(self):
        """Get the three lattice vectors that define the unit cell.
        
        Returns:
            numpy.ndarray: 3x3 array where each row is a lattice vector
        """
        # FCC has a face-centered cubic unit cell
        # The lattice parameter a is related to IAD by: a = IAD * sqrt(2)
        a = self.IAD * sqrt(2)
        
        # For FCC, we can use either:
        # 1. Conventional cubic cell (easier to visualize but larger)
        # 2. Primitive cell (smaller but harder to visualize)
        
        # Using conventional cubic cell:
        return np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])

    # === BASIC QUERY METHODS
    @property
    def IAD(self):
        return self._IAD

    @property
    def FCC111LayerSpacing(self):
        return self.IAD * sqrt(2) / sqrt(3)

    @property
    def FCC100LayerSpacing(self):
        return self.IAD * sqrt(2) / 2

    @property
    def FCC110LayerSpacing(self):
        return self.IAD
