from copy import deepcopy
from math import sqrt

import numpy as np

from matopt.materials.lattices.unit_cell_lattice import UnitCell, UnitCellLattice
from matopt.materials.geometry import Cube
from matopt.materials.tiling import CubicTiling
from matopt.materials.transform_func import ScaleFunc, RotateFunc
from matopt.materials.util import ListHasPoint


class BCCLattice(UnitCellLattice):
    RefIAD = sqrt(3) / 2  # Reference interatomic distance

    # === STANDARD CONSTRUCTOR
    def __init__(self, IAD):
        RefUnitCellShape = Cube(1, BotBackLeftCorner=np.array([0, 0, 0], dtype=float))
        RefUnitCellTiling = CubicTiling(RefUnitCellShape)
        RefFracPositions = [
            np.array([0.0, 0.0, 0.0]),  # Corner atom
            np.array([0.5, 0.5, 0.5]),  # Center atom
        ]
        RefUnitCell = UnitCell(RefUnitCellTiling, RefFracPositions)
        UnitCellLattice.__init__(self, RefUnitCell)
        self._IAD = BCCLattice.RefIAD
        self.applyTransF(ScaleFunc(IAD / BCCLattice.RefIAD))
        self._NthNeighbors = [
            [
                np.array([0.5, 0.5, 0.5]),
                np.array([0.5, 0.5, -0.5]),
                np.array([0.5, -0.5, 0.5]),
                np.array([-0.5, 0.5, 0.5]),
                np.array([0.5, -0.5, -0.5]),
                np.array([-0.5, 0.5, -0.5]),
                np.array([-0.5, -0.5, 0.5]),
                np.array([-0.5, -0.5, -0.5]),
            ]
        ]

    # === MANIPULATION METHODS
    def applyTransF(self, TransF):
        if isinstance(TransF, ScaleFunc):
            if TransF.isIsometric:
                self._IAD *= TransF.Scale[0]
            else:
                raise ValueError("BCCLattice applyTransF: Can only scale isometrically")
        UnitCellLattice.applyTransF(self, TransF)

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
                    if not ListHasPoint(NList, N, 0.001 * BCCLattice.RefIAD):
                        tmp.append(N)
                        NList.append(N)
            self._NthNeighbors.append(tmp)

    def getLatticeVectors(self):
        """Get the three lattice vectors that define the unit cell.

        Returns:
            numpy.ndarray: 3x3 array where each row is a lattice vector
        """
        # BCC has a body-centered cubic unit cell
        # The lattice parameter a is related to IAD by: a = IAD * 2/sqrt(3)
        a = self.IAD * 2 / sqrt(3)

        return np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])

    # === BASIC QUERY METHODS
    @property
    def IAD(self):
        return self._IAD

    @classmethod
    def alignedWith100(cls, IAD):
        """Create a BCC lattice with [100] aligned with x-axis."""
        return cls(IAD)  # Default implementation

    @classmethod
    def alignedWith110(cls, IAD):
        """Create a BCC lattice with [110] aligned with x-axis."""
        result = cls(IAD)
        thetaY = np.pi * 0.25  # 45° rotation around y-axis
        result.applyTransF(RotateFunc.fromXYZAngles(0, thetaY, 0))
        return result

    @classmethod
    def alignedWith111(cls, IAD):
        """Create a BCC lattice with [111] aligned with x-axis."""
        result = cls(IAD)
        thetaX = -np.pi * 0.25
        thetaY = -np.arctan2(-sqrt(2), 2)
        result.applyTransF(RotateFunc.fromXYZAngles(thetaX, thetaY, 0))
        return result

    @classmethod
    def alignedWith(cls, IAD, MI):
        """Create a BCC lattice aligned with specified Miller indices.

        Args:
            IAD: Interatomic distance
            MI: Miller indices as string ("100", "110", or "111")
        """
        if MI == "100":
            return cls.alignedWith100(IAD)
        elif MI == "110":
            return cls.alignedWith110(IAD)
        elif MI == "111":
            return cls.alignedWith111(IAD)
        else:
            raise ValueError(
                "BCCLattice.alignedWith: Input direction must be '100', '110', or '111'"
            )

    @property
    def BCC100LayerSpacing(self):
        """Get the spacing between (100) planes."""
        return self.IAD * 2 / sqrt(3)  # equals lattice parameter 'a'

    @property
    def BCC110LayerSpacing(self):
        """Get the spacing between (110) planes."""
        return self.IAD / sqrt(2)

    @property
    def BCC111LayerSpacing(self):
        """Get the spacing between (111) planes."""
        return self.IAD * 2 / (3 * sqrt(2))
