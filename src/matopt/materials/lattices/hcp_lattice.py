from copy import deepcopy
from math import sqrt

import numpy as np

from matopt.materials.lattices.unit_cell_lattice import UnitCell, UnitCellLattice
from matopt.materials.geometry import Parallelepiped
from matopt.materials.tiling import CubicTiling
from matopt.materials.transform_func import ScaleFunc, RotateFunc
from matopt.materials.util import ListHasPoint


class HCPLattice(UnitCellLattice):
    RefIAD = 1
    IdealCA = sqrt(8 / 3)  # Ideal c/a ratio for HCP

    # === STANDARD CONSTRUCTOR
    def __init__(self, IAD, c_a_ratio=None):
        if c_a_ratio is None:
            c_a_ratio = HCPLattice.IdealCA

        # HCP has hexagonal symmetry
        a = IAD

        RefUnitCellShape = Parallelepiped(
            np.array([1, 0, 0], dtype=float),
            np.array([-0.5, sqrt(3) / 2, 0], dtype=float),
            np.array([0, 0, c_a_ratio], dtype=float),
            BotBackLeftCorner=np.array([0, 0, 0], dtype=float),
        )
        RefUnitCellTiling = CubicTiling(RefUnitCellShape)
        RefFracPositions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1 / 3, 1 / 3, 0.5]),
            np.array([2 / 3, 2 / 3, 0.0]),
            np.array([0.0, 0.0, 0.5]),
        ]
        RefUnitCell = UnitCell(RefUnitCellTiling, RefFracPositions)
        UnitCellLattice.__init__(self, RefUnitCell)
        self._IAD = HCPLattice.RefIAD
        self.applyTransF(ScaleFunc(IAD / HCPLattice.RefIAD))

        # First nearest neighbors
        self._NthNeighbors = [
            [
                np.array([1 / 3, 1 / 3, 0.5]),
                np.array([-1 / 3, 1 / 3, 0.5]),
                np.array([1 / 3, -1 / 3, 0.5]),
                np.array([1 / 3, 1 / 3, -0.5]),
                np.array([-1 / 3, 1 / 3, -0.5]),
                np.array([1 / 3, -1 / 3, -0.5]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.5, sqrt(3) / 2, 0.0]),
                np.array([-0.5, sqrt(3) / 2, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([-0.5, -sqrt(3) / 2, 0.0]),
                np.array([0.5, -sqrt(3) / 2, 0.0]),
            ]
        ]

    def applyTransF(self, TransF):
        if isinstance(TransF, ScaleFunc):
            if TransF.isIsometric:
                self._IAD *= TransF.Scale[0]
            else:
                raise ValueError("HCPLattice applyTransF: Can only scale isometrically")
        UnitCellLattice.applyTransF(self, TransF)

    def areNeighbors(self, P1, P2):
        return np.linalg.norm(P2 - P1) <= self.IAD + 1e-10

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
                    if not ListHasPoint(NList, N, 0.001 * HCPLattice.RefIAD):
                        tmp.append(N)
                        NList.append(N)
            self._NthNeighbors.append(tmp)

    def getLatticeVectors(self):
        """Get the three lattice vectors that define the unit cell.

        Returns:
            numpy.ndarray: 3x3 array where each row is a lattice vector
        """
        a = self.IAD
        c = a * HCPLattice.IdealCA

        return np.array([[a, 0, 0], [-a / 2, a * sqrt(3) / 2, 0], [0, 0, c]])

    @property
    def IAD(self):
        return self._IAD

    @classmethod
    def alignedWith(cls, IAD, MI):
        """Create an HCP lattice aligned with specified Miller indices.

        Args:
            IAD: Interatomic distance
            MI: Miller indices as string ("0001", "1120", or "1010")

        Returns:
            HCPLattice: Aligned lattice instance
        """
        if MI == "0001":
            return cls(IAD)  # Default orientation
        elif MI == "1120":  # a-direction
            result = cls(IAD)
            thetaY = np.pi / 2  # 90° rotation around y-axis
            result.applyTransF(RotateFunc.fromXYZAngles(0, thetaY, 0))
            return result
        elif MI == "1010":  # m-direction
            result = cls(IAD)
            thetaZ = np.pi / 6  # 30° rotation around z-axis
            thetaY = np.pi / 2  # 90° rotation around y-axis
            result.applyTransF(RotateFunc.fromXYZAngles(0, thetaY, thetaZ))
            return result
        else:
            raise ValueError(
                "HCPLattice.alignedWith: Input direction must be '0001', '1120', or '1010'"
            )

    @classmethod
    def alignedWith0001(cls, IAD):
        """Create an HCP lattice with c-axis [0001] aligned with z-axis."""
        return cls(IAD)

    @classmethod
    def alignedWith1120(cls, IAD):
        """Create an HCP lattice with a-direction [11-20] aligned with x-axis."""
        return cls.alignedWith(IAD, "1120")

    @classmethod
    def alignedWith1010(cls, IAD):
        """Create an HCP lattice with m-direction [10-10] aligned with x-axis."""
        return cls.alignedWith(IAD, "1010")

    @property
    def HCP0001LayerSpacing(self):
        """Get the spacing between (0001) basal planes."""
        return self.IAD * self.IdealCA * sqrt(3) / 2

    @property
    def HCP1120LayerSpacing(self):
        """Get the spacing between (11-20) planes."""
        return self.IAD * sqrt(3) / 2

    @property
    def HCP1010LayerSpacing(self):
        """Get the spacing between (10-10) planes."""
        return self.IAD * 3 / 2
