import unittest
import numpy as np
from pymatgen.core.structure import Structure, Lattice
from pymatgen.util.testing import PymatgenTest
from matopt.materials.atom import Atom
from matopt.materials.canvas import Canvas
from matopt.materials.design import Design
from matopt.materials.tiling import LinearTiling, PlanarTiling, CubicTiling


class TestPymatgenIntegration(PymatgenTest):
    def setUp(self):
        """Set up test structures with different lattices and symmetries,
        expanded to larger supercells to ensure atoms have neighbors."""
        # Expand factor
        supercell_size = (2, 2, 2)

        # Simple cubic structure with 8 atoms, expanded to a larger supercell
        self.cubic = (
            Structure.from_spacegroup(
                "Pm-3m",
                Lattice.cubic(4.0),
                ["Fe"] * 3 + ["Cu"] * 3 + ["Fe"] * 2,
                [
                    [0, 0, 0],
                    [0.5, 0, 0],
                    [0, 0.5, 0],
                    [0, 0, 0.5],
                    [0.5, 0.5, 0],
                    [0.5, 0, 0.5],
                    [0, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                ],
            )
            * supercell_size
        )

        # Face-centered cubic structure with 4 atoms, expanded to a larger supercell
        self.fcc = (
            Structure.from_spacegroup(
                "Fm-3m",
                Lattice.cubic(4.0),
                ["Cu"] * 2 + ["Fe"] * 2,
                [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            )
            * supercell_size
        )

        # Body-centered cubic structure with 2 atoms, expanded to a larger supercell
        self.bcc = (
            Structure.from_spacegroup(
                "Im-3m", Lattice.cubic(4.0), ["Fe"] * 2, [[0, 0, 0], [0.5, 0.5, 0.5]]
            )
            * supercell_size
        )

        # Hexagonal close-packed structure with 2 atoms, expanded to a larger supercell
        self.hcp = (
            Structure.from_spacegroup(
                "P6_3/mmc",
                Lattice.hexagonal(3.0, 4.0),
                ["Ti", "Nb"],
                [[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]],
            )
            * supercell_size
        )

        # Complex oxide structure (perovskite), expanded to a larger supercell
        self.perovskite = self.get_structure("SrTiO3") * supercell_size

        # Create corresponding matopt designs
        self.designs = {
            "cubic": Design.from_pymatgen(self.cubic),
            "fcc": Design.from_pymatgen(self.fcc),
            "bcc": Design.from_pymatgen(self.bcc),
            "hcp": Design.from_pymatgen(self.hcp),
            "perovskite": Design.from_pymatgen(self.perovskite),
        }

    def test_structure_to_design_basic_properties(self):
        """Test basic property preservation during Structure -> Design conversion"""
        for name, structure in {
            "cubic": self.cubic,
            "fcc": self.fcc,
            "bcc": self.bcc,
            "hcp": self.hcp,
            "perovskite": self.perovskite,
        }.items():
            design = self.designs[name]

            # Check number of sites
            self.assertEqual(len(design), len(structure))

            # Check atomic species
            for i, site in enumerate(structure):
                self.assertEqual(design.Contents[i].Symbol, str(site.specie.symbol))

            # Check lattice parameters
            if hasattr(design.Canvas, "_lattice"):
                np.testing.assert_array_almost_equal(
                    (
                        design.Canvas._lattice.matrix
                        if isinstance(design.Canvas._lattice, Lattice)
                        else design.Canvas._lattice.getLatticeVectors()
                    ),
                    structure.lattice.matrix,
                )

    def test_design_to_structure_conversion(self):
        """Test property preservation during Design -> Structure conversion"""
        for name, design in self.designs.items():
            structure = design.to_pymatgen()
            original = getattr(self, name)

            # Check basic properties
            self.assertEqual(len(structure), len(original))
            self.assertEqual(structure.composition, original.composition)

            # Check lattice parameters
            np.testing.assert_array_almost_equal(
                structure.lattice.matrix, original.lattice.matrix
            )

    def test_periodic_boundary_conditions(self):
        """Test handling of different periodic boundary conditions"""
        test_cases = [
            (
                lambda: LinearTiling.fromLatticeVectors([[0, 0, 1]]),
                [False, False, True],
            ),
            (
                lambda: PlanarTiling.fromLatticeVectors([[1, 0, 0], [0, 1, 0]]),
                [True, True, False],
            ),
            (
                lambda: CubicTiling.fromLatticeVectors(
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                ),
                [True, True, True],
            ),
        ]

        for tiling_factory, expected_pbc in test_cases:
            # Create design with specific tiling
            canvas = Canvas(Points=[[0, 0, 0]], DefaultNN=12)
            canvas._tiling = tiling_factory()
            design = Design(canvas, [Atom("Fe")])

            # Convert to structure and check PBC
            structure = design.to_pymatgen()
            self.assertEqual(list(structure.lattice.pbc), expected_pbc)

    def test_supercell_conversion(self):
        """Test conversion of supercell structures"""
        for structure in [self.cubic, self.fcc, self.bcc]:
            # Create supercell
            supercell = structure * (1, 2, 1)

            # Convert to design and back
            design = Design.from_pymatgen(supercell)
            structure_new = design.to_pymatgen()

            # Check properties
            self.assertEqual(len(structure_new), len(supercell))
            self.assertEqual(structure_new.composition, supercell.composition)
            np.testing.assert_array_almost_equal(
                structure_new.lattice.matrix, supercell.lattice.matrix
            )


if __name__ == "__main__":
    unittest.main()
