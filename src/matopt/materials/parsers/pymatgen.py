from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.periodic_table import Element
import numpy as np


def design_to_structure(design):
    """Convert a matopt Design object to a pymatgen Structure.

    Args:
        design (Design): matopt Design object to convert

    Returns:
        Structure: Equivalent pymatgen Structure object
    """
    # Import here to break circular dependency
    from matopt.materials.atom import Atom
    from matopt.materials.geometry import RectPrism
    from matopt.materials.tiling import LinearTiling, PlanarTiling, CubicTiling

    # Get species and coords for non-void sites
    species = []
    coords = []
    for i, content in enumerate(design.Contents):
        if content is not None and content != Atom():
            species.append(Element(content.Symbol))
            coords.append(design.Canvas.Points[i])

    # Get lattice vectors from the lattice object if available
    if hasattr(design.Canvas, "_lattice"):
        if isinstance(design.Canvas._lattice, Lattice):
            lattice_vectors = design.Canvas._lattice.matrix
        else:
            raise ValueError("Lattice object is not a pymatgen Lattice")
    else:
        try:
            # Fallback to bounding box if no lattice object
            bbox = RectPrism.fromPointsBBox(design.Canvas.Points)
            lattice_vectors = np.array([bbox.Vx, bbox.Vy, bbox.Vz])
        except ValueError:
            # Default to unit cell if everything else fails
            lattice_vectors = np.eye(3)

    # Verify lattice vectors shape
    if lattice_vectors.shape != (3, 3):
        raise ValueError(
            f"Invalid lattice vectors shape: {lattice_vectors.shape}. Expected (3, 3)"
        )

    # Set up periodicity based on tiling type
    if hasattr(design.Canvas, "_tiling"):
        if isinstance(design.Canvas._tiling, LinearTiling):
            pbc = [False, False, True]
        elif isinstance(design.Canvas._tiling, PlanarTiling):
            pbc = [True, True, False]
        elif isinstance(design.Canvas._tiling, CubicTiling):
            pbc = [True, True, True]
        else:
            pbc = [False, False, False]
    else:
        pbc = [False, False, False]

    lattice = Lattice(lattice_vectors, pbc=pbc)

    # Create pymatgen Structure
    return Structure(
        lattice=lattice, species=species, coords=coords, coords_are_cartesian=True
    )


def structure_to_design(structure):
    """Convert a pymatgen Structure to a matopt Design.

    Args:
        structure (Structure): pymatgen Structure object to convert

    Returns:
        Design: Equivalent matopt Design object with preserved structural information
    """
    # Import here to break circular dependency
    from matopt.materials.tiling import LinearTiling, PlanarTiling, CubicTiling
    from matopt.materials.canvas import Canvas
    from matopt.materials.design import Design
    from matopt.materials.atom import Atom

    # Extract cartesian coordinates and lattice info
    coords = structure.cart_coords
    lattice_vectors = structure.lattice.matrix
    pbc = (
        structure.lattice.pbc
        if hasattr(structure.lattice, "pbc")
        else [True, True, True]
    )

    # Get all neighbors within a reasonable cutoff, including periodic images
    cutoff = estimate_neighbor_cutoff(structure)
    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)

    # Create Canvas with points and empty neighborhood indices
    canvas = Canvas(Points=coords.tolist())
    canvas._NeighborhoodIndexes = [[] for _ in range(len(coords))]
    canvas._lattice = Lattice(lattice_vectors, pbc=pbc)

    # Set up appropriate tiling based on periodicity
    if any(pbc):
        if all(pbc):
            tiling = CubicTiling.fromLatticeVectors(lattice_vectors)
        elif pbc[0] and pbc[1] and not pbc[2]:
            tiling = PlanarTiling.fromLatticeVectors(lattice_vectors[:2])
        elif not pbc[0] and not pbc[1] and pbc[2]:
            tiling = LinearTiling.fromLatticeVector(lattice_vectors[2])

        # Store tiling in canvas
        canvas._tiling = tiling

        # Process neighbors considering periodicity
        for i, site_neighbors in enumerate(all_neighbors):
            neighbor_indices = []
            # Sort neighbors by distance
            for neigh_tuple in sorted(site_neighbors, key=lambda x: x[1]):
                neigh = neigh_tuple[0]
                # Get periodic image of neighbor point
                neigh_coords = neigh.coords

                # Try to find the point in canvas, considering periodic images
                found_idx = None
                if canvas.hasPoint(neigh_coords):
                    found_idx = canvas.getPointIndex(neigh_coords)
                else:
                    # Check periodic images using tiling directions
                    for direction in tiling.TilingDirections:
                        periodic_coords = neigh_coords + direction
                        if canvas.hasPoint(periodic_coords):
                            found_idx = canvas.getPointIndex(periodic_coords)
                            break

                neighbor_indices.append(found_idx)

            canvas._NeighborhoodIndexes[i] = neighbor_indices

    else:
        # For non-periodic structures, just use direct neighbor mapping
        for i, site_neighbors in enumerate(all_neighbors):
            neighbor_indices = []
            for neigh, dist, neigh_idx in sorted(site_neighbors, key=lambda x: x[1]):
                if canvas.hasPoint(neigh.coords):
                    neighbor_indices.append(canvas.getPointIndex(neigh.coords))
                else:
                    neighbor_indices.append(None)

            canvas._NeighborhoodIndexes[i] = neighbor_indices

    # Create Contents from species
    contents = [Atom(str(site.specie.symbol)) for site in structure]

    # Create the Design
    design = Design(canvas, contents)

    # Validate index consistency
    validate_indices(design)

    return design


def estimate_neighbor_cutoff(structure):
    """Estimate appropriate neighbor cutoff distance based on structure."""
    # Get minimum distance between any two sites
    min_dist = float("inf")
    for i, site1 in enumerate(structure):
        for j, site2 in enumerate(structure):
            if i != j:
                dist = site1.distance(site2)
                if dist < min_dist:
                    min_dist = dist

    # Use 1.2x the minimum distance for first shell neighbors
    # and 1.5x for potential second shell
    return min_dist * 1.5


def validate_indices(design):
    """Validate index consistency between Canvas and Contents"""
    canvas = design.Canvas
    contents = design.Contents

    # Check length matching
    assert len(canvas.Points) == len(
        contents
    ), f"Mismatched lengths: Canvas ({len(canvas.Points)}) vs Contents ({len(contents)})"

    # Check neighborhood indices are valid
    for i, neighbors in enumerate(canvas.NeighborhoodIndexes):
        for idx in neighbors:
            if idx is not None:
                assert (
                    0 <= idx < len(contents)
                ), f"Invalid neighbor index {idx} for point {i}"
                # Verify the neighbor point exists in canvas
                assert canvas.hasPoint(
                    canvas.Points[idx]
                ), f"Neighbor index {idx} points to missing canvas point"

    return True
