<img src="assets/matopt.png" alt="MatOpt Logo" width="300">

# MatOpt: Materials Design via Mathematical Optimization

> [!IMPORTANT]
> This is a standalone version adapted from the [original implementation](https://github.com/IDAES/idaes-pse/tree/main/idaes/apps/matopt)

MatOpt is an Algebraic Modeling Language (AML) system specifically designed to create [Pyomo](https://www.pyomo.org/) objects for optimization-based nanomaterials design. It enables modeling of crystalline nanostructured materials, including particles, surfaces, and periodic bulk structures.

## 🚀 Key Features

- **Simplified materials representation**: Streamline the creation of materials optimization problems
- **Automated optimization workflows**: Speed up the development of new models for materials discovery
- **Intuitive interface**: No need to handle the details of mathematical optimization or Pyomo syntax

## 📦 Installation

```bash
pip install matopt
```

## 🧪 Citation

If you use MatOpt, please consider citing:

- Hanselman, C.L., Yin, X., Miller, D.C. and Gounaris, C.E., 2022. [MatOpt: A Python Package for Nanomaterials Design Using Discrete Optimization.](https://doi.org/10.1021/acs.jcim.1c00984) _Journal of Chemical Information and Modeling_, 62(2), pp.295-308.

## 📚 Package Structure

MatOpt consists of two main sub-modules:

- **`matopt.materials`**: Objects and methods for efficiently representing and manipulating nanomaterials and their design space
- **`matopt.aml`**: Objects and methods for simplified Pyomo model creation with automatic model formulation tools

## 🔧 Dependencies

MatOpt requires access to the CPLEX solver through Pyomo. If you don't have CPLEX access, you can use [NEOS-CPLEX](https://neos-guide.org/neos-interfaces#pyomo) as an alternative. (Users need to set up a `NEOS_EMAIL` environment variable.)

## 🎨 Creating a Material Design Canvas

The `matopt.materials` module is the foundation of MatOpt, providing a rich set of objects and methods for representing materials and their design space. This module enables you to create, manipulate, and visualize nanomaterials with precise control.

### Canvas

The `Canvas` class is the central container that defines the design space for your optimization problem. It manages:

- Coordinates where atoms or building blocks can be placed
- Connectivity between sites (critical for bond-based properties)
- Design space boundaries and periodicity

### Atoms

The `Atom` class represents individual elements with their properties:

- Element symbol and atomic number
- Radius, mass, and other physical properties
- Custom properties for specific modeling needs

### Geometry Tools

The `geometry` module provides utilities for:

- Coordinate transformations and rotations
- Distance and angle calculations
- Spatial operations on atom collections

### Motifs and Building Blocks

The `motifs` module allows you to:

- Create recurring atomic arrangements (e.g., functional groups)
- Define building blocks that can be used as optimization variables
- Reuse common structural elements across designs

### Lattices

The `lattices` submodule implements common crystal lattice structures:

- FCC, BCC, HCP, diamond, wurtzite, perovskite lattices
- Methods for generating coordinates and neighbor relations
- Tools for transforming between lattice representations

### Transformations

The `transform_func` module provides functions to:

- Translate, rotate, and scale designs
- Apply symmetry operations
- Deform structures in controlled ways

### Tilings and Patterns

The `tiling` module helps with:

- Creating regular arrangements of atoms
- Building complex surfaces with specific patterns
- Generating periodic structures

## 🏗️ Building Optimization Models

MatOpt uses `MaterialDescriptor` objects to represent variables, constraints, and objectives. A `MatOptModel` object holds lists of these descriptors. Several universal site descriptors are pre-defined:

| Descriptor | Explanation                                                               |
| ---------- | ------------------------------------------------------------------------- |
| `Yik`      | Presence of a building block of type k at site i                          |
| `Yi`       | Presence of any type of building block at site i                          |
| `Xijkl`    | Presence of a building block of type k at site i and type l at site j     |
| `Xij`      | Presence of any building block at sites i and j                           |
| `Cikl`     | Count of neighbors of type l next to a building block of type k at site i |
| `Ci`       | Count of any type of neighbors next to a building block at site i         |
| `Zn`       | Presense of the n-th cluster, as specified by the user                    |

User-specified descriptors are defined by `DescriptorRule` objects with `Expr` expression objects:

### Expression Types

| Expression         | Purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| `LinearExpr`       | Multiplication and addition of coefficients to descriptors |
| `SiteCombination`  | Summation of site contributions from two sites             |
| `SumNeighborSites` | Summation of site contributions from all neighboring sites |
| `SumNeighborBonds` | Summation of bond contributions to all neighboring sites   |
| `SumSites`         | Summation across sites                                     |
| `SumBonds`         | Summation across bonds                                     |
| `SumSiteTypes`     | Summation across site types                                |
| `SumBondTypes`     | Summation across bond types                                |
| `SumSitesAndTypes` | Summation across sites and site types                      |
| `SumBondsAndTypes` | Summation across bonds and bond types                      |
| `SumConfs`         | Summation across conformation types                        |
| `SumSitesAndConfs` | Summation across sites and conformation types              |
| `SumClusters`      | Summation across clusters (from cluster expansion)         |
| `SumExpressions`   | Summation of a list of expressions                         |

### Descriptor Rules

| Rule                     | Purpose                                                                         |
| ------------------------ | ------------------------------------------------------------------------------- |
| `LessThan`               | Descriptor less than or equal to an expression                                  |
| `EqualTo`                | Descriptor equal to an expression                                               |
| `GreaterThan`            | Descriptor greater than or equal to an expression                               |
| `FixedTo`                | Descriptor fixed to a scalar value                                              |
| `PiecewiseLinear`        | Descriptor equal to the evaluation of a piecewise linear function               |
| `Implies`                | Indicator descriptor that imposes other constraints if equal to 1               |
| `NegImplies`             | Indicator descriptor that imposes other constraints if equal to 0               |
| `ImpliesSiteCombination` | Indicator bond-indexed descriptor that imposes constraints on the two sites     |
| `ImpliesNeighbors`       | Indicator site-indexed descriptor that imposes constraints on neighboring sites |

## 📝 Example: Pt Nanocluster Design

Once the model is fully specified, the user can optimize it in light of a chosen descriptor to serve as the objective to be maximized or minimized, as appropriate. Several functions are provided for users to choose from. The results of the optimization process will be loaded into Design objects automatically.

```python
import numpy as np
from math import sqrt
from matopt.materials.atom import Atom
from matopt.materials.lattices.fcc_lattice import FCCLattice
from matopt.materials.canvas import Canvas
from matopt.aml.expr import SumSites
from matopt.aml.rule import PiecewiseLinear, EqualTo
from matopt.aml.model import MatOptModel

Lat = FCCLattice(IAD=2.7704443686888935)
Canv = Canvas()
Canv.addLocation(np.array([0, 0, 0], dtype=float))
Canv.addShells(2, Lat.getNeighbors)
Canv.setNeighborsFromFunc(Lat.getNeighbors)
Atoms = [Atom("Pt")]

N = 22
m = MatOptModel(Canv, Atoms)
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
    "Ecoh", rules=EqualTo(SumSites(desc=m.CNRi, coefs=(1.0 / sqrt(12) * 1.0 / N)))
)
m.addGlobalDescriptor("Size", bounds=(N, N), rules=EqualTo(SumSites(desc=m.Yi)))

D = m.maximize(m.Ecoh, tilim=100, solver="neos-cplex")
```

For more examples and detailed tutorials, check the `examples` folder or the [original repository](https://github.com/IDAES/idaes-pse/tree/main/idaes/apps/matopt).

## 💾 Exporting Design Results

MatOpt provides several methods to export your optimized designs:

```python
# Assuming D is your optimized Design object from m.maximize() or m.minimize()

# Export to common crystal structure formats
D.toPDB("result.pdb")     # Export to Protein Data Bank format
D.toXYZ("result.xyz")     # Export to XYZ format
D.toCFG("result.cfg")     # Export to AtomEye configuration format
D.toPOSCAR("result.vasp") # Export to VASP POSCAR format

# Convert to pymatgen Structure for further analysis
from pymatgen.core import Structure
structure = D.to_pymatgen()  # Convert to pymatgen Structure object
```

## 📊 Exporting Optimization Models

MatOpt allows you to export the underlying optimization models:

```python
# Create your model as usual
m = MatOptModel(Canv, Atoms)
# ... add descriptors and constraints ...

# Export to LP format as standard Mixed-Integer Linear Program
milp = m._make_pyomo_model(m.Ecoh, maximize=True)
milp.write("milp.lp")

# Export as Mixed-Integer Quadratic Constrained Program
miqcqp = m._make_pyomo_model(m.Ecoh, maximize=True, formulation="miqcqp")
miqcqp.write("miqcqp.lp")
```

### Integration with Quantum Computing Solvers

```python
# Using Qiskit Optimization
# Note: Install with `pip install 'qiskit-optimization[cplex]' docplex cplex`
from qiskit_optimization import QuadraticProgram
qiskit_qp = QuadraticProgram()
qiskit_qp.read_from_lp_file("miqcqp.lp")
# Now you can solve with Qiskit's quantum optimization algorithms

# Using D-Wave's dimod
# Note: Install with `pip install dimod`
import dimod
with open("miqcqp.lp", "rb") as f:
    cqm = dimod.lp.load(f)
# Now you can solve with D-Wave's quantum annealing hardware
```

## 📚 Applications

- Hanselman, C.L. and Gounaris, C.E., 2016. [A mathematical optimization framework for the design of nanopatterned surfaces.](https://aiche.onlinelibrary.wiley.com/doi/full/10.1002/aic.15359) _AIChE Journal_
- Hanselman, C.L. et al., 2019. [A framework for optimizing oxygen vacancy formation in doped perovskites.](https://www.sciencedirect.com/science/article/pii/S0098135418310998) _Computers & Chemical Engineering_
- Hanselman, C.L. et al., 2019. [Optimization-based design of active and stable nanostructured surfaces.](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.9b08431) _The Journal of Physical Chemistry C_
- Isenberg, N.M. et al., 2020. [Identification of optimally stable nanocluster geometries via mathematical optimization and density-functional theory.](https://pubs.rsc.org/en/content/articlelanding/2019/me/c9me00108e#!divAbstract) _Molecular Systems Design & Engineering_
- Yin, X. et al., 2021. [Designing stable bimetallic nanoclusters via an iterative two-step optimization approach.](https://pubs.rsc.org/en/content/articlelanding/2021/ME/D1ME00027F) _Molecular Systems Design & Engineering_
- Yin, X. and Gounaris, C.E., 2022. [Search methods for inorganic materials crystal structure prediction.](https://doi.org/10.1016/j.coche.2021.100726) _Current Opinion in Chemical Engineering_
