from collections import Counter
import pandas as pd
import ase
from icet import ClusterSpace, ClusterExpansion, StructureContainer
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# MatOpt imports
from matopt.materials.atom import Atom
from matopt.materials.canvas import Canvas
from matopt.aml.expr import SumSites, SumClusters
from matopt.aml.rule import EqualTo, FixedTo
from matopt.aml.model import MatOptModel


import itertools
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure


def parameterize_cluster_expansion(cs, ce, design_atoms, weight=1.0, tol=1e-15):
    """
    Expand a trained ClusterExpansion over a 'design_atoms' structure into
    fully enumerated cluster specs and coefficients for use in MatOpt, all
    in one streamlined pass (bypassing the original three-step logic).

    Args:
        cs            : icet.ClusterSpace object
        ce            : icet.ClusterExpansion object (with learned parameters)
        design_atoms  : ASE Atoms to be expanded
        weight        : overall scaling factor to multiply all coefficients
        tol           : numerical tolerance below which coefficients are ignored

    Returns:
        cluster_specs  : list of cluster specs  [ [(site_index, species_index), ...], ... ]
        cluster_coeffs : corresponding list of float coefficients
    """
    # Prepare the outputs
    cluster_specs = []
    cluster_coeffs = []

    # 0) Handle the zero-let (constant term)
    zero_coeff = ce.parameters[0] * weight
    if abs(zero_coeff) > tol:
        cluster_specs.append([])  # no sites involved in a constant term
        cluster_coeffs.append(zero_coeff)

    # 1) Generate an expanded orbit list for design_atoms
    lolg = LocalOrbitListGenerator(
        cs.orbit_list,
        Structure.from_atoms(design_atoms),
        ce.fractional_position_tolerance,
    )
    orbit_list = lolg.generate_full_orbit_list().get_orbit_list()

    # 2) Prepare the species list.  E.g.: [29, 28, 46, 47] for Cu, Ni, Pd, Ag
    elements = list(cs.species_maps[0].keys())

    # 3) Loop over orbits and their cluster-vector elements
    param_idx = 1  # because param[0] is the zero-let
    for orbit in orbit_list:
        cves = orbit.cluster_vector_elements
        if not cves:
            continue  # skip if no cluster-vector elements

        # By definition in icet, all cves in the same orbit share the same multiplicity
        multiplicity = cves[0]["multiplicity"]

        for cve in cves:
            # ECI for this cluster-vector element
            param_eci = ce.parameters[param_idx]
            param_idx += 1

            # "Outer" scaling factor = (ECI / multiplicity) * user-supplied weight
            outer_coeff = (param_eci / multiplicity) * weight
            if abs(outer_coeff) < tol:
                continue  # skip if negligible

            # 4) Each orbit has multiple "clusters" (actual sets of sites)
            for cluster in orbit.clusters:
                site_indices = [site.index for site in cluster.lattice_sites]

                # 5) Enumerate all possible species assignments on these sites
                #    (simple 1-of-k logic).
                #    If you prefer to only expand species that have basis ~ 1.0,
                #    that is effectively what we do here: no complicated polynomial terms.
                for species_combo in itertools.product(
                    range(len(elements)), repeat=len(site_indices)
                ):
                    # species_combo is e.g. (0,2) meaning site0->species_idx=0, site1->species_idx=2
                    # If we had a site-basis dictionary with polynomial expansions, we would
                    # multiply in basis values. But for the typical "1-of-k" expansions, it's 1.0.
                    final_coeff = outer_coeff

                    # If final_coeff is big enough, record it
                    if abs(final_coeff) > tol:
                        # cluster_spec = [ (site_index, species_index), ... ]
                        cluster_spec = list(zip(site_indices, species_combo))
                        cluster_specs.append(cluster_spec)
                        cluster_coeffs.append(final_coeff)

    return cluster_specs, cluster_coeffs


##############################################################################
# 2. Load real "d1" data and train the multi-species, multi-body CE
##############################################################################

# df containing:
#   df.loc[i, 'atoms'] (an ASE atoms object)
#   df.loc[i, 'MixingEnergy'] (a float property)
reference_energies = {
    "Cu": -240.11121086,
    "Ni": -352.62417749,
    "Pd": -333.69496589,
    "Ag": -173.55506507,
}
reference_structure_indice = 155
cutoffs = [5, 5]
db = ase.io.read("structures_CuNiPdAg.json", ":")
dft_data = pd.read_csv("properties_CuNiPdAg.csv", index_col=0)
indices = dft_data.index.values
mixing_energies = []
df = pd.DataFrame(columns=["atoms", "MixingEnergy", "TotalEnergy"])

for idx, i in enumerate(indices):
    # Calculate mixing energy directly in the loop instead of using a separate function
    st = db[i]
    energy = dft_data.loc[i, "final_energy"]
    dict_representation = dict(Counter(st.get_chemical_symbols()))
    NumAtoms = len(st)
    mixing_energy = (
        energy
        - sum(
            [
                v / NumAtoms * reference_energies[k]
                for k, v in dict_representation.items()
            ]
        )
    ) / NumAtoms

    df.loc[idx] = [st, mixing_energy, energy]


# Suppose we want to use 4 species: Cu, Ni, Pd, Ag
# This means the cluster expansion can place any of these on each site
cs = ClusterSpace(
    structure=df.loc[
        reference_structure_indice, "atoms"
    ],  # pick a reference structure from your dataset
    cutoffs=cutoffs,  # 2 cutoffs for multi-body expansions
    chemical_symbols=list(reference_energies.keys()),
)

# Build the StructureContainer from all data
sc = StructureContainer(cluster_space=cs)
for i in df.index:
    sc.add_structure(
        structure=df.loc[i, "atoms"],
        properties={"mixing_energy": df.loc[i, "MixingEnergy"]},
    )

# Extract design matrix (X) and target energies (y)
X, y = sc.get_fit_data(key="mixing_energy")

# Train with LassoCV
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lasso_model = LassoCV(cv=5, random_state=42)
lasso_model.fit(x_train, y_train)

print("Train R2:", lasso_model.score(x_train, y_train))
print("Test  R2:", lasso_model.score(x_test, y_test))
print("Test MAE:", mean_absolute_error(lasso_model.predict(x_test), y_test))

# Build the cluster expansion object
ce = ClusterExpansion(cluster_space=cs, parameters=lasso_model.coef_)

# Incorporate the intercept into the zero-let:
ce.parameters[0] += lasso_model.intercept_


##############################################################################
# 3. Pick a design structure & fully expand the learned CE for MILP
##############################################################################

# Example: pick the structure at df.loc[0, 'atoms'] as our "design space."
design_atoms = df.loc[0, "atoms"]

# # You could repeat the design structure to get a larger design space
# # But you may need a commercial MILP solver and it will take longer to run
# # We did not do that here for demonstration efficiency purposes.
# design_atoms = design_atoms.repeat(2)

design_atoms = df.loc[0, "atoms"]
cluster_specs, cluster_coeffs = parameterize_cluster_expansion(
    cs, ce, design_atoms, weight=1.0
)


# Example cluster_specs and cluster_coeffs entries
# [(40, 3), (47, 0), (44, 3)] -> 5.806649911771741e-05
# [(44, 3), (57, 1)] -> -0.0001856590690370786

##############################################################################
# 4. Construct the MatOpt MILP
##############################################################################

# 4.1: Build site geometry + adjacency from 2-body orbits
points_list = design_atoms.get_positions().tolist()
num_sites = len(design_atoms)


# 4.2: Create Canvas
Canv = Canvas(points_list)

# (Optional) if you want to add neighboring relationships-based constraints, a way to do it is:
# lolg_design = LocalOrbitListGenerator(
#     cs.orbit_list, Structure.from_atoms(design_atoms), ce.fractional_position_tolerance
# )
# full_orbit_list = lolg_design.generate_full_orbit_list()

# nb_indices_list = [[] for _ in range(num_sites)]
# for orb_idx in range(len(full_orbit_list)):
#     orbit = full_orbit_list.get_orbit(orb_idx)
#     if orbit.order == 2:
#         for clus in orbit.clusters:
#             s1, s2 = [site.index for site in clus.lattice_sites]
#             if s2 not in nb_indices_list[s1]:
#                 nb_indices_list[s1].append(s2)
#             if s1 not in nb_indices_list[s2]:
#                 nb_indices_list[s2].append(s1)
# Canv = Canvas(points_list, nb_indices_list)

# 4.3: List possible Atoms
# Must match the "chemical_symbols" used in your ClusterSpace
# (and in the same order you used in 'elements' if you want a 1-1 mapping).
AllAtoms = [Atom(s) for s in reference_energies.keys()]

# 4.4: Convert cluster_specs => (site, Atom)
cluster_specs_expanded = []
for c in cluster_specs:
    # c = [(site_idx, species_idx), ...]
    c_expanded = []
    for st, spidx in c:
        # Map spidx => correct Atom from AllAtoms
        # This assumes your 'elements' order matches AllAtoms order.
        c_expanded.append((st, AllAtoms[spidx]))
    cluster_specs_expanded.append(c_expanded)


# 4.5: Build the MatOpt Model
m = MatOptModel(Canv, AllAtoms, clusters=cluster_specs_expanded[1:])

# (Optional) Add a composition constraint, e.g. fix a certain # of Ni sites, etc.
# For demonstration, we limit the number of atoms for expensive elements
CompBounds = {
    AllAtoms[0]: (0, len(design_atoms)),
    AllAtoms[1]: (0, len(design_atoms)),
    AllAtoms[2]: (5, 10),
    AllAtoms[3]: (5, 10),
}
m.addGlobalTypesDescriptor(
    "Composition", bounds=CompBounds, rules=EqualTo(SumSites(desc=m.Yik))
)

# We are only optimizing for the chemical composition, not the structure
# If this is not included, the optimizer will try to eliminate all clusters with positive coefficients
m.Yi.rules.append(FixedTo(1.0))

# 4.6: Objective = sum over cluster_specs with cluster_coeffs
obj_expr = SumClusters(desc=m.Zn, coefs=cluster_coeffs[1:])

D = m.minimize(obj_expr, solver="neos-cplex", tilim=360)
D.toPDB("result.pdb")  # Export to Protein Data Bank format
D.toXYZ("result.xyz")  # Export to XYZ format
D.toCFG("result.cfg")  # Export to AtomEye configuration format
D.toPOSCAR("result.vasp")  # Export to VASP POSCAR format

# Convert to pymatgen Structure for further analysis
from pymatgen.core import Structure

structure = D.to_pymatgen()  # Convert to pymatgen Structure object
structure.to_file("result.cif")  # Export to CIF format
