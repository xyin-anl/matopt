from abc import ABC, abstractmethod

from pyomo.environ import *
from pyomo.opt.results import SolutionStatus

from matopt.materials.design import Design
from matopt.aml.desc import MaterialDescriptor
from matopt.aml.rule import Disallow, FixedTo
from matopt.aml.util import addConsForGeneralVars, setDesignFromModel


class BaseModel(ABC):
    """A class for the specification of a materials optimization problem.

    Once all the material information is specified, we use this class to
    specify the material design problem of interest. This class is intended
    to be interpretable without mathematical optimization background while
    the conversion to Pyomo optimization models happens automatically.

    Attributes:
        canv (``Canvas``): The canvas of the material design space
        atoms (list<``BBlock``>): The list of building blocks to consider.
            Note: This list does not need to include a void-atom type. We use
            'None' to represent the absence of any building block at a given site.
        confDs (list<``Design``>): The list of conformations to consider.
    """

    # === STANDARD CONSTRUCTOR
    def __init__(self, canv, atoms=None, confDs=None, **kwargs):
        """Standard constructor for materials optimization problems.

        Args:
        canv (``Canvas``): The canvas of the material design space
        atoms (list<``BBlock``>): The list of building blocks to consider.
            Note: This list does not need to include a void-atom type. We use
            'None' to represent the absence of any building block at a given site.
        confDs (list<``Design``>): The list of conformations to consider.
        """
        self._canv = canv
        self._atoms = atoms
        self._confDs = confDs
        self._descriptors = []
        self.addSitesDescriptor("Yi", binary=True, rules=None)
        self.addBondsDescriptor("Xij", binary=True, rules=None)
        self.addNeighborsDescriptor("Ci", integer=True, rules=None)
        self.addSitesTypesDescriptor("Yik", binary=True, rules=None)
        self.addBondsTypesDescriptor("Xijkl", binary=True, rules=None)
        self.addNeighborsTypesDescriptor("Cikl", integer=True, rules=None)
        self.addSitesConfsDescriptor("Zic", binary=True, rules=None)

    # === MANIPULATION METHODS
    def addGlobalDescriptor(
        self, name, bounds=(None, None), integer=False, binary=False, rules=None
    ):
        """
        Method to add scalar descriptor to the model.

        Args:
            name (string): A unique (otherwise Pyomo will complain) name.
            bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
                descriptor values across all indices. If dict, the bounds can be
                individually set for each index. Otherwise, advanced users can
                specify a function to be interpreted by Pyomo.
            integer (bool): Flag to indicate if the descriptor is integer.
            binary (bool): Flag to indicate if the descriptor is boolean.
            rules (list<DescriptorRules>): List of rules to define and constrain
                the material descriptor design space.
        """
        assert not hasattr(self, name)
        Desc = MaterialDescriptor(
            name=name, bounds=bounds, integer=integer, binary=binary, rules=rules
        )
        setattr(self, name, Desc)
        self._descriptors.append(Desc)

    def addSitesDescriptor(
        self,
        name,
        sites=None,
        bounds=(None, None),
        integer=False,
        binary=False,
        rules=None,
    ):
        """Method to add a site-indexed descriptor to the model.

        Args:
            name (string): A unique name.
            sites (list<int>): Optional, subset of canvas sites to index the new
                descriptor over.
                Default: None (i.e., all sites in canvas are considered)
            bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
                descriptor values across all indices. If dict, the bounds can be
                individually set for each index. Otherwise, advanced users can
                specify a function.
            integer (bool): Flag to indicate if the descriptor is integer.
            binary (bool): Flag to indicate if the descriptor is boolean.
            rules (list<DescriptorRules>): List of rules to define and constrain
                the material descriptor design space.
        """
        assert not hasattr(self, name)
        sites = sites if sites is not None else list(range(len(self.canv)))
        Desc = MaterialDescriptor(
            name=name,
            canv=self.canv,
            sites=sites,
            bounds=bounds,
            integer=integer,
            binary=binary,
            rules=rules,
        )
        setattr(self, name, Desc)
        self._descriptors.append(Desc)

    def addBondsDescriptor(
        self,
        name,
        bonds=None,
        bounds=(None, None),
        integer=False,
        binary=False,
        rules=None,
        symmetric_bonds=False,
    ):
        """Method to add a bond-indexed descriptor to the model.

        Args:
            name (string): A unique name.
            bonds (list<tuple<int,int>>): Optional, subset of canvas neighbor
                pairs to index the new descriptor over.
                Default: None (i.e., all neighbor pairs included)
            bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
                descriptor values across all indices. If dict, the bounds can be
                individually set for each index. Otherwise, advanced users can
                specify a function.
            integer (bool): Flag to indicate if the descriptor is integer.
            binary (bool): Flag to indicate if the descriptor is boolean.
            rules (list<DescriptorRules>): List of rules to define and constrain
                the material descriptor design space.
        """
        assert not hasattr(self, name)
        bonds = (
            bonds
            if bonds is not None
            else [
                (i, j)
                for i in range(len(self.canv))
                for j in self.canv.NeighborhoodIndexes[i]
                if (j is not None and (not symmetric_bonds or j > i))
            ]
        )
        Desc = MaterialDescriptor(
            name=name,
            canv=self.canv,
            bonds=bonds,
            bounds=bounds,
            integer=integer,
            binary=binary,
            rules=rules,
        )
        setattr(self, name, Desc)
        self._descriptors.append(Desc)

    def addNeighborsDescriptor(
        self,
        name,
        sites=None,
        bounds=(None, None),
        integer=False,
        binary=False,
        rules=None,
    ):
        """Method to add a neighborhood-indexed descriptor to the model.

        Args:
            name (string): A unique name.
            sites (list<int>): Optional, subset of canvas sites to index the new
                descriptor over.
                Default: None (i.e., all sites in canvas are considered)
            bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
                descriptor values across all indices. If dict, the bounds can be
                individually set for each index. Otherwise, advanced users can
                specify a function.
            integer (bool): Flag to indicate if the descriptor is integer.
            binary (bool): Flag to indicate if the descriptor is boolean.
            rules (list<DescriptorRules>): List of rules to define and constrain
                the material descriptor design space.
        """
        assert not hasattr(self, name)
        sites = sites if sites is not None else list(range(len(self.canv)))
        Desc = MaterialDescriptor(
            name=name,
            canv=self.canv,
            sites=sites,
            bounds=bounds,
            integer=integer,
            binary=binary,
            rules=rules,
        )
        setattr(self, name, Desc)
        self._descriptors.append(Desc)

    def addGlobalTypesDescriptor(
        self,
        name,
        site_types=None,
        bond_types=None,
        bounds=(None, None),
        integer=False,
        binary=False,
        rules=None,
    ):
        """Method to add a type-indexed descriptor to the model.

        Args:
            name (string): A unique name.
            site_types (list<BBlock>): Optional, subset of building block types
                to index the new descriptor over.
                Note: If both site_types and bond_types are left to None, then
                we decide to index over building block types by default.
            bond_types (list<tuple<BBlock,BBlock>>): Optional, subset of building
                block pairs to index the new descriptor over.
                Note: If both site_types and bond_types are left to None, then
                we decide to index over building block types by default.
            bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
                descriptor values across all indices. If dict, the bounds can be
                individually set for each index. Otherwise, advanced users can
                specify a function.
            integer (bool): Flag to indicate if the descriptor is integer.
            binary (bool): Flag to indicate if the descriptor is boolean.
            rules (list<DescriptorRules>): List of rules to define and constrain
                the material descriptor design space.
        """
        assert not hasattr(self, name)
        site_types = (
            site_types
            if site_types is not None
            else (self.atoms if bond_types is None else None)
        )
        bond_types = bond_types
        Desc = MaterialDescriptor(
            name=name,
            atoms=self.atoms,
            site_types=site_types,
            bond_types=bond_types,
            bounds=bounds,
            integer=integer,
            binary=binary,
            rules=rules,
        )
        setattr(self, name, Desc)
        self._descriptors.append(Desc)

    def addSitesTypesDescriptor(
        self,
        name,
        sites=None,
        site_types=None,
        bounds=(None, None),
        integer=False,
        binary=False,
        rules=None,
    ):
        """Method to add a site-and-type-indexed descriptor to the model.

        Args:
            name (string): A unique name.
            sites (list<int>): Optional, subset of canvas sites to index the new
                descriptor over.
                Default: None (i.e., all sites in canvas are considered)
            site_types (list<BBlock>): Optional, subset of building block types
                to index the new descriptor over.
                Default: None (i.e., all building block types are considered)
            bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
                descriptor values across all indices. If dict, the bounds can be
                individually set for each index. Otherwise, advanced users can
                specify a function.
            integer (bool): Flag to indicate if the descriptor is integer.
            binary (bool): Flag to indicate if the descriptor is boolean.
            rules (list<DescriptorRules>): List of rules to define and constrain
                the material descriptor design space.
        """
        assert not hasattr(self, name)
        sites = sites if sites is not None else list(range(len(self.canv)))
        site_types = site_types if site_types is not None else self.atoms
        Desc = MaterialDescriptor(
            name=name,
            atoms=self.atoms,
            canv=self.canv,
            sites=sites,
            site_types=site_types,
            bounds=bounds,
            integer=integer,
            binary=binary,
            rules=rules,
        )
        setattr(self, name, Desc)
        self._descriptors.append(Desc)

    def addBondsTypesDescriptor(
        self,
        name,
        bonds=None,
        bond_types=None,
        bounds=(None, None),
        integer=False,
        binary=False,
        rules=None,
        symmetric_bonds=False,
    ):
        """Method to add a bond-and-type-indexed descriptor to the model.

        Args:
            name (string): A unique name.
            bonds (list<tuple<int,int>>): Optional, subset of canvas neighbor
                pairs to index the new descriptor over.
                Default: None (i.e., all neighbor pairs included)
            bond_types (list<tuple<BBlock,BBlock>>): Optional, subset of
                building block pairs to index the new descriptor over.
                Default: None (i.e., all pairs of building blocks considered)
            bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
                descriptor values across all indices. If dict, the bounds can be
                individually set for each index. Otherwise, advanced users can
                specify a function.
            integer (bool): Flag to indicate if the descriptor is integer.
            binary (bool): Flag to indicate if the descriptor is boolean.
            rules (list<DescriptorRules>): List of rules to define and constrain
                the material descriptor design space.
        """
        assert not hasattr(self, name)
        bonds = (
            bonds
            if bonds is not None
            else [
                (i, j)
                for i in range(len(self.canv))
                for j in self.canv.NeighborhoodIndexes[i]
                if (j is not None and (not symmetric_bonds or j > i))
            ]
        )
        bond_types = (
            bond_types
            if bond_types is not None
            else [(k, l) for k in self.atoms for l in self.atoms]
        )
        Desc = MaterialDescriptor(
            name=name,
            atoms=self.atoms,
            canv=self.canv,
            bonds=bonds,
            bond_types=bond_types,
            bounds=bounds,
            integer=integer,
            binary=binary,
            rules=rules,
        )
        setattr(self, name, Desc)
        self._descriptors.append(Desc)

    def addNeighborsTypesDescriptor(
        self,
        name,
        sites=None,
        bond_types=None,
        bounds=(None, None),
        integer=False,
        binary=False,
        rules=None,
    ):
        """Method to add a neighborhood-bond-type-indexed descriptor.

        Args:
            name (string): A unique name.
            sites (list<int>): Optional, subset of canvas sites to index the new
                descriptor over.
                Default: None (i.e., all sites in canvas are considered)
            bond_types (list<tuple<BBlock,BBlock>>): Optional, subset of building
                block pairs to index the new descriptor over.
                Default: None (i.e., all pairs of building blocks considered)
            bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
                descriptor values across all indices. If dict, the bounds can be
                individually set for each index. Otherwise, advanced users can
                specify a function.
            integer (bool): Flag to indicate if the descriptor is integer.
            binary (bool): Flag to indicate if the descriptor is boolean.
            rules (list<DescriptorRules>): List of rules to define and constrain
                the material descriptor design space.
        """
        assert not hasattr(self, name)
        sites = sites if sites is not None else list(range(len(self.canv)))
        bond_types = (
            bond_types
            if bond_types is not None
            else [(k, l) for k in self.atoms for l in self.atoms]
        )
        Desc = MaterialDescriptor(
            name=name,
            atoms=self.atoms,
            canv=self.canv,
            sites=sites,
            bond_types=bond_types,
            bounds=bounds,
            integer=integer,
            binary=binary,
            rules=rules,
        )
        setattr(self, name, Desc)
        self._descriptors.append(Desc)

    def addSitesConfsDescriptor(
        self,
        name,
        sites=None,
        confs=None,
        bounds=(0, 1),
        integer=True,
        binary=True,
        rules=None,
    ):
        """Method to add a site-and-conformation-indexed descriptor.

        Args:
            name (string): A unique name.
            sites (list<int>): Optional, subset of canvas sites to index the new
                descriptor over.
                Default: None (i.e., all sites in canvas are considered)
            confs (list<int>): Optional, subset of conformation indices to index
                the new descriptor over.
                Default: None (i.e., all conformations included)
            bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
                descriptor values across all indices. If dict, the bounds can be
                individually set for each index. Otherwise, advanced users can
                specify a function.
            integer (bool): Flag to indicate if the descriptor is integer.
            binary (bool): Flag to indicate if the descriptor is boolean.
            rules (list<DescriptorRules>): List of rules to define and constrain
                the material descriptor design space.
        """
        sites = sites if sites is not None else list(range(len(self.canv)))
        confs = (
            confs
            if confs is not None
            else (list(range(len(self.confDs))) if self.confDs is not None else None)
        )
        Desc = MaterialDescriptor(
            name=name,
            canv=self.canv,
            confDs=self.confDs,
            sites=sites,
            confs=confs,
            bounds=bounds,
            integer=integer,
            binary=binary,
            rules=rules,
        )
        setattr(self, name, Desc)
        self._descriptors.append(Desc)

    # === BASIC QUERY METHODS
    @property
    def canv(self):
        return self._canv

    @property
    def atoms(self):
        return self._atoms

    @property
    def confDs(self):
        return self._confDs

    @property
    def descriptors(self):
        return self._descriptors

    @abstractmethod
    def optimize(self, func, sense, **kwargs):
        pass


def makeMyPyomoBaseModel(C, Atoms=None, Confs=None):
    """
    Make the Pyomo model for a basic materials design problem.

    Creates the basic sets and variables that make up the problem.
    All variables are created sparsely, so they are initialized
    only after they are actually referenced in a constraint. In
    this way, we smartly eliminate unnecessary variables and
    constraints.

    Basic Variables:
        Yi: Presence of building block at site i
        Xij: Presence of building blocks at both sites i and j
        Ci: Count of building block bonds to neighbors to site i

    Common Descriptors:
        Zi: Presence of target site at i

    Atom-Specific Variables:
        Yik: Presence of building block of type k at site i
        Xijkl: Presence of type k and l at sites i and j, respectively
        Cikl: Count of neighbors of type l next to site i with type k

    Conformation-Specific Descriptors:
        Zic: Presence of conformation c at site i

    The basic variables and atom-specific variables above are
    automatically encoded by calling addConsForGeneralVars.
    The descriptor variables Zi and Zic must be explicitly constrained.
    Some standard approaches are formalized in:

    * addConsBoundDescriptorsWithImpl
    * addConsIndFromDescriptors
    * addConsZicFromYi
    * addConsZicFromYiLifted
    * addConsZicFromYik
    * addConsZicFromYikLifted

    Additional variables and constraints  can be created and managed on
    a model-specific level.

    Args:
        C (Canvas): The design space over which to model the problem.
        Atoms (list<Atom/Any>): Optional, the set of building blocks.
            If present, generates a meaningful set for K building blocks.
            Else, only the general variables for presence/absence are
            meaningful. (Default value = None)
        Confs (list<Design>): Optional, the set of conformations to
            potentially use for indicator variables.
            (Default value = None)

    Returns:
        (ConcreteModel): Pyomo model with basic sets and variables initialized.

    """
    m = ConcreteModel()
    # Adding Model formulation information
    m.Canvas = C
    m.Ni = C.NeighborhoodIndexes
    m.Atoms = Atoms
    m.Confs = Confs
    # Adding Basic Sets
    m.nI = len(C)
    m.I = Set(initialize=range(m.nI), ordered=True)
    # Adding Basic Variables
    m.Yi = Var(m.I, domain=Binary, dense=False)
    m.Xij = Var(m.I, m.I, domain=Binary, dense=False)

    def _ruleCiBounds(m, i):
        return 0, len(m.Ni[i])

    m.Ci = Var(m.I, domain=NonNegativeIntegers, bounds=_ruleCiBounds, dense=False)
    # Adding Common Descriptors
    m.Zi = Var(m.I, domain=Binary, dense=False)
    # Adding Atoms Set
    m.nK = len(Atoms) if Atoms is not None else 0
    m.K = Set(initialize=Atoms)
    m.Yik = Var(m.I, m.K, domain=Binary, dense=False)
    m.Xijkl = Var(m.I, m.I, m.K, m.K, domain=Binary, dense=False)

    def _ruleCiklBounds(m, i, k, l):
        return 0, len(m.Ni[i])

    m.Cikl = Var(
        m.I, m.K, m.K, domain=NonNegativeIntegers, bounds=_ruleCiklBounds, dense=False
    )
    # Adding Confs Set
    m.nC = len(Confs) if Confs is not None else 0
    m.C = Set(initialize=range(m.nC))
    m.Zic = Var(m.I, m.C, domain=Binary, dense=False)
    return m


class MatOptModel(BaseModel):
    def maximize(self, func, **kwargs):
        """Method to maximize a target functionality of the material model.

        Args:
            func (``MaterialDescriptor``/``Expr``): Material functionality to optimize.
            **kwargs: Arguments to ``MatOptModel.optimize``

        Returns:
            (``Design``/list<``Design``>) Optimal designs.

        Raises:
            ``pyomo.common.errors.ApplicationError`` if MatOpt can not find
            usable solver (CPLEX or NEOS-CPLEX)

        See ``MatOptModel.optimize`` method for details.
        """
        return self.optimize(func, sense=maximize, **kwargs)

    def minimize(self, func, **kwargs):
        """Method to minimize a target functionality of the material model.

        Args:
            func (``MaterialDescriptor``/``Expr``): Material functionality to optimize.
            **kwargs: Arguments to ``MatOptModel.optimize``

        Returns:
            (``Design``/list<``Design``>) Optimal designs.

        Raises:
            ``pyomo.common.errors.ApplicationError`` if MatOpt can not find usable
            solver (CPLEX or NEOS-CPLEX)

        See ``MatOptModel.optimize`` method for details.
        """
        return self.optimize(func, sense=minimize, **kwargs)

    def optimize(
        self,
        func,
        sense,
        nSolns=1,
        tee=True,
        disp=1,
        keepfiles=False,
        tilim=3600,
        trelim=None,
        solver="cplex",
    ):
        """Method to create and optimize the materials design problem.

        This method automatically creates a new optimization model every
        time it is called. Then, it solves the model via Pyomo with the
        CPLEX solver.

        If multiple solutions (called a 'solution pool') are desired, then
        the nSolns argument can be provided and the populate method will
        be called instead.

        Args:
            func (``MaterialDescriptor``/``Expr``): Material functionality to optimize.
            sense (int): flag to indicate the choice to minimize or maximize the
                functionality of interest.
                Choices: minimize/maximize (Pyomo constants 1,-1 respectively)
            nSolns (int): Optional, number of Design objects to return.
                Default: 1 (See ``MatOptModel.populate`` for more information)
            tee (bool): Optional, flag to turn on solver output.
                Default: True
            disp (int): Optional, flag to control level of MatOpt output.
                Choices: 0: No MatOpt output (other than solver tee) 1: MatOpt
                output for outer level method 2: MatOpt output for solution pool &
                individual solns. Default: 1
            keepfiles (bool): Optional, flag to save temporary pyomo files.
                Default: True
            tilim (float): Optional, solver time limit (in seconds).
                Default: 3600
            trelim (float): Optional, solver tree memory limit (in MB).
                Default: None (i.e., Pyomo/CPLEX default)
            solver (str): Solver choice. Currently only cplex or neos-cplex are supported
                Default: cplex

        Returns:
            (``Design``/list<``Design``>) Optimal design or designs, depending
            on the number of solutions requested by argument ``nSolns``.

        Raises:
            ``pyomo.common.errors.ApplicationError`` if MatOpt can not find
            usable solver (CPLEX or NEOS-CPLEX)
        """
        if nSolns > 1:
            return self.populate(
                func,
                sense=sense,
                nSolns=nSolns,
                tee=tee,
                disp=disp,
                keepfiles=keepfiles,
                tilim=tilim,
                trelim=trelim,
                solver=solver,
            )
        elif nSolns == 1:
            self._pyomo_m = self._make_pyomo_model(func, sense)
            return self.__solve_pyomo_model(tee, disp, keepfiles, tilim, trelim, solver)

    def populate(
        self,
        func,
        sense,
        nSolns,
        tee=True,
        disp=1,
        keepfiles=False,
        tilim=3600,
        trelim=None,
        solver="cplex",
    ):
        """Method to a pool of solutions that optimize the material model.

        This method automatically creates a new optimization model every
        time it is called. Then, it solves the model via Pyomo with the
        CPLEX solver.

        The populate method iteratively solves the model, interprets the
        solution as a Design object, creates a constraint to disallow that
        design, and resolves to find the next best design. We build a pool
        of Designs that are guaranteed to be the nSolns-best solutions in the
        material design space.

        Args:
            func (``MaterialDescriptor``/``Expr``): Material functionality to optimize.
            sense (int): flag to indicate the choice to minimize or maximize
                the functionality of interest.
                Choices: minimize/maximize (Pyomo constants 1,-1 respectively)
            nSolns (int): Optional, number of Design objects to return.
                Default: 1 (See ``MatOptModel.populate`` for more information)
            tee (bool): Optional, flag to turn on solver output.
                Default: True
            disp (int): Optional, flag to control level of MatOpt output.
                Choices: 0: No MatOpt output (other than solver tee) 1: MatOpt
                output for outer level method 2: MatOpt output for solution
                pool & individual solns. Default: 1
            keepfiles (bool): Optional, flag to save temporary pyomo files.
                Default: True
            tilim (float): Optional, solver time limit (in seconds).
                Default: 3600
            trelim (float): Optional, solver tree memory limit (in MB).
                Default: None (i.e., Pyomo/CPLEX default)
            solver (str): Solver choice. Currently only cplex or neos-cplex are
                supported Default: cplex

        Returns:
            (list<``Design``>) A list of optimal Designs in order of decreasing
            optimality.

        Raises:
            ``pyomo.common.errors.ApplicationError`` if MatOpt can not find
            usable solver (CPLEX or NEOS-CPLEX)
        """
        self._pyomo_m = self._make_pyomo_model(func, sense)
        self._pyomo_m.iSolns = Set(initialize=list(range(nSolns)))
        self._pyomo_m.IntCuts = Constraint(self._pyomo_m.iSolns)

        def dispPrint(*args):
            if disp > 0:
                print(*args)
            else:
                pass

        Ds = []
        for iSoln in range(nSolns):
            dispPrint("Starting populate for solution #{}... ".format(iSoln))
            D = self.__solve_pyomo_model(
                tee, disp - 1, keepfiles, tilim, trelim, solver
            )
            if D is not None:
                dispPrint(
                    "Found solution with objective: {}".format(value(self._pyomo_m.obj))
                )
                Ds.append(D)
                if len(self._pyomo_m.Yik) > 0:
                    self._pyomo_m.IntCuts.add(
                        index=iSoln, expr=(Disallow(D)._pyomo_expr(self.Yik) >= 1)
                    )
                elif len(self._pyomo_m.Yi) > 0:
                    self._pyomo_m.IntCuts.add(
                        index=iSoln, expr=(Disallow(D)._pyomo_expr(self.Yi) >= 1)
                    )
                else:
                    raise NotImplementedError("Decide what to do " "in this case...")
            else:
                dispPrint("No solution found. Terminating populate.")
                break
        dispPrint("Identified {} solutions via populate.".format(len(Ds)))
        return Ds

    def _make_pyomo_model(self, obj_expr, sense):
        """Method to create a Pyomo concrete model object.

        This method creates a Pyomo model and also modifies several objects
        in the MatOpt framework. It creates Pyomo variable objects and
        attaches references to those variables on each of the
        MaterialDescriptors attached to the MatOptModel.

        Args:
            obj_expr (MaterialDescriptor/Expr): Material functionality to
                optimize.
            sense (int): flag to indicate the choice to minimize or maximize the
                functionality of interest.
                Choices: minimize/maximize (Pyomo constants 1,-1 respectively)

        Returns:
            (ConcreteModel) Pyomo model object.
        """
        m = makeMyPyomoBaseModel(self.canv, Atoms=self.atoms, Confs=self.confDs)
        self.Yi._pyomo_var = m.Yi
        self.Xij._pyomo_var = m.Xij
        self.Ci._pyomo_var = m.Ci
        self.Yik._pyomo_var = m.Yik
        self.Xijkl._pyomo_var = m.Xijkl
        self.Cikl._pyomo_var = m.Cikl
        self.Zic._pyomo_var = m.Zic
        for desc in self._descriptors:
            if desc.name not in ("Yik", "Yi", "Xijkl", "Xij", "Cikl", "Ci", "Zic"):
                v = Var(
                    *desc.index_sets,
                    domain=(
                        Binary if desc.binary else (Integers if desc.integer else Reals)
                    ),
                    bounds=desc._pyomo_bounds,
                    dense=False
                )
                setattr(m, desc.name, v)
                setattr(desc, "_pyomo_var", v)
        for desc in self._descriptors:
            for c, pyomo_con in enumerate(desc._pyomo_cons(m)):
                setattr(m, "Assign{}_{}".format(desc.name, c), pyomo_con)
        if sum(obj_expr.dims) == 0:
            m.obj = Objective(expr=obj_expr._pyomo_expr(index=(None,)), sense=sense)
        else:
            raise TypeError(
                "The MaterialDescriptor chosen is not supported to be an objective, please contact MatOpt "
                "developer for potential fix"
            )
        # NOTE: The timing of the call to addConsForGeneralVars is important
        #       We need to call it after all user-defined descriptors are
        #       encoded.
        #       Else, lots of constraints for basic variables that are not
        #       necessary will be written.
        addConsForGeneralVars(m)
        for desc in self._descriptors:
            for r in desc.rules:
                if isinstance(r, FixedTo):
                    desc._fix_pyomo_var_by_rule(r, m)
        return m

    def __solve_pyomo_model(self, tee, disp, keepfiles, tilim, trelim, solver):
        """Method to solve the formulated Pyomo optimization model.

        This function is intended to standardize the printout and
        approach for converting results into Designs that is used
        by the optimize and populate methods.

        Args:
            tee (bool): Flag to turn on solver output.
            disp (int): Flag to control level of MatOpt output.
            keepfiles (bool): Flag to save temporary pyomo files.
            tilim (float): Solver time limit (in seconds).
            trelim (float): Solver tree memory limit (in MB).
            solver (str): Solver choice. Currently only cplex or
                neos-cplex are supported

        Returns:
            (Design) The best design identified by the solver, if any.
                The quality of the solution (optimal vs best found at
                termination) can be found by reading the output display.
                In the case that the model was infeasible or no solution
                could be identified, the method returns 'None'.
        """
        if solver == "cplex":
            opt = SolverFactory("cplex")
            opt.options["mip_tolerances_absmipgap"] = 0.0
            opt.options["mip_tolerances_mipgap"] = 0.0
            if tilim is not None:
                opt.options["timelimit"] = tilim
            if trelim is not None:
                opt.options["mip_limits_treememory"] = trelim
            res = opt.solve(
                self._pyomo_m, tee=tee, symbolic_solver_labels=True, keepfiles=keepfiles
            )
        elif solver == "neos-cplex":
            with SolverManagerFactory("neos") as manager:
                opt = SolverFactory("cplex")
                opt.options["absmipgap"] = 0.0  # NOTE: different option names
                opt.options["mipgap"] = 0.0
                if tilim is not None:
                    opt.options["timelimit"] = tilim
                if trelim is not None:
                    opt.options["treememory"] = trelim
                try:
                    res = manager.solve(self._pyomo_m, opt=opt)
                except Exception as e:
                    print(e)
        else:
            raise NotImplementedError(
                "MatOpt is tailored to perform best with CPLEX (locally or through NEOS), "
                "please contact MatOpt developer for additional solver support "
            )
        solver_status = res.solver.status
        solver_term = res.solver.termination_condition
        soln_status = res.solution.status
        has_solution = (
            soln_status == SolutionStatus.optimal
            or soln_status == SolutionStatus.feasible
            or soln_status == SolutionStatus.bestSoFar
            or soln_status == SolutionStatus.globallyOptimal
            or soln_status == SolutionStatus.locallyOptimal
        )
        # NOTE: The block below is a hack to get around the fact that Pyomo
        #       solution statuses are not flagged correctly all the time. If
        #       solution status was unknown (but actually optimal or feasible)
        #       then this should (hopefully) flag the solution as available
        if soln_status == SolutionStatus.unknown:
            value(self._pyomo_m.obj)
            has_solution = True

        def dispPrint(*args):
            if disp > 0:
                print(*args)
            else:
                pass

        if solver_status == SolverStatus.ok:
            dispPrint("The solver exited normally.")
            if (
                solver_term == TerminationCondition.optimal
                or solver_term == TerminationCondition.locallyOptimal
                or solver_term == TerminationCondition.globallyOptimal
            ):
                #  NOTE: This assertion should be re-enabled when Pyomo bug
                #        described above is fixed.
                # assert(soln_status==SolutionStatus.optimal)
                dispPrint("A feasible and provably optimal solution " "is available.")
            else:
                dispPrint(
                    "The solver exited due to termination criteria: {}".format(
                        solver_term
                    )
                )
                if has_solution:
                    dispPrint(
                        "A feasible (but not provably optimal) "
                        "solution is available."
                    )
                else:
                    dispPrint("No solution available.")
        else:
            dispPrint(
                "The solver did not exit normally. Status: {}".format(solver_status)
            )
        if has_solution:
            dispPrint("The Design has objective: {}".format(value(self._pyomo_m.obj)))
            result = Design(self.canv)
            setDesignFromModel(result, self._pyomo_m)
            return result
        else:
            return None
