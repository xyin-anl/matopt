from pyomo.environ import *
from matopt.aml.expr import IndexedElem
from matopt.aml.util import fixYik, fixYi, fixXijkl, fixXij, fixCikl, fixCi, fixZic


class MaterialDescriptor(IndexedElem):
    """A class to represent material geometric and energetic descriptors.

    This class holds the information to define mathematical optimization
    variables for the properties of materials. Additionally, each descriptor
    has a 'rules' list to which the user can append rules defining the
    descriptor and constraining the design space.

    Attributes:
        name (string): A unique (otherwise Pyomo will complain) name
        canv (``Canvas``): The canvas that the descriptor will be indexed over
        atoms (list<``BBlock``>): The building blocks to index the descriptor over.
        confDs (list<``Design``>): The designs for conformations to index over.
        integer (bool): Flag to indicate if the descriptor takes integer values.
        binary (bool): Flag to indicate if the descriptor takes boolean values.
        rules (list<``DescriptorRules``>): List of rules to define and constrain
            the material descriptor design space.
        bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
            descriptor values across all indices. If dict, the bounds can be
            individually set for each index.

    See ``IndexedElem`` for more information on indexing.
    See ``DescriptorRule`` for information on defining descriptors.
    """

    DBL_TOL = 1e-5

    # === STANDARD CONSTRUCTOR
    def __init__(
        self,
        name,
        canv=None,
        atoms=None,
        confDs=None,
        bounds=(None, None),
        integer=False,
        binary=False,
        rules=None,
        **kwargs
    ):
        """Standard constructor for material descriptors.

        Note: It is generally not necessary for users to create
              MaterialDescriptors themselves. Instead, use the
              MatOptModel.add____Descriptor() methods for the right
              type of descriptor (i.e., Site, Bond, etc.).

        Args:
        name (string): A unique (otherwise Pyomo will complain) name
        canv (Canvas): The canvas that the descriptor will be indexed over
        atoms (list<BBlock>): Building blocks to index the descriptor over.
        confDs (list<Design>): The designs for conformations to index over.
        bounds (tuple/dict/func): If tuple, the lower and upper bounds on the
            descriptor values across all indices. If dict, the bounds can be
            individually set for each index. Otherwise, advanced users can
            specify a function to be interpreted by Pyomo.
        integer (bool): Flag to indicate if the descriptor is integer.
        binary (bool): Flag to indicate if the descriptor is boolean.
        rules (list<DescriptorRules>): List of rules to define and constrain
            the material descriptor design space.
        **kwargs: Optional, index information passed to IndexedElem if
            interested in a subset of indices.
            Possible choices: sites, bonds, site_types, bond_types, confs.
        """
        self._name = name
        self._canv = canv
        self._atoms = atoms
        self._confDs = confDs
        self._integer = integer or binary
        self._binary = binary
        if rules is None:
            rules = []
        self._rules = rules if type(rules) is list else [rules]
        self._bounds = bounds
        self._pyomo_var = None  # Will be set by MatOptModel._make_pyomo_model
        IndexedElem.__init__(self, **kwargs)

    # === AUXILIARY METHODS
    def _fix_pyomo_var_by_rule(self, r, m):
        if self.name in ("Yik", "Yi", "Xijkl", "Xij", "Cikl", "Ci", "Zic"):
            return self.__fix_basic_pyomo_vars_by_rule(r, m)
        else:
            Comb = IndexedElem.fromComb(self, r)
            for k in Comb.keys():
                self._pyomo_var[k].fix(r.val)

    def __fix_basic_pyomo_vars_by_rule(self, r, m):
        Comb = IndexedElem.fromComb(self, r)
        if self.name == "Yik":
            for i in Comb.sites:
                for k in Comb.site_types:
                    fixYik(m, i, k, r.val)
        elif self.name == "Yi":
            for i in Comb.sites:
                fixYi(m, i, r.val)
        elif self.name == "Xijkl":
            for i, j in Comb.bonds:
                for k, l in Comb.bond_types:
                    fixXijkl(m, i, j, k, l, r.val)
        elif self.name == "Xij":
            for i, j in Comb.bonds:
                fixXij(m, i, j, r.val)
        elif self.name == "Cikl":
            for i in Comb.sites:
                for k, l in Comb.bond_types:
                    fixCikl(m, i, k, l, r.val)
        elif self.name == "Ci":
            for i in Comb.sites:
                fixCi(m, i, r.val)
        elif self.name == "Zic":
            for i in Comb.sites:
                for c in Comb.confs:
                    fixZic(m, i, c, r.val)

    # === PROPERTY EVALUATION METHODS
    def _pyomo_cons(self, m):
        """Create a list of Pyomo constraints related to this descriptor."""
        result = []
        for rule in self.rules:
            if rule is not None:
                result.extend(rule._pyomo_cons(self))
        return result

    @property
    def _pyomo_bounds(self):
        """Creates a bound rule/tuple that can interpreted by Pyomo."""
        if type(self.bounds) is tuple:
            return self.bounds
        elif type(self.bounds) is dict:

            def rule_gen(m, *args):
                if args is not None and len(args) == 1:
                    args = args[0]
                return self.bounds[args]

            return rule_gen
        else:
            # Else, assume that the user knows what they're doing
            # with functions for pyomo bounds
            return self.bounds

    def _pyomo_expr(self, index=None):
        """Interprets a variable as a Pyomo expression.

        Note: This is just necessary so that we can conveniently interpret
            MaterialDescriptor objects in place of Expr objects.
        """
        return self._pyomo_var[index]

    @property
    def values(self):
        """Creates a dictionary of descriptor values after optimization.

        Note: Uses the Pyomo 'value' function and only works after the
            optimization of a model.

        Returns:
            (dict) Dictionary of keys to values after optimization.
        """
        return {index: value(self._pyomo_var[index]) for index in self.keys()}

    # === BASIC QUERY METHODS
    @property
    def name(self):
        return self._name

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
    def bounds(self):
        return self._bounds

    @property
    def integer(self):
        return self._integer

    @property
    def binary(self):
        return self._binary

    @property
    def continuous(self):
        return not self.integer

    @property
    def rules(self):
        return self._rules
