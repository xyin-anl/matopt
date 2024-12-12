from abc import abstractmethod
from pyomo.environ import *
from matopt.aml.expr import IndexedElem, LinearExpr
from matopt.aml.util import getLB, getUB


class DescriptorRule(IndexedElem):
    """An abstract base class for rules to define material descriptors.

    This class is only the abstract interface for other well-defined rules.
    Rules get attached to MaterialDescriptor objects and are intended to be
    interpretable in relation to the descriptor that they are attached to.

    Examples:
        'Coordination number is equal to the sum of variables for the presence
            of any bond to the neighbors of a site.'
            ->  m.CNi.rules.append(EqualTo(SumNeighborBonds(m.Bondij)))
       'Size of a cluster is equal to the number of of atoms'
           ->  m.ClusterSize.rules.append(EqualTo(SumSites(m.Yi)))

    Rules can be applied over a subset of the design space (i.e., only for
    some combinations of sites, site-types, etc.) by providing keywords
    that are passed to the constructor of IndexedElem. Else, the relevant
    indexes are inferred from the rule components.

    Attributes:
        (index information inherited from IndexedElem)
    """

    def __init__(self, **kwargs):
        """Standard constructor for DescriptorRule base class.

        Args:
            **kwargs: Optional, index information passed to IndexedElem if
                interested in a subset of indices
                Possible choices: sites, bonds, site_types, bond_types, confs.
        """
        IndexedElem.__init__(self, **kwargs)

    @abstractmethod
    def _pyomo_cons(self, var):
        """Abstract method to define necessary interface for descriptor rules.

        Args:
        var (MaterialDescriptor): Variable that the rule is attached to.
            (i.e., the variable that should be read before the class name to
            interpret the rule)

        Returns:
        (list<Constraint>) list of Pyomo constraint objects.
        """
        raise NotImplementedError


class SimpleDescriptorRule(DescriptorRule):
    """An base class for simple rules with a left and right hand side.

    This class is just intended to create a common interface for the
    EqualTo, LessThan, and GreaterThan rules.

    Attributes:
        expr (Expr): Right-hand side expression for the rule.
            (index information inherited from IndexedElem)
    """

    # === STANDARD CONSTRUCTOR
    def __init__(self, e, **kwargs):
        """Standard constructor for simple descriptor rules.

        Args:
            e (float/int/Expr): An expression to use are right hand side of
                a rule. If a float or int, a LinearExpr is created for the user.
            **kwargs: Optional, index information passed to IndexedElem if
                interested in a subset of indices
                Possible choices: sites, bonds, site_types, bond_types, confs.
        """
        self.expr = LinearExpr(offset=e) if type(e) is float or type(e) is int else e
        kwargs = {**self.expr.index_dict, **kwargs}
        DescriptorRule.__init__(self, **kwargs)

    # === PROPERTY EVALUATION METHODS
    def _pyomo_cons(self, var):
        """Method to create a Pyomo constraint from this rule.

        Args:
        var (MaterialDescriptor): The descriptor to be defined by this rule.

        Returns:
        (list<Constraint>) list of Pyomo constraint objects.
        """
        ConIndexes = IndexedElem.fromComb(var, self)
        return [Constraint(*ConIndexes.index_sets, rule=self._pyomo_rule(var))]

    def _pyomo_rule(self, LHS, operator, RHS):
        """Method to create a function for a Pyomo constraint rule.

        Args:
            LHS (MaterialDescriptor/Expr): The left hand side of a simple rule.
            operator (function): The relationship to encode in a rule.
            RHS (MaterialDescriptor/Expr): The right hand side of a simple rule.

        Returns:
            (function) A function interpretable by Pyomo for a 'rule' argument
        """
        ConIndexes = IndexedElem.fromComb(LHS, RHS)

        def rule(m, *args):
            LHS_index = LHS.mask(args, ConIndexes)
            RHS_index = RHS.mask(args, ConIndexes)
            return operator(LHS._pyomo_expr(LHS_index), RHS._pyomo_expr(RHS_index))

        return rule


class LessThan(SimpleDescriptorRule):
    """A class for rules implementing 'less than or equal to' an expression.

    Spelled out: 'the descriptor is less than or equal to a linear expression'

    Attributes:
        expr (Expr): Right-hand side expression for the rule.
            (index information inherited from IndexedElem)

    See DescriptorRule for more information.
    """

    # === STANDARD CONSTRUCTOR
    # --- Inherited from SimpleDescriptorRule ---

    # === PROPERTY EVALUATION METHODS
    def _pyomo_rule(self, desc):
        """Method to create a function for a Pyomo constraint rule.

        Args:
            desc (MaterialDescriptor/Expr): A descriptor to define as 'less than'
                the expression for this rule.

        Returns:
            (function) A function in the format of a Pyomo rule to construct a
            constraint.
        """

        def less_than(LHS, RHS):
            return LHS <= RHS

        return SimpleDescriptorRule._pyomo_rule(self, desc, less_than, self.expr)


class EqualTo(SimpleDescriptorRule):
    """A class for rules implementing 'equal to' an expression.

    Spelled out: 'the descriptor is equal to a linear expression'

    Attributes:
        expr (Expr): Right-hand side expression for the rule.
            (index information inherited from IndexedElem)

    See DescriptorRule for more information.
    """

    # === STANDARD CONSTRUCTOR
    # --- Inherited from SimpleDescriptorRule ---

    # === PROPERTY EVALUATION METHODS
    def _pyomo_rule(self, desc):
        """Method to create a function for a Pyomo constraint rule.

        Args:
        desc (MaterialDescriptor/Expr): A descriptor to define as 'equal to'
            the expression for this rule.

        Returns:
        (function) A function in the format of a Pyomo rule to construct a
            constraint.
        """

        def equal_to(LHS, RHS):
            return LHS == RHS

        return SimpleDescriptorRule._pyomo_rule(self, desc, equal_to, self.expr)


class GreaterThan(SimpleDescriptorRule):
    """A class for rules implementing 'greater than or equal to' an expr.

    Spelled out: 'descriptor is greater than or equal to a linear expression'

    Attributes:
        expr (Expr): Right-hand side expression for the rule.
            (index information inherited from IndexedElem)

    See DescriptorRule for more information.
    """

    # === STANDARD CONSTRUCTOR
    # --- Inherited from SimpleDescriptorRule ---

    # === PROPERTY EVALUATION METHODS
    def _pyomo_rule(self, desc):
        """Method to create a function for a Pyomo constraint rule.

        Args:
            desc (MaterialDescriptor/Expr): A descriptor to define as 'greater
                than' the expression for this rule.

        Returns:
            (function) A function in the format of a Pyomo rule to construct a
            constraint.
        """

        def greater_than(LHS, RHS):
            return LHS >= RHS

        return SimpleDescriptorRule._pyomo_rule(self, desc, greater_than, self.expr)


class FixedTo(DescriptorRule):
    """A class for rules that fix descriptors to required values.

    Spelled out: 'the descriptor is fixed to a scalar value'

    Attributes:
        val (float): the value that the descriptor is fixed to.
            (index information inherited from IndexedElem)

    See DescriptorRule for more information.
    """

    # === STANDARD CONSTRUCTOR
    def __init__(self, val, **kwargs):
        """Standard constructor for FixedTo rules.

        Args:
            val (float): The value to fix descriptors to.
            **kwargs: Optional, index information passed to IndexedElem if
                interested in a subset of indices
                Possible choices: sites, bonds, site_types, bond_types, confs.
        """
        self.val = val
        DescriptorRule.__init__(self, **kwargs)

    def _pyomo_cons(self, var):
        """Method to create a Pyomo constraint from this rule.

        Args:
        var (MaterialDescriptor): The descriptor to be defined by this rule.

        Returns:
        (list<Constraint>) list of Pyomo constraint objects.
        """
        # NOTE: This method is used to ensure that basic variables that
        #       are fixed get referenced to write basic constraints in
        #       the model. Don't make constraints, but do instantiate
        #       variables
        Comb = IndexedElem.fromComb(var, self)
        for k in Comb.keys():
            var._pyomo_var[k]
        return []


class Disallow(DescriptorRule):
    """A class for rules that disallow a previously-identified design.

    Spelled out: 'the descriptors must attain a different solution than
        a given design'

    Attributes:
        D (Design): the design from which variable values to disallow are inferred
            (index information inherited from IndexedElem)

    See DescriptorRule for more information.
    """

    # === STANDARD CONSTRUCTOR
    def __init__(self, D, **kwargs):
        """Standard constructor for the Disallow rule.

        Args:
            D (Design): A design object to make infeasible in the resulting model
            **kwargs: Optional, index information passed to IndexedElem if
                interested in a subset of indices
                Possible choices: sites, bonds, site_types, bond_types, confs.
        """
        self.D = D
        DescriptorRule.__init__(self, **kwargs)

    def _pyomo_expr(self, var):
        """Method to create the integer cut for this disallowed design.

        Args:
            var (MaterialDescriptor): The descriptor to be defined by this rule.

        Returns:
            An instance of a Pyomo expression.
        """
        if var.name == "Yi":
            result = 0
            for i in range(len(self.D.Canvas)):
                if self.D.Contents[i] is None:
                    result += var._pyomo_var[i]
                else:
                    result += 1 - var._pyomo_var[i]
        elif var.name == "Yik":
            result = 0
            for i in range(len(self.D.Canvas)):
                for k in self.D.NonVoidElems:
                    if self.D.Contents[i] is not k:
                        result += var._pyomo_var[i, k]
                    else:
                        result += 1 - var._pyomo_var[i, k]
        else:
            # NOTE: This rule was intended to disallow structures
            #       or labelings of structures (i.e., Yi or Yik
            #       variables). It is not clear how we can generally
            #       disallow any general combination of variables.
            raise ValueError("Decide what to do in this case...")
        return result

    def _pyomo_cons(self, var):
        """
        Method to create a Pyomo constraint from this rule.

        Args:
            var (MaterialDescriptor): The descriptor to be defined by this rule.

        Returns:
            (list<Constraint>) list of Pyomo constraint objects.
        """
        return Constraint(expr=(self._pyomo_expr(var) >= 1))


class PiecewiseLinear(DescriptorRule):
    """A class for rules implementing 'equal to a piecewise linear function'.

    Spelled out: 'the descriptor is equal to a piecewise linear expression'

    Note: Innequalities of 'less than' or 'greater than' a piecewie function
    can be achieved by introducing an auxiliary descriptor to be equal to the
    piecewise function. Then, inequalities can be introduced using the
    auxiliary descriptor. Alternatively, users can modify the con_type
    attribute that is interpreted by Pyomo.

    Attributes:
        values (list<float>): values of univariate piecewise linear function at
            each breakpoint.
        breakpoints (list<float>): breakpoints of the piecewise linear function.
        input_desc (MaterialDescriptor): descriptor as the argument to the
            piecewise linear function
        con_type (string): indicates the bound type of the piecewise function.
            Options:
            “UB” - relevant descriptor is bounded above by piecewise function.
            “LB” - relevant descriptor is bounded below by piecewise function.
            “EQ” - relevant descriptor is equal to the piecewise function. (Default)

    See DescriptorRule for more information.
    """

    # === STANDARD CONSTRUCTOR
    def __init__(self, values, breakpoints, input_desc, con_type="EQ", **kwargs):
        """Standard constructor for simple descriptor rules.

        Args:
            values (list<float>): values of the function.
            breakpoints (list<float>): breakpoints of the function.
            input_desc (MaterialDescriptor): argument to the function
            con_type (string): Optional, indicates the bound type of the
                piecewise function
                Options:
                    “UB” - bounded above by piecewise function.
                    “LB” - bounded below by piecewise function.
                    “EQ” - equal to the piecewise function. (Default)
            **kwargs: Optional, index information passed to IndexedElem if
                interested in a subset of indices
                Possible choices: sites, bonds, site_types, bond_types, confs.
        """
        self.values = values
        self.breakpoints = breakpoints
        self.input_desc = input_desc
        self.con_type = con_type.upper()
        DescriptorRule.__init__(self, **kwargs)

    # === PROPERTY EVALUATION METHODS
    def _pyomo_cons(self, var):
        """Method to create a Pyomo constraint from this rule.

        Args:
            var (MaterialDescriptor): The descriptor to be defined by this rule.

        Returns:
            (list<Block>) list of Pyomo model block objects created by Piecewise
                function.
        """
        Comb = IndexedElem.fromComb(var, self)
        return [
            Piecewise(
                *Comb.index_sets,
                var._pyomo_var,
                self.input_desc._pyomo_var,
                pw_pts=self.breakpoints,
                f_rule=self.values,
                pw_constr_type=self.con_type,
                pw_repn="MC"
            )
        ]


class Implies(DescriptorRule):
    """A class for rules that define simple logical implications.

    Spelled out: 'if this descriptor is true (i.e., equal to one),
        then another set of simple rules also apply'

    Attributes:
        concs (list<tuple<MaterialDescriptor,SimpleDescriptorRule>>):
            list of conclusions to enforce if the logical predicate is true.
            (index information inherited from IndexedElem)

    See DescriptorRule for more information.
    """

    DEFAULT_BIG_M = 9999

    # === STANDARD CONSTRUCTOR
    def __init__(self, concs, **kwargs):
        """Standard constructor for Implies rule.

        Args:
            concs (list<tuple<MaterialDescriptor,SimpleDescriptorRule>>):
                list of conclusions to conditionally enforce. Also, a single
                conclusion can be provided (i.e., a tuple<MaterialDescriptor,
                SimpleDescriptorRule>) and will be placed in a list.
            **kwargs: Optional, index information passed to IndexedElem if
                interested in a subset of indices
                Possible choices: sites, bonds, site_types, bond_types, confs.
        """
        self.concs = concs if type(concs) is list else [concs]
        Comb = IndexedElem.fromComb(
            *(desc for desc, conc in self.concs), *(conc for desc, conc in self.concs)
        )
        kwargs = {**Comb.index_dict, **kwargs}
        DescriptorRule.__init__(self, **kwargs)

    def _pyomo_cons(self, var):
        """Method to create a Pyomo constraint from this rule.

        Args:
            var (MaterialDescriptor): The descriptor to be defined by this rule.

        Returns:
            (list<Constraint>) list of Pyomo constraint objects.
        """

        result = []
        for desc, conc in self.concs:
            ConIndexes = IndexedElem.fromComb(var, desc, conc)

            def rule_lb(m, *args):
                v = var._pyomo_expr(index=var.mask(args, ConIndexes))
                d = desc._pyomo_expr(index=desc.mask(args, ConIndexes))
                c = conc.expr._pyomo_expr(index=conc.expr.mask(args, ConIndexes))
                body_lb = getLB(d - c)
                MLB = body_lb if body_lb is not None else -Implies.DEFAULT_BIG_M
                return MLB * (1 - v) <= d - c

            def rule_ub(m, *args):
                v = var._pyomo_expr(index=var.mask(args, ConIndexes))
                d = desc._pyomo_expr(index=desc.mask(args, ConIndexes))
                c = conc.expr._pyomo_expr(index=conc.expr.mask(args, ConIndexes))
                body_ub = getUB(d - c)
                MUB = body_ub if body_ub is not None else Implies.DEFAULT_BIG_M
                return d - c <= MUB * (1 - v)

            if isinstance(conc, LessThan) or isinstance(conc, EqualTo):
                result.append(Constraint(*ConIndexes.index_sets, rule=rule_ub))
            if isinstance(conc, GreaterThan) or isinstance(conc, EqualTo):
                result.append(Constraint(*ConIndexes.index_sets, rule=rule_lb))
        return result


class NegImplies(DescriptorRule):
    """A class for rules that define logical implications with negation.

    Spelled out: 'if this descriptor is not true (i.e., is equal to zero),
        then another simple rule also applies'

    Attributes:
        concs (list<tuple<MaterialDescriptor,SimpleDescriptorRule>>):
            list of conclusions to enforce if the logical predicate is false.
            (index information inherited from IndexedElem)

    See DescriptorRule for more information.
    """

    DEFAULT_BIG_M = 9999

    # === STANDARD CONSTRUCTOR
    def __init__(self, concs, **kwargs):
        """Standard constructor for NegImplies rule.

        Args:
            concs (list<tuple<MaterialDescriptor,SimpleDescriptorRule>>):
                list of conclusions to conditionally enforce. Also, a single
                conclusion can be provided (i.e., a tuple<MaterialDescriptor,
                SimpleDescriptorRule>) and will be placed in a list.
            **kwargs: Optional, index information passed to IndexedElem if
                interested in a subset of indices
                Possible choices: sites, bonds, site_types, bond_types, confs.
        """
        self.concs = concs if type(concs) is list else [concs]
        Comb = IndexedElem.fromComb(
            *(desc for desc, conc in self.concs), *(conc for desc, conc in self.concs)
        )
        kwargs = {**Comb.index_dict, **kwargs}
        DescriptorRule.__init__(self, **kwargs)

    def _pyomo_cons(self, var):
        """Method to create a Pyomo constraint from this rule.

        Args:
            var (MaterialDescriptor): The descriptor to be defined by this rule.

        Returns:
            (list<Constraint>) list of Pyomo constraint objects.
        """
        result = []
        for desc, conc in self.concs:
            ConIndexes = IndexedElem.fromComb(var, desc, conc)

            def rule_lb(m, *args):
                v = var._pyomo_expr(index=var.mask(args, ConIndexes))
                d = desc._pyomo_expr(index=desc.mask(args, ConIndexes))
                c = conc.expr._pyomo_expr(index=conc.expr.mask(args, ConIndexes))
                body_lb = getLB(d - c)
                MLB = body_lb if body_lb is not None else -NegImplies.DEFAULT_BIG_M
                return MLB * (v) <= d - c

            def rule_ub(m, *args):
                v = var._pyomo_expr(index=var.mask(args, ConIndexes))
                d = desc._pyomo_expr(index=desc.mask(args, ConIndexes))
                c = conc.expr._pyomo_expr(index=conc.expr.mask(args, ConIndexes))
                body_ub = getUB(d - c)
                MUB = body_ub if body_ub is not None else NegImplies.DEFAULT_BIG_M
                return d - c <= MUB * v

            if isinstance(conc, LessThan) or isinstance(conc, EqualTo):
                result.append(Constraint(*ConIndexes.index_sets, rule=rule_ub))
            if isinstance(conc, GreaterThan) or isinstance(conc, EqualTo):
                result.append(Constraint(*ConIndexes.index_sets, rule=rule_lb))
        return result


class ImpliesSiteCombination(DescriptorRule):
    """A class for rules that define logical implications between two sites.

    Spelled out: 'if this bond-indexed descriptor is true (i.e., is equal
        to one), then a pair of simple rules hold on the two bonding sites'

    Attributes:
        canv (Canvas): the data structure to identify neighbor connections
            to apply rules over.
        concis (list<tuple<MaterialDescriptor,SimpleDescriptorRule>>):
            list of conclusions to enforce at the first site in the pair if
            the logical predicate is true.
        concjs (list<tuple<MaterialDescriptor,SimpleDescriptorRule>>):
            list of conclusions to enforce at the second site in the pair if
            the logical predicate is true.
        symmetric_bonds (bool): flag to indicate if implications should be
            applied over symmetric bond indices
            (index information inherited from IndexedElem)

    See DescriptorRule for more information.
    """

    DEFAULT_BIG_M = 9999

    # === STANDARD CONSTRUCTOR
    def __init__(self, canv, concis, concjs, symmetric_bonds=False, **kwargs):
        """Standard constructor for ImpliesSiteCombination rules.

        Args:
            canv (Canvas): the data structure to identify neighbor connections
                to apply rules over.
            concis (list<tuple<MaterialDescriptor,SimpleDescriptorRule>>):
                list of conclusions to conditionally enforce at the first
                site in a bond.
                Note: single conclusions can be provided (i.e., a
                tuple<MaterialDescriptor,SimpleDescriptorRule>) and will
                be placed in lists.
            concjs (list<tuple<MaterialDescriptor,SimpleDescriptorRule>>):
                list of conclusions to conditionally enforce at the second
                site in a bond.
                Note: single conclusions can be provided (i.e., a
                tuple<MaterialDescriptor,SimpleDescriptorRule>) and will
                be placed in lists.
            symmetric_bonds (bool): flag to indicate if a symmetric versions
                of bonds should be enumerated or if both directions should
                be included.
            **kwargs: Optional, index information passed to IndexedElem if
                interested in a subset of indices
                Possible choices: sites, bonds, site_types, bond_types, confs.
        """

        self.canv = canv
        self.concis = concis if type(concis) is list else [concis]
        self.concjs = concjs if type(concjs) is list else [concjs]
        self.symmetric_bonds = symmetric_bonds
        Combi = IndexedElem.fromComb(
            *(desc for desc, conc in self.concis), *(conc for desc, conc in self.concis)
        )
        Combj = IndexedElem.fromComb(
            *(desc for desc, conc in self.concjs), *(conc for desc, conc in self.concjs)
        )
        assert Combi.sites is not None and Combj.sites is not None
        if "bonds" not in kwargs:
            kwargs["bonds"] = [
                (i, j)
                for i in Combi.sites
                for j in canv.NeighborhoodIndexes[i]
                if (
                    j is not None
                    and j in Combj.sites
                    and (not symmetric_bonds or j > i)
                )
            ]
        if sum(Combi.dims) > 1:
            raise NotImplementedError(
                "Additional indexes are not supported, please contact MatOpt developer for "
                "possible feature addition"
            )
        DescriptorRule.__init__(self, **kwargs)

    def _pyomo_cons(self, var):
        """Method to create a Pyomo constraint from this rule.

        Args:
            var (MaterialDescriptor): The descriptor to be defined by this rule.

        Returns:
            (list<Constraint>) list of Pyomo constraint objects.
        """
        assert var.bonds is not None
        Comb = IndexedElem.fromComb(var, self)
        result = []
        # NOTE: After much confusion, I found a bug in the line of code
        #       below. Be careful not to use variable names "expr"
        #       because it gets mixed up with the Pyomo module "expr".
        #       No error, but it gives garbage expressions and wasn't
        #       clear to me what was being generated...
        # NOTE: Not clear if this was caused by module "expr" or the
        #       conflict of two local "expr,conc" objects in the two
        #       for loops...
        # for expr,conc in self.concis:
        for expri, conci in self.concis:

            def rule_i_lb(m, i, j):
                e = expri._pyomo_expr(index=(i,))
                c = conci.expr._pyomo_expr(index=(i,))
                body = e - c
                body_LB = getLB(body)
                MLBi = (
                    body_LB
                    if body_LB is not None
                    else -ImpliesSiteCombination.DEFAULT_BIG_M
                )
                return MLBi * (1 - var._pyomo_var[i, j]) <= body

            def rule_i_ub(m, i, j):
                e = expri._pyomo_expr(index=(i,))
                c = conci.expr._pyomo_expr(index=(i,))
                body = e - c
                body_UB = getUB(body)
                MUBi = (
                    body_UB
                    if body_UB is not None
                    else ImpliesSiteCombination.DEFAULT_BIG_M
                )
                return body <= MUBi * (1 - var._pyomo_var[i, j])

            if isinstance(conci, GreaterThan) or isinstance(conci, EqualTo):
                result.append(Constraint(*Comb.index_sets, rule=rule_i_lb))
            if isinstance(conci, LessThan) or isinstance(conci, EqualTo):
                result.append(Constraint(*Comb.index_sets, rule=rule_i_ub))
        # NOTE: See note above for variable name "expr"
        # for expr,conc in self.concjs:
        for exprj, concj in self.concjs:

            def rule_j_lb(m, i, j):
                e = exprj._pyomo_expr(index=(j,))
                c = concj.expr._pyomo_expr(index=(j,))
                body = e - c
                body_LB = getLB(body)
                MLBj = (
                    body_LB
                    if body_LB is not None
                    else -ImpliesSiteCombination.DEFAULT_BIG_M
                )
                return MLBj * (1 - var._pyomo_var[i, j]) <= body

            def rule_j_ub(m, i, j):
                e = exprj._pyomo_expr(index=(j,))
                c = concj.expr._pyomo_expr(index=(j,))
                body = e - c
                body_UB = getUB(body)
                MUBj = (
                    body_UB
                    if body_UB is not None
                    else ImpliesSiteCombination.DEFAULT_BIG_M
                )
                return body <= MUBj * (1 - var._pyomo_var[i, j])

            if isinstance(concj, GreaterThan) or isinstance(concj, EqualTo):
                result.append(Constraint(*Comb.index_sets, rule=rule_j_lb))
            if isinstance(concj, LessThan) or isinstance(concj, EqualTo):
                result.append(Constraint(*Comb.index_sets, rule=rule_j_ub))
        return result


class ImpliesNeighbors(DescriptorRule):
    """A class for rules that define logical implications on neighbor sites.

    Spelled out: 'if this site-indexed descriptor is true (i.e., is equal
        to one), then a set of simple rules hold on each of the neighboring
        sites'

    Attributes:
        concs (list<tuple<MaterialDescriptor,SimpleDescriptorRule>>):
            list of conclusions to enforce if the logical predicate is true.
        neighborhoods (list<list<int>>): neighborhood data structure to use
            if you do not want to use the neighborhoods of the descriptor
            that this rule is attached to.
            (index information inherited from IndexedElem)

    See DescriptorRule for more information on rules and Canvas for more
    information on 'neighborhoods'.
    """

    DEFAULT_BIG_M = 9999

    # === STANDARD CONSTRUCTOR
    def __init__(self, concs, neighborhoods=None, **kwargs):
        """Standard constructor for ImpliesNeighbors rules.

        Args:
            concs (list<tuple<MaterialDescriptor,SimpleDescriptorRule>>):
                list of conclusions to conditionally enforce. Also, a single
                conclusion can be provided (i.e., a tuple<MaterialDescriptor,
                SimpleDescriptorRule>) and will be placed in a list.
            neighborhoods (list<list<int>>) Optional, data structure to use
                as neighborhoods of interest. If not provided, then the
                neighborhoods of the descriptor that this rule is attached to
                is used.
            **kwargs: Optional, index information passed to IndexedElem if
                interested in a subset of indices
                Possible choices: sites, bonds, site_types, bond_types, confs.
        """
        self.concs = concs if type(concs) is list else [concs]
        self.neighborhoods = neighborhoods
        Comb = IndexedElem.fromComb(
            *(desc for desc, conc in self.concs), *(conc for desc, conc in self.concs)
        )
        assert Comb.sites is not None
        kwargs = {**Comb.index_dict, **kwargs}
        DescriptorRule.__init__(self, **kwargs)

    # === PROPERTY EVALUATION METHODS
    def _pyomo_cons(self, var):
        """Method to create a Pyomo constraint from this rule.

        Args:
            var (MaterialDescriptor): The descriptor to be defined by this rule.

        Returns:
            (list<Constraint>) list of Pyomo constraint objects.
        """
        var_dict_wo_s = var.index_dict
        var_dict_wo_s.pop("sites")  # no need to capture these sites
        neighborhoods = (
            self.neighborhoods
            if self.neighborhoods is not None
            else var.canv.NeighborhoodIndexes
        )
        bonds = [(i, j) for i in var.sites for j in neighborhoods[i] if j is not None]
        result = []
        # NOTE: After much confusion, I found a bug in the line of code
        #       below. Be careful not to use variable names "expr"
        #       because it gets mixed up with the Pyomo module "expr".
        #       No error, but it gives garbage expressions and wasn't
        #       clear to me what was being generated...
        # for expr,conc in self.concs:
        for expr_, conc in self.concs:
            Comb = IndexedElem.fromComb(expr_, conc)
            r_dict_wo_s = Comb.index_dict
            r_dict_wo_s.pop("sites")  # no need to capture these sites
            ConIndexes = IndexedElem.fromComb(
                IndexedElem(bonds=bonds),
                IndexedElem(**var_dict_wo_s),
                IndexedElem(**r_dict_wo_s),
            )

            def rule_lb(m, *args):
                i, j, *args = args
                v = var._pyomo_var[var.mask((i, None, *args), ConIndexes)]
                e = expr_._pyomo_expr(index=expr_.mask((j, None, *args), ConIndexes))
                c = conc.expr._pyomo_expr(
                    index=conc.expr.mask((j, None, *args), ConIndexes)
                )
                body = e - c
                body_LB = getLB(body)
                MLB = (
                    body_LB if body_LB is not None else -ImpliesNeighbors.DEFAULT_BIG_M
                )
                return MLB * (1 - v) <= body

            def rule_ub(m, *args):
                i, j, *args = args
                v = var._pyomo_var[var.mask((i, None, *args), ConIndexes)]
                e = expr_._pyomo_expr(index=expr_.mask((j, None, *args), ConIndexes))
                c = conc.expr._pyomo_expr(
                    index=conc.expr.mask((j, None, *args), ConIndexes)
                )
                body = e - c
                body_UB = getUB(body)
                MUB = body_UB if body_UB is not None else ImpliesNeighbors.DEFAULT_BIG_M
                return body <= MUB * (1 - v)

            if isinstance(conc, GreaterThan) or isinstance(conc, EqualTo):
                result.append(Constraint(*ConIndexes.index_sets, rule=rule_lb))
            if isinstance(conc, LessThan) or isinstance(conc, EqualTo):
                result.append(Constraint(*ConIndexes.index_sets, rule=rule_ub))
        return result
