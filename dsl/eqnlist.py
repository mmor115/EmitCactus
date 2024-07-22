from functools import cached_property
from typing import TypeVar, Literal, List, Dict, Union, Tuple, Any, Set, Generic, Iterator, Optional, Type, cast, Any

from sympy.core.symbol import Symbol
from sympy.core.expr import Expr
from sympy import symbols, Function, diff, IndexedBase
from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sympy.core.function import UndefinedFunction as UFunc
from enum import Enum

import dsl.use_indices
from util import OrderedSet

from dsl.sympywrap import *
from emit.ccl.schedule.schedule_tree import IntentRegion


class EqnList:
    """
    This class models a generic list of equations. As such, it knows nothing about the rest of NRPy+.
    Ultimately, the information in this class will be used to generate a loop to be output by NRPy+.
    All it knows are the following things:
    (1) params - These are quantities that are generated outside the loop.
    (2) inputs - These are quantities which are read by equations but never written by them.
    (3) outputs - These are quantities which are written by equations but never read by them.
    (4) equations - These relate inputs to outputs. These may contain temporary variables, i.e.
                    quantities that are both read and written by equations.

    This class can remove equations and parameters that are not needed, but will complain
    about inputs that are not needed. It can also detect errors in the classification of
    symbols as inputs/outputs/params.
    """

    def __init__(self, thorn_def: "dsl.use_indices.ThornDef") -> None:
        self.eqns: Dict[Math, Expr] = dict()
        self.params: Set[Math] = OrderedSet()
        self.inputs: Set[Math] = OrderedSet()
        self.outputs: Set[Math] = OrderedSet()
        self.order: List[Math] = list()
        self.verbose = True
        self.read_decls: Dict[Math, IntentRegion] = dict()
        self.write_decls: Dict[Math, IntentRegion] = dict()
        # TODO: need a better default
        self.default_read_write_spec: IntentRegion = IntentRegion.Everywhere  #Interior
        self.thorn_def: "dsl.use_indices.ThornDef" = thorn_def
        self.temporaries: Set[Math] = OrderedSet()

    @cached_property
    def variables(self) -> Set[Math]:
        return self.inputs | self.outputs | self.temporaries

    def add_param(self, lhs: Symbol) -> None:
        assert lhs not in self.outputs, f"The symbol '{lhs}' is alredy in outputs"
        assert lhs not in self.inputs, f"The symbol '{lhs}' is alredy in outputs"
        self.params.add(lhs)

    def add_input(self, lhs: Math) -> None:
        assert lhs not in self.outputs, f"The symbol '{lhs}' is alredy in outputs"
        assert lhs not in self.params, f"The symbol '{lhs}' is alredy in outputs"
        self.inputs.add(lhs)

    def add_output(self, lhs: Math) -> None:
        assert lhs not in self.inputs, f"The symbol '{lhs}' is alredy in outputs"
        assert lhs not in self.params, f"The symbol '{lhs}' is alredy in outputs"
        self.outputs.add(lhs)

    def add_eqn(self, lhs: Math, rhs: Expr) -> None:
        assert lhs not in self.eqns, f"Equation for '{lhs}' is already defined"
        self.eqns[lhs] = rhs

    def diagnose(self) -> None:
        """ Discover inconsistencies and errors in the param/input/output/equation sets. """
        needed: Set[Math] = OrderedSet()
        complete: Dict[Math, int] = dict()
        self.order = list()

        read: Set[Math] = OrderedSet()
        written: Set[Math] = OrderedSet()

        self.read_decls.clear()
        self.write_decls.clear()

        self.lhs: Math

        def ftrace(sym: Symbol) -> bool:
            if sym.is_Function:
                if self.thorn_def.is_stencil.get(sym.func, False):
                    for arg in sym.args:
                        sym2 = cast(Symbol, arg)
                        self.read_decls[sym2] = IntentRegion.Everywhere
                        self.write_decls[self.lhs] = IntentRegion.Interior
                else:
                    for arg in sym.args:
                        sym2 = cast(Symbol, arg)
                        self.read_decls[sym2] = IntentRegion.Everywhere
                        self.write_decls[self.lhs] = IntentRegion.Everywhere
            return False

        def noop(x: Symbol) -> Symbol:
            return x

        for lhs in self.eqns:
            self.lhs = lhs
            rhs = self.eqns[lhs]
            rhs.replace(ftrace, noop)  # type: ignore[no-untyped-call]

        for arg in self.inputs:
            if arg not in self.read_decls:
                self.read_decls[arg] = self.default_read_write_spec

        for lhs in self.eqns:
            rhs = self.eqns[lhs]
            print(colorize("EQN:", "cyan"), lhs, colorize("=", "cyan"), rhs)

        if self.verbose:
            print(colorize("Inputs:", "green"), self.inputs)
            print(colorize("Outputs:", "green"), self.outputs)
            print(colorize("Params:", "green"), self.params)

        for k in self.eqns:
            written.add(k)
            for q in finder(self.eqns[k]):
                read.add(q)

        if self.verbose:
            print(colorize("Read:", "green"), read)
            print(colorize("Written:", "green"), written)

        for k in self.inputs:
            assert k in read, f"Symbol '{k}' is in inputs, but it is never read. {read}"
            assert k not in written, f"Symbol '{k}' is in inputs, but it is assigned to."

        for k in self.outputs:
            assert k in written, f"Symbol '{k}' is in outputs, but it is never written"

        for k in written:
            if k not in self.outputs:
                self.temporaries.add(k)

        for k in read:
            if k not in self.inputs and k not in self.params:
                self.temporaries.add(k)

        if self.verbose:
            print(colorize("Temps:", "green"), self.temporaries)

        for k in self.temporaries:
            assert k in read, f"Temporary variable '{k}' is never read"
            assert k in written, f"Temporary variable '{k}' is never written"

        for k in read:
            assert k in self.inputs or self.params or self.temporaries, f"Symbol '{k}' is read, but it is not a temp, parameter, or input."

        for k in self.outputs:
            # The outputs are all needed
            # for a successful computation
            needed.add(k)
        for k in self.inputs:
            complete[k] = 0

        again = True
        generation = 0
        find_cycle: Optional[Math] = None
        while True:
            if not again and len(needed) > len(complete) and find_cycle is not None:
                again = True
                # The symbol in find_cycle is not complete
                # but we assert that it is to help us find
                # symbols that are part of a cycle.
                complete[find_cycle] = generation
            find_cycle = None
            if not again:
                break
            generation += 1
            again = False
            for k in list(needed):
                if (k in self.inputs or k in self.params) and k not in complete:
                    complete[k] = generation
                    again = True
                elif k not in complete:
                    assert k in self.eqns, f"Symbol '{k}' is needed but is not written"
                    v = self.eqns[k]
                    can_add = True
                    for k2 in finder(v):
                        if k2 not in complete:
                            # A variable is only complete
                            # if all its free symbols are
                            # complete
                            can_add = False
                        if k2 not in needed:
                            # Since k2 is needed to assign
                            # a value to k, k2 is also needed
                            needed.add(k2)
                            # more work to do
                            again = True
                    if can_add:
                        complete[k] = generation
                        self.order.append(k)
                        again = True
                    else:
                        find_cycle = k
        print(colorize("Order:", "green"), self.order)

        default_read_spec = None
        default_write_spec = None
        for var in self.inputs:
            for val in self.read_decls.values():
                if default_read_spec is not None:
                    assert default_read_spec == val
                default_read_spec = val
        for var in self.outputs:
            for val in self.write_decls.values():
                if default_write_spec is not None:
                    assert default_write_spec == val
                default_write_spec = val

        # Figure out the rest of the READ/WRITEs
        spec: IntentRegion
        for var in self.order:
            if var in self.inputs and var not in self.read_decls:
                if default_read_spec is None:
                    default_read_spec = IntentRegion.Everywhere
                self.read_decls[var] = default_read_spec
            elif var in self.outputs and var not in self.write_decls:
                if default_write_spec is None:
                    default_write_spec = IntentRegion.Everywhere
                spec = default_write_spec
                # if there are variables only valid in the interior on the RHS of the
                # equation where the var is assigned, then it must have spec = Interior
                for rvar in finder(self.eqns[var]):
                    if self.read_decls.get(rvar, IntentRegion.Everywhere) == IntentRegion.Interior:
                        spec = IntentRegion.Interior
                        break
                self.write_decls[var] = spec
            elif var in self.temporaries and var in self.write_decls and var not in self.read_decls:
                # temporaries don't really have reads/writes, we figure out
                # what they would be just to connect the reads/writes of the inputs/outputs.
                self.read_decls[var] = self.write_decls[var]
            elif var in self.temporaries and var not in self.write_decls:
                if default_write_spec is None:
                    default_write_spec = IntentRegion.Everywhere
                spec = default_write_spec
                if spec != IntentRegion.Interior:
                    for rvar in finder(self.eqns[var]):
                        if self.read_decls.get(rvar, IntentRegion.Everywhere) == IntentRegion.Interior:
                            spec = IntentRegion.Interior
                            break
                self.write_decls[var] = spec
                self.read_decls[var] = spec

        if self.verbose:
            print(colorize("READS:", "green"), end="")
            for var, spec in self.read_decls.items():
                if var in self.inputs:
                    print(" ", var, "=", colorize(repr(spec), "yellow"), sep="", end="")
            print()
            print(colorize("WRITES:", "green"), end="")
            for var, spec in self.write_decls.items():
                if var in self.outputs:
                    print(" ", var, "=", colorize(repr(spec), "yellow"), sep="", end="")
            print()

        for k, v in self.eqns.items():
            assert k in complete, f"Eqn '{k} = {v}' does not contribute to the output."
            val1: int = complete[k]
            for k2 in finder(v):
                val2: int = complete[cast(Symbol, k2)]
                assert val1 >= val2, f"Symbol '{k}' is part of an assignment cycle."
        for k in needed:
            if k not in complete:
                print(f"Symbol '{k}' needed but could not be evaluated. Cycle in assignment?")
        for k in self.inputs:
            assert k in complete, f"Symbol '{k}' appears in inputs but is not complete"
        for k in self.eqns:
            assert k in complete, f"Equation '{k} = {self.eqns[k]}' is never complete"

    def trim(self) -> None:
        """ Remove temporaries of the form "a=b". They are clutter. """
        subs: Dict[Math, Symbol] = dict()
        for k, v in self.eqns.items():
            if v.is_symbol:
                # k is not not needed
                subs[k] = cast(Symbol, v)
                print(f"Warning: equation '{k} = {v}' can be trivially eliminated")

        new_eqns: Dict[Math, Expr] = dict()
        for k in self.eqns:
            if k not in subs:
                v = self.eqns[k]
                v2 = do_subs(v, subs)
                new_eqns[k] = v2

        self.eqns = new_eqns

    def cse(self) -> None:
        """ Invoke Sympy's CSE method, but ensure that the order of the resulting assignments is correct. """
        indexes: List[Math] = list()
        old_eqns: List[Expr] = list()
        for k in self.eqns:
            indexes.append(k)
            old_eqns.append(self.eqns[k])
        new_eqns, mod_eqns = cse(old_eqns)
        e: Tuple[Symbol, Expr]
        for e in new_eqns:
            assert e[0] not in self.inputs and e[0] not in self.params and e[0] not in self.eqns
            self.add_eqn(e[0], e[1])
        for i in range(len(indexes)):
            k = indexes[i]
            v = old_eqns[i]
            m = mod_eqns[i]
            self.eqns[k] = m

    def dump(self) -> None:
        print(colorize("Dumping Equations:", "green"))
        for k in self.order:
            print(" ", colorize(k, "cyan"), "=", self.eqns[k])


if __name__ == "__main__":
    a, b, c, d, e, f, g, q, r = symbols("a b c d e f g q r")
    try:
        div = mkFunction("div")
        el = EqnList(dsl.use_indices.ThornDef('Foo', 'Bar'))
        el.default_read_write_spec = IntentRegion.Interior
        # el.add_func(div, True)
        el.add_input(a)
        el.add_eqn(c, sympify(3))
        el.add_eqn(b, 2 * c + sympify(8))
        el.add_eqn(f, div(a) + c)
        el.add_eqn(d, b + c + f)
        el.add_output(c)
        el.add_output(d)

        el.diagnose()
        assert el.read_decls[a] == IntentRegion.Everywhere
        assert el.write_decls[d] == IntentRegion.Interior
        assert el.write_decls[c] == IntentRegion.Everywhere

        el.default_read_write_spec = IntentRegion.Everywhere
        # el.add_func(div, True)
        el.diagnose()
        assert el.read_decls[a] == IntentRegion.Everywhere
        assert el.write_decls[d] == IntentRegion.Interior
        assert el.write_decls[c] == IntentRegion.Everywhere

        el.default_read_write_spec = IntentRegion.Everywhere
        # el.add_func(div, False)
        el.diagnose()
        assert el.read_decls[a] == IntentRegion.Everywhere
        assert el.write_decls[d] == IntentRegion.Everywhere
        assert el.write_decls[c] == IntentRegion.Everywhere
    finally:
        print("Test zero passed")
        print()

    try:
        el = EqnList(dsl.use_indices.ThornDef('Foo', 'Bar'))
        el.add_eqn(r, q)  # cycle
        el.add_eqn(q, r)  # cycle
        el.add_eqn(a, r)
        el.add_output(a)
        el.diagnose()
    except AssertionError as ae:
        assert "cycle" in str(ae)
        print("Test one passed")
        print()

    try:
        el = EqnList(dsl.use_indices.ThornDef('Foo', 'Bar'))
        el.add_input(a)
        el.add_input(f)
        el.add_input(b)
        el.add_output(d)
        el.add_eqn(c, g + f + a ** 3 + q)
        el.add_eqn(e, a ** 2)
        el.add_eqn(g, e)
        el.add_eqn(d, c * b + a ** 3)
        el.diagnose()
    except AssertionError as ae:
        assert "'q' is never written" in str(ae)
        print("Test two passed")
        print()

    try:
        el = EqnList(dsl.use_indices.ThornDef('Foo', 'Bar'))
        el.add_input(a)
        el.add_input(f)
        el.add_output(d)
        el.add_eqn(a, f + 2)
        el.add_eqn(d, a)
        el.diagnose()
    except AssertionError as ae:
        assert "Symbol 'a' is in inputs, but it is assigned to." in str(ae)
        print("Test three passed")
        print()

    try:
        el = EqnList(dsl.use_indices.ThornDef('Foo', 'Bar'))
        el.add_input(a)
        el.add_input(f)
        el.add_output(d)
        el.add_output(e)
        el.add_eqn(d, a + f)
        el.diagnose()
    except AssertionError as ae:
        assert "Symbol 'e' is in outputs, but it is never written" in str(ae)
        print("Test four passed")
        print()
