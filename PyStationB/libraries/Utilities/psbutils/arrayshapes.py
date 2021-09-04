# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import collections
import math
import re
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple, Type, Union, cast

import numpy as np
from numpy.typing import ArrayLike

IntStr = Union[int, str]
IntOrTuple = Union[int, Tuple[int, ...]]
Bindings = Dict[str, IntOrTuple]
Checker = Callable[[Dict[str, Any]], bool]
DTypes = Optional[Union[Type[np.number], Sequence[Type[np.number]]]]


class Writable(Protocol):
    def write(self, msg):
        ...  # pragma: no cover


class Shaped(Protocol):
    @property
    def shape(self) -> Tuple[int, ...]:
        ...  # pragma: no cover


CHAIN_LENGTH_KEY = "#chain_length#"


class DummyShapes:
    """
    Dummy version of Shapes class, that also serves as parent class of Shapes. Once array shapes are debugged for a
    particular module, this class can be imported in place of Shapes (e.g. import ... DummyShapes as Shapes) for better
    efficiency without having to change all the individual "Shapes" calls.
    """

    def __init__(self, array: Optional[ArrayLike] = None, spec: Optional[str] = None, dtypes: DTypes = None):
        """
        :param array: a numpy array, or something that can be coerced to one, or that has a "shape" attribute
        :param spec: a string indicating its desired shape, as in the examples below.
        :param dtypes: a numpy dtype, such as np.int32, or a sequence of them. Then
                       the array must not only have a matching shape, it must also be of one of the indicated dtypes.
        If both array and spec are provided and are not None, the __call__ method is invoked on them. If one of them
        is None the other must be too.
        """
        self._arrays: List[ArrayLike] = []
        self._bindings: Dict[str, Any] = {}
        if array is None and spec is None:
            return
        elif array is not None and spec is not None:
            self(array, spec, dtypes)
        else:
            raise ValueError("array and spec must both be None or both be non-None")

    def __call__(self, array: ArrayLike, spec: str, dtypes: DTypes = None) -> "DummyShapes":
        """
        Arguments: as for the constructor.
        Returns: this object, so calls can be chained.
        """
        self._arrays.append(array)
        return self

    @property
    def bindings(self):
        """
        Returns a copy of the variable-value bindings without the chain-length binding.
        """
        bdgs = self._bindings.copy()
        del bdgs[CHAIN_LENGTH_KEY]
        return bdgs

    def __getitem__(self, index: int) -> ArrayLike:
        """
        Returns the index-th array passed in the chain. Thus for example
           Shapes(a0, s0)(a1, s1)(a2, s2)(a3, s3)[-1]
        will return a3.
        """
        return self._arrays[index]

    def last(self) -> ArrayLike:
        """
        Returns the last array passed in the chain. Thus for example
           Shapes(a0, s0)(a1, s1)(a2, s2)(a3, s3).last()
        """
        return self[-1]

    def where(self, is_valid: Checker, message: str = "Check failed") -> None:
        """
        :param is_valid: a function that takes the current bindings as its single argument, and returns a bool.
        A value of True means all is well; False will lead to an exception being raised with the specified message.
        Thus a suitable function could be:
            lambda bdgs: bdgs.get("X", None) in [1, 3, 5]
        """
        if not is_valid(self._bindings):
            raise ValueError(message)


class Shapes(DummyShapes):
    """
    Class to check the shapes of arrays in running code. Typical calling pattern:
       Shapes(a0, s0)(a1, s1)(a2, s2)(a3, s3)
    or equivalently
       Shapes()(a0, s0)(a1, s1)(a2, s2)(a3, s3)
    where each a_i is an arraylike object and each s_i is a string specifying array dimensions and variables that
    can be bound to those dimensions.
       Each a_i should normally be something with a "shape" attribute - e.g. a numpy array or Torch tensor. If it
    does not have a "shape", it should be convertable (and will be converted internally) to a numpy array by wrapping
    it in np.array(...).
       As each (a_i, s_i) pair is processed, the shape specification s_i is checked against the shape of a_i, with new
    variables in s_i being bound as required. If this is not possible, a ValueError is raised. For example, if
    a0.shape == (2, 3) and a1.shape == (3, 4), then:
       Shapes(a0, "X,Y")(a1, "Y,Z")
    will succeed, with X being bound to 2, Y to 3 and Z to 4. However if a1.shape == (5, 4) then a ValueError will be
    raised because Y cannot be bound to both 3 and 5.
      For full details of the specifications s_i, see "_check_shape".
    """

    # Binding dictionary class variable. Keys are strings denoting a file and line number where there is a Shapes call,
    # and values are the binding lists for those calls. This is used by shape inference - see ShapeInferrer class.
    _BDCT: Dict[str, List[Bindings]] = collections.defaultdict(list)

    def __init__(self, array: Optional[ArrayLike] = None, spec: Optional[str] = None, dtypes: DTypes = None):
        """
        Initialize as in the superclass, and add the current calling context (if any) to _BDCT.
        """
        super().__init__(array, spec, dtypes)
        cur_frame = currentframe()
        if cur_frame is not None:
            prev_frame = cur_frame.f_back
            if prev_frame is not None:
                caller_file, caller_line, _, _, _ = getframeinfo(prev_frame)
                caller_key = f"{caller_file}:{caller_line}"
                self._BDCT[caller_key].append(self._bindings)

    @classmethod
    def clear(cls) -> None:
        """
        Clear the memory of bindings
        """
        cls._BDCT = collections.defaultdict(list)

    def __call__(self, array: ArrayLike, spec: str, dtypes: DTypes = None) -> "Shapes":
        """
        If dtypes are not None, checks the type of array is one of the provided dtypes.
        Then checks that the shape of array matches spec, with bindings being added if necessary.
        If either check fails, a ValueError is returned. If both succeed, the chain length (as a special
        binding) is incremented, for possible use in shape inference.
        """
        super().__call__(array, spec, dtypes)
        arr2 = self._check_dtype(array, dtypes)
        shape = arr2.shape
        spec2 = tuple(spec.replace(" ", "").rstrip(",").split(","))
        self._check_shape(shape, spec2)
        self._increment_chain_length()
        return self

    def _increment_chain_length(self):
        """
        Increase the bound value of CHAIN_LENGTH_KEY by 1. Only binding sets with maximal chain length
        will be used for shape inference, on the assumption that non-maximal values imply an error in
        the shape.
        """
        self._bindings[CHAIN_LENGTH_KEY] = 1 + self._bindings.get(CHAIN_LENGTH_KEY, 0)

    def _check_dtype(self, array: ArrayLike, dtypes: DTypes) -> Shaped:
        dtypes2: Sequence[Type[np.number]]
        if dtypes is None:
            dtypes2 = []
        elif isinstance(dtypes, Sequence):
            dtypes2 = dtypes
        else:
            dtypes2 = [dtypes]
        if hasattr(array, "shape") and not dtypes2:
            arr2 = array
        else:
            arr2 = np.array(array)
            if dtypes2 and arr2.dtype not in dtypes2:
                raise TypeError(f"Array dtype is {arr2.dtype} and should be one of {dtypes2}")
        # Ignore type because we have logically guaranteed there is a shape attribute
        return cast(Shaped, arr2)

    def _check_shape(self, shape: Tuple[int, ...], spec: Tuple[str, ...]) -> None:
        """
        Checks that the provided shape matches the spec.

        After the removal of any space characters, the spec should conform to the following grammar, in which:
            S stands for "shape expression"
            L for "eLement of shape expression", corresponding to one or, sometimes, more than one dimension, and
            C for "component".

        S -> L
        S -> L "," S

        S should be of the form L,L,L,...,L, with an optional final comma. Spaces are ignored.

        L -> C
        L -> two or more alphabetic characters

        An eLement L is either a component C, or a "splice variable" consisting of two or more letters. A splice
        variable can match zero or more elements of the shape of the array. There can be at most one splice variable
        in a specification.

        C -> single alphabetic character
        C -> string interpretable as an integer
        C -> "(" C ")"
        C -> C Op C
        Op -> "+" | "-" | "*" | "/" | "%"

        C looks like an arithmetic expression containing single-letter variables, integers, parentheses, and any of
        the 5 indicated arithmetic operators. NOTE: expressions are evaluated from left to right; there is no notion
        of operator precedence (yet). If you want to express "A+B*C", write "A+(B*C)".

        Reading left to right, the first time a single alphabetic character is encountered in an expression, it is
        bound to the corresponding element in the shape of the array preceding the shape expression. Subsequent
        occurrences must match this value.
        """
        pairs = self._splice_zip(shape, spec)
        if pairs is None:
            raise ValueError(f"expected {spec} with {len(spec)} dimensions, got {shape} with {len(shape)} dimensions")
        spec_val_list: List[IntStr] = []
        for sz, spec_expr in pairs:
            if isinstance(spec_expr, str) and spec_expr.isalpha():
                if spec_expr not in self._bindings:
                    self._bindings[spec_expr] = sz
                to_add = self._bindings[spec_expr]
                if isinstance(to_add, tuple):
                    spec_val_list.extend(to_add)
                else:
                    spec_val_list.append(to_add)
            else:
                spec_val_list.append(self.evaluate_shape_expression(spec_expr))
        spec_val = tuple(spec_val_list)
        if shape != spec_val:
            if spec != spec_val:
                raise ValueError(f"Got {shape}, expected {spec} = {spec_val}")
            else:  # pragma: no cover
                raise ValueError(f"Got {shape}, expected {spec}")

    def _splice_zip(
        self, shape: Tuple[int, ...], spec: Tuple[IntStr, ...]
    ) -> Optional[List[Tuple[IntOrTuple, IntStr]]]:
        """
        :param shape: shape of an array
        :param spec: tuple of integers or strings
        :return: None if no match possible; otherwise a list of pairs, where each pair consists of
        (1) an element of shape or a tuple of consecutive elements of shape; and
        (2) a member of the "spec" tuple
        """
        long_vars = [
            (i, item) for i, item in enumerate(spec) if isinstance(item, str) and item.isalpha() and len(item) > 1
        ]
        excess = len(shape) - len(spec)
        if len(long_vars) == 1 and excess >= -1:
            i_splice, i_splice_var = long_vars[0]
            result: List[Tuple[IntOrTuple, IntStr]] = []
            if i_splice > 0:
                result.extend(zip(shape[:i_splice], spec[:i_splice]))
            result.append((tuple(shape[i_splice : i_splice + excess + 1]), i_splice_var))  # noqa: E203
            n_rest = len(spec) - 1 - i_splice
            if n_rest > 0:
                result.extend(zip(shape[-n_rest:], spec[-n_rest:]))
            return result
        if excess == 0:
            return list(zip(shape, spec))
        return None

    def evaluate_shape_expression(self, shape_expr: IntStr) -> IntStr:
        """
        Evaluate and return shape_expr with respect to the variable values defined in bindings, which may be
        augmented in the process. The result will be an integer if evaluation is possible, otherwise a string
        with the problems demarcated by <<angle brackets>>. Since a string cannot be part of an array shape,
        a mismatch with the actual shape of the array is guaranteed, and a ValueError will be thrown.

        shape_expr can be an integer which is returned unchanged, or a string which is evaluated.
        """
        if isinstance(shape_expr, int):
            # Return integer unchanged.
            return shape_expr
        if not isinstance(shape_expr, str):
            # We expect a string at this point; if not, the whole thing is a problem.
            return f"<<{shape_expr}>>"
        if shape_expr in self._bindings:
            # shape_expr is a (single-letter) variable which already has a value. Return the value, which
            # must be an int, not a tuple.
            return self._bindings[shape_expr]  # type: ignore
        if shape_expr.startswith("("):
            # shape_expr is (or should be) an expression in parentheses, possibly followed by more stuff.
            close = shape_expr.find(")")
            if close < 0:
                # Mismatched parentheses.
                return f"<<{shape_expr}>>"
            # Value of string inside the parentheses.
            val = self.evaluate_shape_expression(shape_expr[1:close])
            if close + 1 == len(shape_expr):
                # Just a parenthesized expression; return the value.
                return val
            else:
                # The character after the closing parenthesis should be a valid operator.
                return self._apply_operator(
                    shape_expr[close + 1], val, self.evaluate_shape_expression(shape_expr[close + 2 :])  # noqa: E203
                )
        try:
            # This will succeed if the expression is the string version of an integer.
            return int(shape_expr)
        except ValueError:
            pass
        if len(shape_expr) >= 3:
            # shape_expr should be a single-character variable (which ought to be in bindings), followed by a
            # single-character operator, followed by other stuff.
            return self._apply_operator(
                shape_expr[1],
                self.evaluate_shape_expression(shape_expr[0]),
                self.evaluate_shape_expression(shape_expr[2:]),
            )
        # We couldn't parse it, so the whole thing is a problem.
        return f"<<{shape_expr}>>"

    def _apply_operator(self, op: str, lhs: IntStr, rhs: IntStr) -> IntStr:
        """
        Applies operator "op", which should be one of "+-*/%", to lhs and rhs, and returns the result: an int
        if all is well-defined, otherwise a string showing where the problem(s) are. Problems are surrounded by
        <<angle brackets>>.
        """
        if not (isinstance(lhs, int) and isinstance(rhs, int)):
            # Problem in lhs and/or rhs; return concatenation.
            return f"{lhs}{op}{rhs}"
        if op == "+":
            return lhs + rhs
        elif op == "-":
            return lhs - rhs
        elif op == "*":
            return lhs * rhs
        elif rhs == 0:
            # Return attempted divide-by-zero as a problem.
            return f"{lhs}{op}<<{rhs}>>"
        elif op == "/":
            return int(lhs / rhs)
        elif op == "%":
            return lhs % rhs
        else:
            # "op" itself is a problem, so return concatenation with op marked as such.
            return f"{lhs}<<{op}>>{rhs}"

    @classmethod
    def infer(cls, output: Optional[Union[Writable, bool]] = None) -> List[str]:
        """
        Returns the result of Shape inference as a list of strings, each one giving the file and line number
        of a Shapes(...) call followed by inferred variable values and relationships. Typically, after running
        some code that makes Shapes(...) calls, you would call Shapes.infer() and print the results, then consider
        tightening the variable constraints in the Shapes(...) calls accordingly.

        If "output" is provided, the results are written to it. It should be either something with a "write" method,
        or for convenience, True, which is equivalent to sys.stdout.

        IMPORTANT CAVEATS:

        (1) The returned constraints should be *correct* in the sense that they are true for all the calls
            that were made in a particular context. If not, it's a bug that should be reported.
            However, they are not necessarily true for all _possible_ calls, because generalizing from finite data is
            always error prone. To minimize this problem, try to run each Shapes(...) call on which you want to do
            inference on as many different examples as possible.

        (2) The returned constraints are not, and never will be, *complete*. Only a limited number of relationships
            are exploited: basically, fixed values for single variables and linear relationships between pairs of
            variables (although the notion of "variable" is expanded somewhat - see ShapeInferrer documentation).
        """
        result: List[str] = []
        for caller, bdg_list in cls._BDCT.items():
            expr_list = ShapeInferrer(bdg_list).constraints()
            header = f"{len(expr_list)} constraints from {len(bdg_list)} sets of bindings"
            for expr in [header] + expr_list:
                result.append(f"{caller}: {expr}")
        if not output:
            pass
        elif isinstance(output, bool):  # pragma: no cover
            for line in result:
                print(line)
        else:  # pragma: no cover
            for line in result:
                output.write(line + "\n")
        return result


class ShapeInferrer:
    """
    Shape inference class, called from Shapes.infer() on the binding list from each of one or more Shapes(...) calls in
    previously-run code.
    """

    def __init__(self, bdg_list: List[Bindings]):
        """
        :param: bdg_list: the _bindings value of one previously-run Shapes(...) call.
        Sets up attributes ready for calculating constraints.
        """
        # Uniquified version of bdg_list, keeping only those that reached maximum chain length, and then extended
        # with derived variables and their values.
        self._extended_bdgs = [self._extended_bindings(bdgs) for bdgs in self._get_unique_bindings(bdg_list)]
        # All variable names, sorted so we have shortest first, breaking ties by alphabetical order. This will lead to
        # more intuitive expressions.
        if self._extended_bdgs:
            self._var_names = sorted(self._extended_bdgs[0].keys(), key=lambda key: (len(key), key))
        else:
            self._var_names = []
        # Set of variables whose value is already determined by a constraint we've found, so to avoid redundancy,
        # we won't consider it for inclusion in any other constraints.
        self._already_determined: Set[str] = set()

    @property
    def bindings(self):
        """
        Returns extended bindings - mainly for testing and diagnostics.
        """
        return self._extended_bdgs.copy()

    def constraints(self) -> List[str]:
        """
        Returns suggested variable constraints for the binding list handed to the constructor. A constraint is
        a string of the form "expr1=expr2", where expr1 is a (possibly derived) variable and expr2 is a linear
        expression in them.
        """
        if len(self._extended_bdgs) < 2:
            # If we have less than two unique sets of bindings, there is no chance of working out any relationships
            # between variables, so we propose no constraints.
            return []
        self._already_determined.clear()
        result: List[str] = []
        # Pair each variable name with its values from each member of extended_bdgs.
        value_lists = [(var, [bdgs[var] for bdgs in self._extended_bdgs]) for var in self._var_names]
        for i, (var1, vals1) in enumerate(value_lists):
            if self._is_already_determined(var1):
                pass
            elif len(set(vals1)) == 1:
                # A single value for a variable means can constrain it to be that value.
                result.append(f"{var1}={vals1[0]}")
                self._already_determined.add(var1)
            else:
                # If var1 has multiple values, then two-variable constraints are possible
                for var2, vals2 in value_lists[i + 1 :]:  # noqa
                    constraint = self._constraint_for_pair_of_variables(var1, vals1, var2, vals2)
                    if constraint is not None:
                        result.append(constraint)
        return sorted(result)

    def _is_already_determined(self, var: str) -> bool:
        """
        Returns whether "var" is already determined, so that we don't want any more constraints mentioning it.
        A variable is determined if it's already appeared on the left hand side of a constraint, or in these
        cases: (1) var is "X*Y" and either X or Y is determined; (2) var is e.g. "prod(XX)" (or another projection
        of a splice variable) and XX is determined.
        """
        if var in self._already_determined:
            return True
        # product term
        if (
            len(var) == 3
            and var[1] == "*"
            and (var[0] in self._already_determined or var[2] in self._already_determined)
        ):
            return True
        # projection term: e.g. prod(XX) when XX is already determined
        pos = var.find("(")
        if pos > 0 and var[-1] == ")" and var[pos + 1 : -1] in self._already_determined:  # noqa
            return True  # pragma: no cover
        return False

    def _constraint_for_pair_of_variables(self, var1, vals1, var2, vals2) -> Optional[str]:
        if self._is_already_determined(var2):
            return None
        if var2.find(var1) >= 0:
            # var2 is derived from var1, e.g. it's "prod(XX)"" vs "XX" or "X*Y" vs "Y", so constraint
            # is not interesting.
            return None
        if self._derived_from_same_splice_variable_and_one_is_prod(var1, var2):
            # For example: "first(XX)" and "prod(XX)"; constraint is not interesting.
            return None
        unique_vals1 = set(vals1)
        unique_vals2 = set(vals2)
        if len(unique_vals2) != len(unique_vals1):
            # short cut: no linear relationship possible if different numbers of unique values.
            return None
        val_pairs = sorted(set(zip(vals1, vals2)))
        if len(val_pairs) != len(unique_vals1):
            # no non-degenerate linear relationship possible if vals1 and vals2 do not uniquely determine
            # each other.
            return None
        if self._is_all_ints(vals1) and self._is_all_ints(vals2):
            # Ignore type clash here because is_all_ints calls above make this OK
            fit = self._linear_fit(val_pairs)
            if fit is None:
                # no linear relationship between var1 and var2
                return None
            const, grad_num, grad_den = fit
            self._already_determined.add(var2)  # because the constraint is "var2=...".
            return self._build_expression(var1, var2, const, grad_num, grad_den)
        if vals1 == vals2:  # both must be tuples; if they're always equal, specify that constraint.
            self._already_determined.add(var2)  # because the constraint is "var2=...".
            if "(" not in var2:
                # If XX is determined, so are dims(XX), etc.
                for op in ["dims", "first", "last", "prod"]:
                    self._already_determined.add(f"{op}({var2})")
            return f"{var2}={var1}"
        return None

    @staticmethod
    def _get_unique_bindings(bdg_list) -> List[Bindings]:
        """
        Given a list of bindings, filter out any for which the end of the chain of calls was not reached, indicated
        by CHAIN_LENGTH_KEY having less that its maximum value. (If all calls failed at the same point, that's OK
        for inference). Also discard any repeats, and return the remainder with CHAIN_LENGTH_KEY removed.
        """
        if not bdg_list:
            return []
        unique_bdgs: List[Bindings] = []
        # "type: ignore"s here are because we know the value of CHAIN_LENGTH_KEY should always be an int not a tuple.
        max_chain_length: int = max([bdgs.get(CHAIN_LENGTH_KEY, 0) for bdgs in bdg_list])  # type: ignore
        for bdgs in bdg_list:
            if bdgs.get(CHAIN_LENGTH_KEY, 0) < max_chain_length:  # type: ignore
                continue
            bdgs_copy = bdgs.copy()
            if bdgs_copy not in unique_bdgs:
                unique_bdgs.append(bdgs_copy)
        for bdgs in unique_bdgs:
            bdgs.pop(CHAIN_LENGTH_KEY, None)
        return unique_bdgs

    @staticmethod
    def _derived_from_same_splice_variable_and_one_is_prod(var1: str, var2: str) -> bool:
        regex = r"[a-z]+\(([A-Za-z]+)\)"
        m1 = re.match(regex, var1)
        m2 = re.match(regex, var2)
        return (
            m1
            and m2
            and (m1.group(1) == m2.group(1))
            and (var1.startswith("prod(") or var2.startswith("prod("))
            or False
        )

    @staticmethod
    def _extended_bindings(bdgs: Bindings) -> Bindings:
        """
        :param bdgs: a dictionary from variables (e.g. "X") to values (e.g. 3).
        Returns a copy of bdgs, extended with derived variables and their values.
        Derived variables and values are:
          - for any pair of distinct ordinary (single-character) variables "X" and "Y",
            their product "X*Y"
          - for any splice variable "XX", various projections from its (tuple) value to
            integers:
               "dims(XX)" -> number of dimensions in XX (length of the tuple)
               "first(XX)" -> value of first element of tuple (if tuple is not empty)
               "last(XX)" -> value of last element of tuple (if tuple is not empty)
               "prod(XX)" -> product of all elements of tuple (1 if empty)
        """
        result = {}
        vars = sorted(key for key in bdgs.keys() if key != CHAIN_LENGTH_KEY)
        for var1 in vars:
            val1 = bdgs[var1]
            result[var1] = val1
            if isinstance(val1, int):
                for var2 in vars:
                    if var2 > var1:
                        val2 = bdgs[var2]
                        if isinstance(val2, int):
                            result[f"{var1}*{var2}"] = val1 * val2
            else:
                result[f"dims({var1})"] = len(val1)
                if val1:
                    result[f"first({var1})"] = val1[0]
                    result[f"last({var1})"] = val1[-1]
                result[f"prod({var1})"] = math.prod(val1)
        return result

    @staticmethod
    def _is_all_ints(vals: Iterable[Any]) -> bool:
        """
        Returns whether every member of vals is an integer
        """
        return all(isinstance(val, int) for val in vals)

    @staticmethod
    def _linear_fit(pairs: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int]]:
        """
        :param pairs: a list of pairs, each representing a point in (x, y) space
        Returns const, grad_num, grad_den where
           y = const + (x * grad_num) // grad_den for all (x, y)
        if such a solution exists, otherwise None. grad_num and grad_den are simplified
        to have no common factors other than 1.
        """
        x1, y1 = pairs[0]
        xn, yn = pairs[1]
        xd = xn - x1
        yd = yn - y1
        grad_num, grad_den = ShapeInferrer._simplify_fraction(yd, xd)
        const = (y1 * grad_den - x1 * grad_num) // grad_den
        for x, y in pairs:
            if y != const + (x * grad_num) // grad_den:
                return None
        return const, grad_num, grad_den

    @staticmethod
    def _simplify_fraction(num: int, den: int) -> Tuple[int, int]:
        """
        Given a fraction num/den, returns a tuple (num2, den2) where
        num/den == num2/den2 and num2 and den2 have no common factors other than 1.
        """
        hcf = ShapeInferrer._highest_common_factor(num, den)
        return num // hcf, den // hcf

    @staticmethod
    def _highest_common_factor(x: int, y: int) -> int:
        """
        Returns the highest common factor of positive integers x and y.
        """
        assert y > 0, f"Unexpected non-positive denominator: {y}"
        while y > 0:
            x, y = y, x % y
        return x

    @staticmethod
    def _build_expression(x: str, y: str, const: int, grad_num: int, grad_den: int) -> str:
        """
        Returns a string expressing the relationship "y=x*grad_num/grad_den+const",
        simplified to omit adding 0 and multiplying or dividing by 1.
        """
        expr = x
        if grad_num != 1:
            expr = f"{grad_num}*{expr}"
        if grad_den != 1:
            expr = f"{expr}/{grad_den}"
        if const > 0:
            expr = f"{expr}+{const}"
        elif const < 0:
            expr = f"{expr}{const}"
        return f"{y}={expr}"
