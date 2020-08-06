"""
This file shouldn't really have to exist... HOWEVER it's the best I could come
up with for now, to keep type check working in a relatively sane way. The gist
is this: if you were to import things from sympy and pyomo the 'normal' way,
you'd do something like:

>>> from pyomo.environ import ConcreteModel
>>> m = ConcreteModel()

but for whatever reason, the type checker would find that the type of `m` is
'Unknown', or worse, it would complain that
    "ConcreteModel" is unknown import symbol
which is totally silly. Autocompletion stops working. The "fix" is to go:

>>> from pyomo.core.base.PyomoModel import ConcreteModel
>>> m = ConcreteModel()

This not only makes typing useful again, but also gives you your dot
completions again. Ie, this works

>>> m.  # list of options appears

If you're still not sure what I mean, create a file with the two code fragments
and observe the differences

Anyway, the idea is that this file contains all the workaround imports, so
you can import them easily in other files. Ie:

>>> from argh import ConcreteModel

This is a temporary thing, until the type checker gets its shit together. Delete
as soon as possible!

Lastly, it's debatable if this is even worth it. At the time of writing
(15 July 2020) doing a ctrl+F for  `type: ignore` returns 18 results. Lots of that
is from matplotlib, for some reason. Also, there are so many incorrect sympy and
pyomo errors that I ended up typing them as `Any` in any case
"""
from typing import Any

# this is a real sneaky... during runtime, we import from sympy as normal
# but while type checking ("compile time") mypy checks the stubs/sympy/__init__.pyi
# file
from sympy import Matrix as Mat

# from typing import Iterable, cast
# from typing_extensions import Protocol
# class Mat_T(Protocol):
#     def __init__(self, m: Iterable, *args) -> None: pass
#     def T(self) -> Any: pass  # actually returns Mat_T
#     def jacobian(self, *args, **kwargs) -> Any: pass # actually returns Mat_T
#     def __iter__(self) -> Any: pass
#     def __next__(self) -> Any: pass
# from sympy.matrices.dense import MutableDenseMatrix
# def Mat(x: Iterable) -> Mat_T:
#     return cast(Mat_T, MutableDenseMatrix(x))
# Mat = lambda x: cast(Mat_T, MutableDenseMatrix(x))

# instead of `from pyomo.environ import ConcreteModel, Objective, ...`
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core.base.objective import Objective
from pyomo.core.base.param import Param
from pyomo.core.base.sets import Set
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.rangeset import RangeSet
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.expr.logical_expr import inequality
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.current import atan

ConcreteModel: Any
Objective: Any
Param: Any
Set: Any
Var: Any
Constraint: Any
# RangeSet: Any
# ConstraintList: Any
# inequality: Any
# value: Any
# atan: Any

# from typing import Protocol, cast
# class ConcreteModel_T(Protocol):
#     def __init__(self, *args) -> None: pass
#     def __setattr__(self, name: str, value: Any) -> None: pass 
#     def __getattribute__(self, name: str) -> Any: pass
#     def Constraint(self, *args, rule: Any, **kwargs) -> Any: pass
#     def del_component(self, constraint) -> None: pass
