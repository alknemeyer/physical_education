# TODO: this is a work in progress!

from typing import Any
# from pyomo.core.base.PyomoModel import ConcreteModel
# from pyomo.core.base.objective import Objective
# from pyomo.core.base.param import Param
# from pyomo.core.base.sets import Set
# from pyomo.core.base.var import Var
# from pyomo.core.base.constraint import Constraint, ConstraintList
# from pyomo.core.base.rangeset import RangeSet
# from pyomo.core.expr.logical_expr import inequality
# from pyomo.core.expr.numvalue import value
# from pyomo.core.expr.current import atan

class ConcreteModel:
    def __setattr__(self, name: str, value: Any) -> None: ...

class Objective: pass

class Param: pass

class Set: pass

class Var: pass

class Constraint: pass

class RangeSet: pass

class ConstraintList: pass

def inequality(*args: Any) -> Any: ...
def value(*args: Any) -> float: ...
def atan(*args: Any) -> Any: ...
