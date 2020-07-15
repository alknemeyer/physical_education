"""
    Usage is along the lines of:
    ```
    >>> from collocation import radau_3
    >>> m.interp_q  = Constraint(m.fe, m.cp, m.vars, rule=utils.radau_3(m.q,  m.dq))
    >>> m.interp_dq = Constraint(m.fe, m.cp, m.vars, rule=utils.radau_3(m.dq, m.ddq))
    ```
    Or,
    ```
    >>> col = collocation.get_collocation_func('radau_3')
    ```
    Or,
    ```
    >>> collocation.check_collocation_method('radua_56')  # raises ValueError
    ```
"""
from pyomo.environ import ConcreteModel, Var, Constraint
from typing import Dict, Callable, Tuple

_collocation_mapping: Dict[str, Tuple[Callable, int]] = {}


def get_ncp(collocation: str):
    check_collocation_method(collocation)
    return _collocation_mapping[collocation][1]


def get_collocation_func(collocation: str):
    check_collocation_method(collocation)
    return _collocation_mapping[collocation][0]


def check_collocation_method(collocation: str):
    if not collocation in _collocation_mapping.keys():
        raise ValueError(
            f'Invalid collocation method. Valid options are {_collocation_mapping.keys()}. Got: {collocation}')


def explicit_euler(q: Var, dq: Var):
    def func(m: ConcreteModel, fe: int, cp: int, var: Var):
        assert cp == 1 and len(m.cp) == 1
        if fe > 1:
            return q[fe, cp, var] == q[fe-1, cp, var] + m.hm0 * m.hm[fe] * dq[fe-1, cp, var]
        else:
            return Constraint.Skip
    return func


_collocation_mapping['explicit_euler'] = (explicit_euler, 1)


def implicit_euler(q: Var, dq: Var):
    def func(m: ConcreteModel, fe: int, cp: int, var: Var):
        assert cp == 1 and len(m.cp) == 1
        if fe > 1:
            return q[fe, cp, var] == q[fe-1, cp, var] + m.hm0 * m.hm[fe] * dq[fe, cp, var]
        else:
            return Constraint.Skip
    return func


_collocation_mapping['implicit_euler'] = (implicit_euler, 1)


def radau_3(q: Var, dq: Var):
    R = [
        [0.19681547722366, -0.06553542585020,  0.02377097434822],
        [0.39442431473909,  0.29207341166523, -0.04154875212600],
        [0.37640306270047,  0.51248582618842,  0.11111111111111],
    ]

    def func(m: ConcreteModel, fe: int, cp: int, var: Var):
        assert 1 <= cp <= 3 and len(m.cp) == 3
        if fe > 1:
            inc = sum(R[cp-1][pp-1]*dq[fe, pp, var] for pp in m.cp)
            return q[fe, cp, var] == q[fe-1, m.cp[-1], var] + m.hm0*m.hm[fe] * inc
        else:
            return Constraint.Skip
    return func


_collocation_mapping['radau_3'] = (radau_3, 3)


def radau_2(q: Var, dq: Var):
    R = [
        [0.416666125000187, -0.083333125000187],
        [0.749999625000187,  0.250000374999812],
    ]

    def func(m: ConcreteModel, fe: int, cp: int, var: Var):
        assert 1 <= cp <= 2 and len(m.cp) == 2
        if fe > 1:
            inc = sum(R[cp-1][pp-1]*dq[fe, pp, var] for pp in m.cp)
            return q[fe, cp, var] == q[fe-1, m.cp[-1], var] + m.hm0*m.hm[fe] * inc
        else:
            return Constraint.Skip
    return func


_collocation_mapping['radau_2'] = (radau_2, 2)
