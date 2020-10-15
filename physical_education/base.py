import sympy as sp
from sympy import Matrix as Mat
from pyomo.environ import (
    ConcreteModel, Set, Var, Param, Constraint,
)
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional, TYPE_CHECKING, Union
# from collections import namedtuple
from . import utils

if TYPE_CHECKING:
    from .variable_list import VariableList
    import numpy as np
    # from .links import Link3D


# DummyVar = namedtuple('DummyVar', ['forcename', 'expr', 'sym'])
@dataclass
class _DummyVar:
    forcename: str
    expr: 'sp.Expression'
    sym: sp.Symbol
    func: Union[Callable[...,float],None] = None


class SimpleForceBase3D:
    name: str
    _plot_config: Dict[str, Any] = {}

    # not to be used/overwritten by other classes that inherit

    _dummyvars: List[_DummyVar] = []

    def _dummify(self, expr: 'sp.Expression', forcename: str):
        sym = sp.Symbol(forcename)
        self._dummyvars.append(
            _DummyVar(forcename, expr, sym)
        )
        return sym

    def _get_model_if_dummy_vars(self) -> Optional[ConcreteModel]:
        if len(self._dummyvars) > 0:
            return self.pyomo_vars[self._dummyvars[0].forcename].model()
        else:
            return None

    ####

    def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
        return sp.zeros(*q.shape)

    def add_vars_to_pyomo_model(self, m: ConcreteModel) -> None:
        self.pyomo_params: Dict[str, Param] = {}
        self.pyomo_sets: Dict[str, Set] = {}
        self.pyomo_vars: Dict[str, Var] = {}

        for d in self._dummyvars:
            self.pyomo_vars[d.forcename] = (
                Var(m.fe, m.cp, name=d.forcename)
            )

        utils.add_to_pyomo_model(m, self.name, [
            self.pyomo_params.values(),
            self.pyomo_sets.values(),
            self.pyomo_vars.values(),
        ])

    def __getitem__(self, varname: str) -> Var:
        return self.pyomo_vars[varname]

    def get_pyomo_vars(self, fe: int, cp: int) -> List[Var]:
        return [
            self.pyomo_vars[d.forcename][fe, cp] for d in self._dummyvars
        ]

    def get_sympy_vars(self) -> List[sp.Symbol]:
        return [d.sym for d in self._dummyvars]

    def add_equations_to_pyomo_model(self,
                                     sp_variables: List[sp.Symbol],
                                     pyo_variables: 'VariableList',
                                     collocation: str):
        m = self._get_model_if_dummy_vars()

        if m is None:
            return

        ncp = len(m.cp)
        for d in self._dummyvars:
            forcefunc = utils.lambdify_EOM(d.expr, sp_variables)[0]
            pyo_var = self.pyomo_vars[d.forcename]

            def def_force(m: ConcreteModel, fe: int, cp: int):
                if fe == 1 and cp < ncp:
                    return Constraint.Skip
                else:
                    return pyo_var[fe, cp] == forcefunc(pyo_variables[fe, cp])

            setattr(m, f'{self.name}_{d.forcename}_constr',
                    Constraint(m.fe, m.cp, rule=def_force))
            
            d.func = forcefunc

    def save_data_to_dict(self) -> Dict[str, Any]:
        dct = {'name': self.name}

        m = self._get_model_if_dummy_vars()
        if m is not None:
            for d in self._dummyvars:
                s = d.forcename
                dct[s] = utils.get_vals(self.pyomo_vars[s], (m.cp,))

        return dct

    def init_from_dict_one_point(self,
                                 data: Dict[str, Any],
                                 fed: int, cpd: int,
                                 fes: Optional[int] = None, cps: Optional[int] = None,
                                 **kwargs
                                 ) -> None:
        if fes is None:
            fes = fed - 1

        if cps is None:
            cps = cpd - 1

        assert self.name == data['name']

        v = self.pyomo_vars
        for d in self._dummyvars:
            utils.maybe_set_var(v[d.forcename][fed, cpd],
                                data[d.forcename][fes, cps], **kwargs)

    def animation_setup(self, fig, ax, data: List[List[float]]):
        pass

    def animation_update(self, fig, ax,
                         fe: Optional[int] = None,
                         t: Optional[float] = None,
                         t_arr: Optional['np.ndarray'] = None,
                         track: bool = False):
        pass

    def cleanup_animation(self, fig, ax):
        pass

    def plot(self) -> None:
        import matplotlib.pyplot as plt
        m = self._get_model_if_dummy_vars()
        
        if m is None:
            return
        
        for d in self._dummyvars:
            F = utils.get_vals(self.pyomo_vars[d.forcename], (m.cp,))
            plt.plot(F)
            plt.title(f'{F} force in {self.name}')
            plt.xlabel('Finite element')
            plt.ylabel('$F$ [N]')
            plt.show()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self.name}")'
