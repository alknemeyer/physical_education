import sympy as sp
from sympy import Matrix as Mat
from pyomo.environ import (
    ConcreteModel, Set, Var, Param, Constraint,
)
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional, TYPE_CHECKING, Tuple, Union
from . import utils

if TYPE_CHECKING:
    from .variable_list import VariableList
    import numpy as np


@dataclass
class _DummyVar:
    name: str
    expr: 'sp.Expression'
    sym: sp.Symbol
    func: Union[Callable[..., float], None] = None


@dataclass
class _DummyParamToFind:
    name: str
    sym: sp.Symbol
    initialvalue: float
    lims: Tuple[float, float]


class SimpleForceBase3D:
    name: str
    _plot_config: Dict[str, Any] = {}

    # not to be used/overwritten by other classes that inherit

    _dummyvars: List[Union[_DummyVar, _DummyParamToFind]]

    def __init__(self) -> None:
        self._dummyvars = []

    def _dummyvar(self, expr: 'sp.Expression', forcename: str):
        if not hasattr(self, '_dummyvars'):
            raise RuntimeError('SimpleForceBase3D hasnt been initialized')

        sym = sp.Symbol(forcename + '_{%s}' % self.name)
        self._dummyvars.append(
            _DummyVar(forcename, expr, sym)
        )
        return sym

    # TODO: symname which is different from paramname?
    def _dummyparam(self, paramname: str,
                    initiavalue: float,
                    lims: Tuple[float, float],
                    symname: Optional[str] = None):
        if not hasattr(self, '_dummyvars'):
            raise RuntimeError('SimpleForceBase3D hasnt been initialized')

        if symname is None: symname = paramname
        sym = sp.Symbol(symname + '_{%s}' % self.name)

        assert lims[0] <= lims[1]
        self._dummyvars.append(
            _DummyParamToFind(paramname, sym, initiavalue, lims)
        )
        return sym

    def _get_model_if_dummy_vars(self) -> Optional[ConcreteModel]:
        if len(self._dummyvars) > 0:
            return self.pyomo_vars[self._dummyvars[0].name].model()
        else:
            return None

    ####

    # def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
    #     return sp.zeros(*q.shape)

    def add_vars_to_pyomo_model(self, m: ConcreteModel) -> None:
        if not hasattr(self, 'pyomo_params'):
            self.pyomo_params: Dict[str, Param] = {}

        if not hasattr(self, 'pyomo_sets'):
            self.pyomo_sets: Dict[str, Set] = {}

        if not hasattr(self, 'pyomo_vars'):
            self.pyomo_vars: Dict[str, Var] = {}

        for d in self._dummyvars:
            self.pyomo_vars[d.name] = (
                Var(m.fe, m.cp, name=d.name)
            )

        utils.add_to_pyomo_model(m, self.name, [
            self.pyomo_params.values(),
            self.pyomo_sets.values(),
            self.pyomo_vars.values(),
        ])

        for d in self._dummyvars:
            if isinstance(d, _DummyParamToFind):
                for v in self.pyomo_vars[d.name][:, :]:
                    v.value = d.initialvalue
                    v.setlb(d.lims[0])
                    v.setub(d.lims[1])

    def __getitem__(self, varname: str) -> Var:
        return self.pyomo_vars[varname]

    def get_pyomo_vars(self, fe: int, cp: int) -> List[Var]:
        return [
            self.pyomo_vars[d.name][fe, cp] for d in self._dummyvars
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
        nfe = len(m.fe)
        for d in self._dummyvars:
            pyo_var = self.pyomo_vars[d.name]

            if isinstance(d, _DummyParamToFind):
                def def_constraint(m: ConcreteModel, fe: int, cp: int):
                    if (fe == 1 and cp < ncp) or fe == nfe:
                        return Constraint.Skip
                    elif cp < ncp:
                        return pyo_var[fe, cp] == pyo_var[fe, cp+1]
                    else:
                        # cp == ncp
                        return pyo_var[fe, cp] == pyo_var[fe+1, 1]
            else:
                forcefunc = utils.lambdify_EOM(d.expr, sp_variables)[0]

                def def_constraint(m: ConcreteModel, fe: int, cp: int):
                    if fe == 1 and cp < ncp:
                        return Constraint.Skip
                    else:
                        return pyo_var[fe, cp] == forcefunc(pyo_variables[fe, cp])

                d.func = forcefunc

            setattr(m, f'{self.name}_{d.name}_constr',
                    Constraint(m.fe, m.cp, rule=def_constraint))

    def save_data_to_dict(self) -> Dict[str, Any]:
        dct = {'name': self.name}

        m = self._get_model_if_dummy_vars()
        if m is not None:
            for d in self._dummyvars:
                s = d.name
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
            utils.maybe_set_var(v[d.name][fed, cpd],
                                data[d.name][fes, cps], **kwargs)

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
            F = utils.get_vals(self.pyomo_vars[d.name], (m.cp,))
            plt.plot(F)
            plt.xlabel('Finite element')

            if isinstance(d, _DummyParamToFind):
                plt.title(f'Value of parameter {d.name} in {self.name}')
            else:
                plt.title(f'{d.name} force in {self.name}')
                plt.ylabel('$F$ [N]')
            plt.show()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self.name}")'
