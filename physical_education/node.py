# TODO: import from `typing` when not limited by python version
from typing_extensions import Protocol
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pyomo.environ import (
    ConcreteModel, Param, Set, Var,
)
from sympy import Matrix as Mat
import sympy as sp
import numpy as np

if TYPE_CHECKING:
    from .variable_list import VariableList


# commented out sections represent optional stuff


class Node(Protocol):
    name: str
    pyomo_params: Dict[str, Param]
    pyomo_sets: Dict[str, Set]
    pyomo_vars: Dict[str, Var]
    # _plot_config: Dict[str, Any]

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Initialize the component, setting up symbolic variables and as
        many of the equations as possible. If possible, add typing to
        the variables and `trigsimp` as you go

        Also, add `assert` statements for common errors
        """
        ...

    def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
        """
        Calculate the force input mapping (`Q`) defined by this
        component, and return it
        """
        ...

    def add_vars_to_pyomo_model(self, m: ConcreteModel) -> None:
        """
        Define the pyomo variables that will be used in the pyomo model,
        and store them in three dictionaries

        If the component has sub components (eg a link having a foot),
        make sure to call the method on them too
        """
        ...

    def get_pyomo_vars(self, fe: int, cp: int) -> List[Var]:
        """
        Get the pyomo variables defined by this model. Note that
        `fe` and `cp` are one-based.
        NB: keep in sync with get_sympy_vars(). Eg:

        ```python
        fe -= 1
        cp -= 1

        v = self.pyomo_vars
        return [
            *v['my_var'][fe,cp,:],
        ]
        ```
        """
        ...

    def get_sympy_vars(self) -> List[sp.Symbol]:
        """
        Get the sympy variables defined by this model.
        NB: keep in sync with get_pyomo_vars()
        """
        ...

    def save_data_to_dict(self) -> Dict[str, Any]:
        """
        Save the data in this model to a dictionary, saving whatever
        is "useful" in with whatever keys you want

        NB keep in sync with `init_from_dict_one_point`
        """
        ...

    def init_from_dict_one_point(self, data: Dict[str, Any],
                                 fed: int, cpd: int,
                                 fes: Optional[int] = None, cps: Optional[int] = None,
                                 **kwargs
                                 ) -> None:
        """
        Load data, saved using the `save_data_to_dict` method

        NB keep in sync with `save_data_to_dict`
        """
        ...

    def add_equations_to_pyomo_model(self,
                                     sp_variables: List[sp.Symbol],
                                     pyo_variables: 'VariableList',
                                     collocation: str) -> None:
        """
        Add the model (and subcomponents) equations to the pyomo model.
        The sympy variables are passed so that lambda functions can be
        created. `sp_variables` will correspond to `pyo_variables`
        """
        ...

    def __getitem__(self, varname: str) -> Var:
        """
        Shorthand for `self.pyomo_vars[varname]`
        """
        ...

    def animation_setup(self, fig, ax, data: List[List[float]]) -> None:
        """
        Run setup code code an animation, like geometric calculations
        and initial setup of plot objects

        Possibly altered by options in `self._plot_config`, which is why this
        method accepts no other arguments

        TODO: explain how `data` is structured
        """
        ...

    def animation_update(self, fig, ax,
                         fe: Optional[int] = None,
                         t: Optional[float] = None,
                         t_arr: Optional[np.ndarray] = None,
                         track: bool = False) -> None:
        """
        Updates plot objects defined in `animation_setup`

        TODO: explain what the args are
        """
        ...

    def cleanup_animation(self, fig, ax):
        """
        Delete plot objects defined in `animation_setup`. Useful when jupyter
        causes handles to plot objects to remain, causing issues with python's
        garbage collector and matplotlib

        ```python
        try:
            del self.line
        except:
            pass
        ```
        """
        ...

    def plot(self, *args, **kwargs):
        """
        Produce line plots of the data held by the object - eg. position vs time

        Possibly altered by options in `self._plot_config`, which is why this
        method accepts no arguments
        """
        ...

    def __repr__(self) -> str:
        ...
        # return f'TemplateBody(name="{self.name}")'
