# https://stackoverflow.com/questions/15853469/putting-current-class-as-return-type-annotation
from __future__ import annotations
from typing import List, Protocol
from pyomo.environ import (
    ConcreteModel, Param, Set, Var
)
import sympy as sp

from . import variable_list
VariableList = variable_list.VariableList

# TODO: this is likely super out of date! Update it!

class RigidBody(Protocol):
    def __init__(self, name: str, *args, **kwargs):
        """
        Initialize the component, setting up symbolic variables and as
        many of the equations as possible. If possible, add typing to
        the variables and `trigsimp` as you go

        Also, add `assert` statements for common errors
        """
        self.name = name  # type: ignore
        pass

    def add_xyz(self, otherthing) -> RigidBody:
        """
        Methods to add things to the object should be prefixed with `add_`
        to help with discovery. If you want to be able to chain things like:
        >>> my_obj.add_foo().add_bar().add_baz()
        then make sure to return the object itself
        """
        return self
    
    def add_vars_to_pyomo_model(self, m: ConcreteModel) -> None:
        """
        Define the pyomo variables that will be used in the pyomo model,
        and store them in three dictionaries

        If the component has sub components (eg a link having a foot),
        make sure to call the method on them too
        """
        my_param = Param(initialize=1.5)
        my_set = Set(initialize=range(3), ordered=True)
        my_var = Var(my_set)

        self.pyomo_params = {'my_param': my_param}  # type: ignore
        self.pyomo_sets = {'my_set': my_set}  # type: ignore
        self.pyomo_vars = {'my_var': my_var}  # type: ignore

        # self.sub_component.add_vars_to_pyomo_model(m)
    
    def add_equations_to_pyomo_model(self,
                                     m: ConcreteModel,
                                     sp_variables: List[sp.Symbol],
                                     pyo_variables: VariableList) -> None:
        """
        Add the model (and subcomponents) equations to the pyomo model.
        The sympy variables are passed so that lambda functions can be
        created. `sp_variables` will correspond to `pyo_variables`
        """
        pass

        # self.sub_component.add_equations_to_pyomo_model(
        #   m, sp_variables, pyo_variables
        # )
    
    def get_pyomo_vars(self, fe: int, cp: int) -> List[Var]:
        """Get the pyomo variables defined by this model. Note that
        `fe` and `cp` are one-based.
        NB: keep in sync with get_sympy_vars()"""
        fe -= 1
        cp -= 1

        v = self.pyomo_vars
        return [
            *v['my_var'][fe,cp,:],
        ]
    
    def get_sympy_vars(self) -> List[sp.Symbol]:
        """Get the sympy variables defined by this model.
        NB: keep in sync with get_pyomo_vars()"""
        # foot_vars = self.foot.get_sympy_vars() if self.has_foot() else []
        return [
            # *self.q,
            # *foot_vars
        ]

    def __repr__(self) -> str:
        return f'TemplateBody(name="{self.name}")'
