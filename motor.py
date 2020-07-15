import visual
import sympy as sp
import numpy as np
from sympy import Matrix as Mat
from typing import Any, Callable, Dict, List, Optional, Tuple
from pyomo.environ import (
    ConcreteModel, Set, Var, Param, inequality,
)
import itertools
from . import utils
from .variable_list import VariableList


class _TorqueSpeedLimit:
    def __init__(self, limit_to: float) -> None:
        self.limit_to = limit_to
        # self.limit_to_f: List[Callable] = []
        self.relative_angle_velocities = []

    def add_rel_vel(self, rel_velocity):
        self.relative_angle_velocities.append(rel_velocity)

    def add_equations_to_pyomo_model(self, Tc: Var, Tc_set: Set):
        m = Tc.model()
        Tmax = self.limit_to

        visual.warn('Not sure if the equation is correct!')

        @m.Constraint(m.fe, Tc_set)
        def Tmax_constr(m, fe, idx):
            rel_ang_vel = self.relative_angle_velocities[idx]
            return inequality(- Tmax, Tmax/Tc[fe, idx] * rel_ang_vel, Tmax)


class Motor3D:
    def __init__(self, name: str, Rb_I: Mat,
                 bounds: Tuple = (-1, 1), limit_to: Optional[float] = None):
        self.name = name
        self.Rb_I = Rb_I
        self.bounds = bounds

        if limit_to is not None:
            self.torque_speed_limit = _TorqueSpeedLimit(limit_to)

        self.input_torques: List[sp.Symbol] = []
        self.torques_on_body = Mat([0., 0., 0.])

        # a list of tuples of [torques_on_body, rotation, about] on other bodies
        self.other_bodies: List[Tuple[Mat, Mat, str]] = []

    def add_input_torques_at(self, otherlink, about: str):
        """ Add input torques between two links, about axis `about` of `self`
        >>> link_body.add_input_torques_at(link_UFL, about='xy')
        TODO: THINK MORE ABOUT WHAT THIS REALLY MEANS! The second torque should really be rotated
        """
        assert all(axis in 'xyz' for axis in about)

        τ = Mat([
            *sp.symbols(r'\tau_{%s/%s/:%s}' % (self.name, otherlink.name, len(about)))
        ])
        self.input_torques.extend(τ)
        
        torques_on_other_body = Mat([0, 0, 0])

        for idx, axis in enumerate(about):
            constraint_axis = Mat([0, 0, 0])
            constraint_axis['xyz'.index(axis)] = 1
            torque = τ[idx] * self.Rb_I @ Mat(constraint_axis)
            self.torques_on_body += torque
            torques_on_other_body -= torque

        self.other_bodies.append(
            (torques_on_other_body, otherlink.Rb_I, about))

    def calc_eom(self, q, dq, ddq) -> Mat:
        ang_vel_body = utils.skew_symmetric(self.Rb_I, q, dq)
        dW = Mat([self.torques_on_body.dot(ang_vel_body)])
        Q = dW.jacobian(dq).T

        for torques_on_body, Rb_I, about in self.other_bodies:
            ang_vel = utils.skew_symmetric(Rb_I, q, dq)
            dW = Mat([torques_on_body.dot(ang_vel)])
            Q += dW.jacobian(dq).T

            if hasattr(self, 'torque_speed_limit'):
                for ax in about:
                    self.torque_speed_limit.add_rel_vel(
                        (ang_vel_body - ang_vel)['xyz'.index(ax)]
                    )

        return Q

    def add_vars_to_pyomo_model(self, m: ConcreteModel) -> None:
        Tc_set = Set(initialize=range(len(self.input_torques)),
                     name='Tc_set', ordered=True)
        Tc = Var(m.fe, Tc_set, name='Tc', bounds=self.bounds)

        self.pyomo_params: Dict[str, Param] = {}
        self.pyomo_sets: Dict[str, Set] = {'Tc_set': Tc_set}
        self.pyomo_vars: Dict[str, Var] = {'Tc': Tc}

        for v in itertools.chain(self.pyomo_params.values(),
                                 self.pyomo_sets.values(),
                                 self.pyomo_vars.values()):
            newname = f'{self.name}_{v}'
            assert not hasattr(m, newname), \
                f'The pyomo model already has a variable with the name "{newname}"'
            setattr(m, newname, v)

    def __getitem__(self, varname: str) -> Var:
        return self.pyomo_vars[varname]

    def get_pyomo_vars(self, fe: int, cp: int) -> List:
        return [
            *self.pyomo_vars['Tc'][fe, :],
        ]

    def get_sympy_vars(self) -> List[sp.Symbol]:
        return [
            *self.input_torques,
        ]

    def save_data_to_dict(self) -> Dict[str, Any]:
        Tc_set = self.pyomo_sets['Tc_set']

        return {
            'Tc': utils.get_vals(self.pyomo_vars['Tc'], (Tc_set,)) if len(Tc_set) > 0 else [],
        }

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

        if len(data['Tc']) > 0:
            for idx, T in enumerate(self.pyomo_sets['Tc_set']):
                utils.maybe_set_var(
                    self.pyomo_vars['Tc'][fed, T], data['Tc'][fes, idx], **kwargs)

    def add_equations_to_pyomo_model(self,
                                     sp_variables: List[sp.Symbol],
                                     pyo_variables: VariableList,
                                     collocation: str):
        # t1(i) < or > TShoulderMax - TShoulderMax/shoulderWSS*( dAlpha1_0(i) );
        if hasattr(self, 'torque_speed_limit'):
            self.torque_speed_limit.add_equations_to_pyomo_model(
                self.pyomo_vars['Tc'], self.pyomo_sets['Tc'])

    def torque_squared_cost(self):
        return sum([Tc**2 for Tc in self.pyomo_vars['Tc'][:, :]])

    def animation_setup(self, fig, ax, data: List[List[float]]):
        pass

    def animation_update(self, fig, ax,
                         fe: Optional[int] = None,
                         t: Optional[float] = None,
                         t_arr: Optional[np.ndarray] = None,
                         track: bool = False):
        pass

    def cleanup_animation(self, fig, ax):
        pass

    def plot(self) -> None:
        from matplotlib import pyplot as plt
        try:
            Tc_set = self.pyomo_sets['Tc_set']
            Tc = utils.get_vals(self.pyomo_vars['Tc'], (Tc_set,))
            plt.plot(Tc)
            plt.legend(Tc_set)
            plt.grid(True)
            plt.title('Input torques in link ' + self.name)
            plt.xlabel('Finite element')
            plt.ylabel('$T$ [Nm/body_weight]')
            plt.show()
        except StopIteration:
            pass

    def __repr__(self) -> str:
        return f'Motor3D(name="{self.name}")'  # , axes="{self.axes}
