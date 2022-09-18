import sympy as sp
import numpy as np
from typing import (Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING, Union)
import pyomo.environ as pyo
from sympy import Matrix as Mat
from . import utils
from .system import System3D, System2D

if TYPE_CHECKING:
    from .links import Link3D, Link2D
    from .variable_list import VariableList
    from matplotlib.pyplot import Axes


class _TorqueSpeedLimit:
    def __init__(self, name: str, torque_bounds: Tuple[float, float], no_load_speed: float) -> None:
        """
        Limit the input torque to:
            $$ τ = τ_s - ω / ω_n * τ_s $$

        where
            τ = input torque
            ω = rotational velocity about the same axis
            ω_n = no load speed (x-axis intercept on graphs you may see)
            τ_s = stall torque (y-axis intercept on graphs you may see)

        From http://lancet.mit.edu/motors/motors3.html
        """
        assert torque_bounds[0] < torque_bounds[1]

        self.name = name
        self.torque_bounds = torque_bounds
        self.no_load_speed = no_load_speed

        self.relative_angle_velocities: List['sp.Expression'] = []
        self.axes: List[str] = []

    def add_rel_vel(self, rel_velocity: 'sp.Expression', axis: str):
        self.relative_angle_velocities.append(rel_velocity)
        self.axes.append(axis)

    def add_equations_to_pyomo_model(self, sp_variables: List[sp.Symbol], pyo_variables: 'VariableList',
                                     collocation: str, Tc: pyo.Var, Tc_set: pyo.Set):
        m = Tc.model()
        self.Tc = Tc
        self.Tc_set = Tc_set
        self.pyo_variables = pyo_variables

        self.rel_angle_vels_f = utils.lambdify_EOM(self.relative_angle_velocities, sp_variables)

        τ_sn, τ_sp = self.torque_bounds
        ω_n = self.no_load_speed

        def torque_speed_limit_constr(m, fe, idx, posneg):
            ω = self.rel_angle_vels_f[idx](pyo_variables[fe, 1])
            τ = Tc[fe, idx]

            if posneg == '+':
                return τ <= τ_sp * (1 - ω / ω_n)
            elif posneg == '-':
                return τ >= τ_sn * (1 + ω / ω_n)
            else:
                raise ValueError(f'posneg should be "+" or "-". Got: {posneg}')

        constraintname = self.name + '_torque_speed_limit'
        assert not hasattr(m, constraintname), \
            f'The model already has a constraint with the name {constraintname}'

        setattr(m, constraintname, pyo.Constraint(m.fe, Tc_set, ('+', '-'), rule=torque_speed_limit_constr))

    def plot(self, _ax: Optional['Axes'] = None, markersize: float = 5., save_to: Optional[str] = None):
        m = self.Tc.model()
        # ncp = len(m.cp)
        data: List[List[float]] = [[v.value for v in self.pyo_variables[fe, 1]] for fe in m.fe]

        τ = utils.get_vals(self.Tc, (self.Tc_set, ))
        ω = np.zeros(τ.shape)
        for idx, ω_f in enumerate(self.rel_angle_vels_f):
            for fe in m.fe:
                ω[fe - 1, idx] = ω_f(data[fe - 1])

        import matplotlib.pyplot as plt
        if _ax is None:
            ax = plt.subplot()
        else:
            ax = _ax

        # plot the torque-speed curve line
        τ_sn, τ_sp = self.torque_bounds
        ω_n = self.no_load_speed
        # plt.plot([0,  2*ω_n], [τ_sp, 0], '--', color='black')
        # plt.plot([-2*ω_n, 0], [0, τ_sn], '--', color='black')
        ax.plot([0, 2 * ω_n], [τ_sp, τ_sn], '--', color='black')
        ax.plot([-2 * ω_n, 0], [τ_sp, τ_sn], '--', color='black')

        # torque limit line
        ax.plot([-2 * ω_n, 0], [τ_sp, τ_sp], '--', color='black')
        ax.plot([0, 2 * ω_n], [τ_sn, τ_sn], '--', color='black')

        # plot torque vs speed
        for i in range(τ.shape[1]):
            ax.scatter(ω[:, i], τ[:, i], s=markersize, label=f'$\\tau_{i}$')

        if _ax is not None:
            return ax

        plt.title(f'Torque speed curve in {self.name}')
        plt.xlabel('Angular velocity $\\omega$ [rad/s]')
        plt.ylabel('Input torque $\\tau$ [Nm/body_weight]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_to is not None:
            plt.gcf().savefig(f'{save_to}torque-scatterplot-{self.name}.pdf')
        else:
            plt.show()

        return ax

    def __repr__(self) -> str:
        return f'_TorqueSpeedLimit(torque_bounds={self.torque_bounds}, no_load_speed={self.no_load_speed}, axes={self.axes})'


class Motor2D:
    def __init__(self,
                 name: str,
                 Rb_I: Mat,
                 *,
                 torque_bounds: Tuple[float, float],
                 no_load_speed: Optional[float] = None):
        self.name = name
        self.Rb_I = Rb_I
        self.torque_bounds = torque_bounds

        if no_load_speed is not None:
            self.torque_speed_limit = _TorqueSpeedLimit(name, torque_bounds, no_load_speed)
        self.torque_on_body = Mat([0., 0., 0.])
        self.torque_on_other_body = Mat([0., 0., 0.])

    def add_input_torques_at(self, otherlink: 'Link2D'):
        """ Add input torques between two links.
        >>> link_body.add_input_torques_at(link_UFL)
        """
        τ = Mat([sp.symbols(r'\tau_{%s}' % (self.name))])
        self.input_torque = τ

        self.torque_on_body = Mat([0, 0, self.input_torque])
        self.torque_on_other_body = -Mat([0, 0, self.input_torque])
        self.otherlink_Rb_I = otherlink.Rb_I

    def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
        def rot_2d_to_3d(R) -> Mat:
            return sp.Matrix([
                [R[0, 0], R[0, 1], 0],
                [R[1, 0], R[1, 1], 0],
                [0, 0, 1],
            ])

        # Torque on first link.
        ang_vel_body = utils.skew_symmetric(rot_2d_to_3d(self.Rb_I), utils.vec(*q, sp.symbols("z")),
                                            utils.vec(*dq, sp.symbols("dz")))
        dW = Mat([self.torque_on_body.dot(ang_vel_body)])
        Q = dW.jacobian(dq).T
        # Torque on second link.
        ang_vel = utils.skew_symmetric(rot_2d_to_3d(self.otherlink_Rb_I), utils.vec(*q, sp.symbols("z")),
                                       utils.vec(*dq, sp.symbols("dz")))
        dW = Mat([self.torque_on_other_body.dot(ang_vel)])
        Q += dW.jacobian(dq).T

        if hasattr(self, 'torque_speed_limit'):
            self.torque_speed_limit.add_rel_vel(
                (ang_vel_body - ang_vel)[2],
                "z",
            )

        return Q

    def add_vars_to_pyomo_model(self, m: pyo.ConcreteModel) -> None:
        from pyomo.environ import Set, Var, Param

        Tc_set = Set(initialize=range(len(self.input_torque)), name='Tc_set', ordered=True)
        Tc = Var(m.fe, Tc_set, name='Tc', bounds=self.torque_bounds)

        self.pyomo_params: Dict[str, Param] = {}
        self.pyomo_sets: Dict[str, Set] = {'Tc_set': Tc_set}
        self.pyomo_vars: Dict[str, Var] = {'Tc': Tc}

        utils.add_to_pyomo_model(m, self.name, [
            self.pyomo_params.values(),
            self.pyomo_sets.values(),
            self.pyomo_vars.values(),
        ])

    def __getitem__(self, varname: str) -> pyo.Var:
        return self.pyomo_vars[varname]

    def get_pyomo_vars(self, fe: int, cp: int) -> List:
        return [
            *self.pyomo_vars['Tc'][fe, :],
        ]

    def get_sympy_vars(self) -> List[sp.Symbol]:
        return [
            *self.input_torque,
        ]

    def add_equations_to_pyomo_model(self, sp_variables: List[sp.Symbol], pyo_variables: 'VariableList',
                                     collocation: str):
        if hasattr(self, 'torque_speed_limit'):
            self.torque_speed_limit.add_equations_to_pyomo_model(sp_variables, pyo_variables, collocation,
                                                                 self.pyomo_vars['Tc'], self.pyomo_sets['Tc_set'])

    def save_data_to_dict(self) -> Dict[str, Any]:
        Tc_set = self.pyomo_sets['Tc_set']

        return {
            'name': self.name,
            'Tc': utils.get_vals(self.pyomo_vars['Tc'], (Tc_set, )) if len(Tc_set) > 0 else [],
        }

    def init_from_dict_one_point(self,
                                 data: Dict[str, Any],
                                 fed: int,
                                 cpd: int,
                                 fes: Optional[int] = None,
                                 cps: Optional[int] = None,
                                 **kwargs) -> None:
        if fes is None:
            fes = fed - 1

        if cps is None:
            cps = cpd - 1

        if len(data['Tc']) > 0:
            for idx, T in enumerate(self.pyomo_sets['Tc_set']):
                utils.maybe_set_var(self.pyomo_vars['Tc'][fed, T], data['Tc'][fes, idx], **kwargs)

    def torque_squared_cost(self, weight: float = 1.0):
        # TODO: add an option to scale by time and body weight?
        # previous tests slowed things down quite a bit, so I'm not sure
        # it's worth it, especially since we're not actually doing global
        # optimization in any case
        return sum([(weight * Tc)**2 for Tc in self.pyomo_vars['Tc'][:, :]])

    def animation_setup(self, fig, ax, data: List[List[float]]):
        pass

    def animation_update(self,
                         fig,
                         ax,
                         fe: Optional[int] = None,
                         t: Optional[float] = None,
                         t_arr: Optional[np.ndarray] = None,
                         track: bool = False):
        pass

    def cleanup_animation(self, fig, ax):
        pass

    def plot(self, save_to: Optional[str] = None) -> None:
        import matplotlib.pyplot as plt
        try:
            Tc_set = self.pyomo_sets['Tc_set']
            Tc = utils.get_vals(self.pyomo_vars['Tc'], (Tc_set, ))
            plt.plot(Tc)
            plt.legend(Tc_set)
            plt.grid(True)
            plt.title('Input torques in link ' + self.name)
            plt.xlabel('Finite element')
            plt.ylabel('$T$ [Nm/body_weight]')
            plt.tight_layout()
            if save_to is not None:
                plt.gcf().savefig(f'{save_to}torque-lineplot-{self.name}.pdf')
            else:
                plt.show()
        except StopIteration:
            pass

        if hasattr(self, 'torque_speed_limit'):
            self.torque_speed_limit.plot()

    def __repr__(self) -> str:
        if hasattr(self, 'torque_speed_limit'):
            tsl = f'\n{" "*6}{self.torque_speed_limit}'
        else:
            tsl = ''
        return f'Motor2D(name="{self.name}", torque_bounds={self.torque_bounds}{tsl})'


class Motor3D:
    def __init__(self,
                 name: str,
                 Rb_I: Mat,
                 *,
                 torque_bounds: Tuple[float, float],
                 no_load_speed: Optional[float] = None):
        self.name = name
        self.Rb_I = Rb_I
        self.torque_bounds = torque_bounds

        if no_load_speed is not None:
            self.torque_speed_limit = _TorqueSpeedLimit(name, torque_bounds, no_load_speed)

        self.input_torques: List[sp.Symbol] = []
        self.torques_on_body = Mat([0., 0., 0.])

        # a list of tuples of [torques_on_body, rotation, about] on other bodies
        self.other_bodies: List[Tuple[Mat, Mat, str]] = []
        # Keep track of relative velocity of the links on the motor.
        self.relative_angle_velocities: List['sp.Expression'] = []

    def add_input_torques_at(self, otherlink: 'Link3D', about: str):
        """ Add input torques between two links, about axis `about` of `self`
        >>> link_body.add_input_torques_at(link_UFL, about='xy')
        """
        assert all(axis in 'xyz' for axis in about)

        τ = Mat([*sp.symbols(r'\tau_{%s/%s/:%s}' % (self.name, otherlink.name, len(about)))])
        self.input_torques.extend(τ)

        torques_on_other_body = Mat([0, 0, 0])

        for idx, axis in enumerate(about):
            constraint_axis = Mat([0, 0, 0])
            constraint_axis['xyz'.index(axis)] = 1
            torque = τ[idx] * self.Rb_I @ Mat(constraint_axis)
            self.torques_on_body += torque
            torques_on_other_body -= torque

        self.other_bodies.append((torques_on_other_body, otherlink.Rb_I, about))

    def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
        ang_vel_body = utils.skew_symmetric(self.Rb_I, q, dq)
        dW = Mat([self.torques_on_body.dot(ang_vel_body)])
        Q = dW.jacobian(dq).T

        for (torques_on_body, Rb_I, about) in self.other_bodies:
            ang_vel = utils.skew_symmetric(Rb_I, q, dq)
            dW = Mat([torques_on_body.dot(ang_vel)])
            Q += dW.jacobian(dq).T

            for ax in about:
                if hasattr(self, 'torque_speed_limit'):
                    self.torque_speed_limit.add_rel_vel(
                        (ang_vel_body - ang_vel)['xyz'.index(ax)],
                        ax,
                    )
                self.relative_angle_velocities.append((ang_vel_body - ang_vel)['xyz'.index(ax)])
        return Q

    def add_vars_to_pyomo_model(self, m: pyo.ConcreteModel) -> None:
        from pyomo.environ import Set, Var, Param

        Tc_set = Set(initialize=range(len(self.input_torques)), name='Tc_set', ordered=True)
        Tc = Var(m.fe, Tc_set, name='Tc', bounds=self.torque_bounds)

        self.pyomo_params: Dict[str, Param] = {}
        self.pyomo_sets: Dict[str, Set] = {'Tc_set': Tc_set}
        self.pyomo_vars: Dict[str, Var] = {'Tc': Tc}

        utils.add_to_pyomo_model(m, self.name, [
            self.pyomo_params.values(),
            self.pyomo_sets.values(),
            self.pyomo_vars.values(),
        ])

    def __getitem__(self, varname: str) -> pyo.Var:
        return self.pyomo_vars[varname]

    def get_pyomo_vars(self, fe: int, cp: int) -> List:
        return [
            *self.pyomo_vars['Tc'][fe, :],
        ]

    def get_sympy_vars(self) -> List[sp.Symbol]:
        return [
            *self.input_torques,
        ]

    def add_equations_to_pyomo_model(self, sp_variables: List[sp.Symbol], pyo_variables: 'VariableList',
                                     collocation: str):
        self.rel_angle_vels_f = utils.lambdify_EOM(self.relative_angle_velocities, sp_variables)
        if hasattr(self, 'torque_speed_limit'):
            self.torque_speed_limit.add_equations_to_pyomo_model(sp_variables, pyo_variables, collocation,
                                                                 self.pyomo_vars['Tc'], self.pyomo_sets['Tc_set'])

    def save_data_to_dict(self) -> Dict[str, Any]:
        Tc_set = self.pyomo_sets['Tc_set']

        return {
            'name': self.name,
            'Tc': utils.get_vals(self.pyomo_vars['Tc'], (Tc_set, )) if len(Tc_set) > 0 else [],
        }

    def init_from_dict_one_point(self,
                                 data: Dict[str, Any],
                                 fed: int,
                                 cpd: int,
                                 fes: Optional[int] = None,
                                 cps: Optional[int] = None,
                                 **kwargs) -> None:
        if fes is None:
            fes = fed - 1

        if cps is None:
            cps = cpd - 1

        if len(data['Tc']) > 0:
            for idx, T in enumerate(self.pyomo_sets['Tc_set']):
                utils.maybe_set_var(self.pyomo_vars['Tc'][fed, T], data['Tc'][fes, idx], **kwargs)

    def torque_squared_cost(self, weight: float = 1.0):
        # TODO: add an option to scale by time and body weight?
        # previous tests slowed things down quite a bit, so I'm not sure
        # it's worth it, especially since we're not actually doing global
        # optimization in any case
        return sum([(weight * Tc)**2 for Tc in self.pyomo_vars['Tc'][:, :]])

    def animation_setup(self, fig, ax, data: List[List[float]]):
        pass

    def animation_update(self,
                         fig,
                         ax,
                         fe: Optional[int] = None,
                         t: Optional[float] = None,
                         t_arr: Optional[np.ndarray] = None,
                         track: bool = False):
        pass

    def cleanup_animation(self, fig, ax):
        pass

    def plot(self, save_to: Optional[str] = None) -> None:
        import matplotlib.pyplot as plt
        try:
            Tc_set = self.pyomo_sets['Tc_set']
            Tc = utils.get_vals(self.pyomo_vars['Tc'], (Tc_set, ))
            plt.plot(Tc)
            plt.legend(Tc_set)
            plt.grid(True)
            plt.title('Input torques in link ' + self.name)
            plt.xlabel('Finite element')
            plt.ylabel('$T$ [Nm/body_weight]')
            plt.tight_layout()
            if save_to is not None:
                plt.gcf().savefig(f'{save_to}torque-lineplot-{self.name}.pdf')
            else:
                plt.show()
        except StopIteration:
            pass

        if hasattr(self, 'torque_speed_limit'):
            self.torque_speed_limit.plot()

    def __repr__(self) -> str:
        if hasattr(self, 'torque_speed_limit'):
            tsl = f'\n{" "*6}{self.torque_speed_limit}'
        else:
            tsl = ''
        return f'Motor3D(name="{self.name}", torque_bounds={self.torque_bounds}{tsl})'


def add_torque(link, otherlink, about: Optional[str] = None, name: Optional[str] = None, **kwargs):
    name = utils.get_name(name, [link, otherlink], 'torque')
    if about is not None:
        motor = Motor3D(str(name), link.Rb_I, **kwargs)
        link.nodes[name] = motor
        motor.add_input_torques_at(otherlink, about=about)
        return motor
    else:
        motor = Motor2D(str(name), link.Rb_I, **kwargs)
        link.nodes[name] = motor
        motor.add_input_torques_at(otherlink)
        return motor


def torques(robot_or_link: Union[System3D, System2D, 'Link3D', 'Link2D']) -> List[Union[Motor3D, Motor2D]]:
    from typing import cast

    if isinstance(robot_or_link, System3D):
        robot = robot_or_link
        return [
            cast(Motor3D, node) for link in robot.links for node in link.nodes.values() if isinstance(node, Motor3D)
        ]
    elif isinstance(robot_or_link, System2D):
        robot = robot_or_link
        return [
            cast(Motor2D, node) for link in robot.links for node in link.nodes.values() if isinstance(node, Motor2D)
        ]
    else:
        link = robot_or_link
        return [cast(Motor3D, node) for node in link.nodes.values() if isinstance(node, Motor3D)]


def torque_squared_penalty(robot: Union['System3D', 'System2D'], weights: Optional[List[float]] = None):
    if weights is not None:
        return sum(torque.torque_squared_cost(weights[idx]) for idx, torque in enumerate(torques(robot)))
    else:
        return sum(torque.torque_squared_cost() for torque in torques(robot))


# TODO: this outputs an Iterator with `len(m.fe) * len(Tc_set)` elements
# perhaps an Iterator of Iterators would make more sense?
def power(motor: Motor3D, pyo_variables, fe: Optional[int] = None) -> Iterator:
    # if hasattr(motor, 'torque_speed_limit'):
    # rel_angle_vels_f = motor.torque_speed_limit.rel_angle_vels_f
    rel_angle_vels_f = motor.rel_angle_vels_f

    Tc = motor.pyomo_vars['Tc']
    Tc_set = motor.pyomo_sets['Tc_set']
    m = Tc.model()
    v = pyo_variables

    # ω * τ
    if fe is not None:
        return (rel_angle_vels_f[idx](v[fe, 1]) * Tc[fe, idx] for idx in Tc_set)
    else:
        return (rel_angle_vels_f[idx](v[fe, 1]) * Tc[fe, idx] for fe in m.fe for idx in Tc_set)
    # else:
    #     raise RuntimeError('Current impementation requires a TorqueSpeedLimit :/')


def work_penalty(robot: 'System3D'):
    return sum(sum(power(motor, robot.pyo_variables)) for motor in torques(robot))


def work_squared_penalty(robot: 'System3D', with_time: bool):
    if with_time:
        hm = robot.m.hm
        hm0 = robot.m.hm0.value
        nfe = len(robot.m.fe)

        # enumerate returns 0-based, pyomo expects 1-based
        return sum(
            sum(hm[(fe % nfe) + 1] * hm0 * P**2 for (fe, P) in enumerate(power(motor, robot.pyo_variables)))
            for motor in torques(robot))
    else:
        return sum(sum(P**2 for P in power(motor, robot.pyo_variables)) for motor in torques(robot))


def max_power_constraint(robot: 'System3D', maxpower: float):
    def powerlimit_f(m, fe: int):
        P = sum(sum(power(motor, robot.pyo_variables, fe)) for motor in torques(robot))
        return P**2 <= maxpower**2

    constraintname = f'{robot.name}_maxpower'
    assert not hasattr(robot.m, constraintname), \
        f'The model already has a constraint with the name {constraintname}'

    setattr(robot.m, constraintname, pyo.Constraint(robot.m.fe, rule=powerlimit_f))


def constrain_torque(m: pyo.ConcreteModel, Tc_set: pyo.Set, constr_name: str, torque: pyo.Var, lowerbound: float,
                     upperbound: float):
    """
    >>> constrain_torque(robot.m, 'knee', idx, knee, -0.5, , 0.5)
    """
    name = f'torque_limits_{constr_name}'
    setattr(m, name, pyo.Constraint(m.fe, Tc_set, rule=lambda m, fe, idx: (lowerbound, torque[fe, idx], upperbound)))
