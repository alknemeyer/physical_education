# TODO: this file really should be named better...
from pyomo.environ import Var
import numpy as np
from typing import Callable, Iterable, Optional, TYPE_CHECKING

from .foot import feet, feet_penalty
from . import motor
from . import utils


if TYPE_CHECKING:
    from .system import System3D
    from .links import Link3D


def init_vel_acc_from_pos(robot: 'System3D', link: 'Link3D', varname: str):
    """
    >>> robot['LBL']['dq'][:,:,'theta'].value = 0
    >>> robot['LBL']['ddq'][:,:,'theta'].value = 0
    >>> init_vel_acc_from_pos(robot['LBL'], 'theta')
    >>> robot['LBL'].plot()  # velocities and accelerations should look right
    """
    q, dq, ddq = link['q'], link['dq'], link['ddq']
    m = q.model()
    for fe, cp in robot.indices(one_based=True):
        if fe == 1:
            continue

        dt = m.hm0.value * m.hm[fe].value

        if not dq[fe, cp, varname].fixed and q[fe, cp, varname].value is not None and q[fe-1, cp, varname].value is not None:
            dq[fe, cp, varname].value = \
                (q[fe, cp, varname].value - q[fe-1, cp, varname].value)/dt

        if not ddq[fe, cp, varname].fixed and dq[fe, cp, varname].value is not None and dq[fe-1, cp, varname].value is not None:
            ddq[fe, cp, varname].value = \
                (dq[fe, cp, varname].value - dq[fe-1, cp, varname].value)/dt


# TODO: re-add derivatives once I know their scaling is correct, and add a noise parameter!
def sin_around_touchdown(fe: int, nfe: int) -> np.ndarray:
    """Returns position, acceleration and velocity. The leg is assumed to be moving backwards
    at the time of impact with the ground, with a forward leg having a negative angle

    >>> import matplotlib.pyplot as plt
    >>> for x, label in zip(sin_around_touchdown(10, 50), ('pos', 'vel', 'acc')):
    ...     plt.plot(x, label=label)
    >>> plt.gcf().set_size_inches(25,8); plt.grid(True); plt.legend(); plt.xticks(range(50)); plt.show()
    """
    assert 1 <= fe <= nfe
    π = np.pi  # type: ignore
    offset = (fe-1)/(nfe-1) * 2*π
    x = np.linspace(0, 2*π, num=nfe) - offset
    # , np.cos(x), -np.sin(x)  <- velocity and acceleration
    return np.sin(x)  # type: ignore


def bound_penalty_and_add_transport_cost(robot: 'System3D', penalty_limit: float):
    from pyomo.environ import Objective
    utils.info('Deleting previous cost function')
    robot.m.del_component('cost')

    nfe = len(robot.m.fe)
    ncp = len(robot.m.cp)

    for foot in feet(robot):
        contact_penalty = foot.pyomo_vars['contact_penalty']
        friction_penalty = foot.pyomo_vars['friction_penalty']
        slip_penalty = foot.pyomo_vars['slip_penalty']

        for fe, cp in robot.indices(one_based=True):
            contact_penalty[fe].setub(penalty_limit)
            friction_penalty[fe].setub(penalty_limit)
            slip_penalty[fe, cp].setub(penalty_limit)

    body = robot.links[0]
    torque_cost = sum(
        [Tc**2 for link in robot.links for Tc in link['Tc'][:, :]])
    transport_cost = torque_cost / body['q'][nfe, ncp, 'x']
    robot.m.cost = Objective(expr=transport_cost)

    pen_cost = feet_penalty(robot)
    return {'penalty': pen_cost, 'transport': transport_cost}


def remove_contact_constraints(robot: 'System3D'):
    m = utils.get_pyomo_model_or_error(robot)

    for foot in feet(robot):
        fh = foot['foot_height']
        grfz = foot['GRFz']
        for fe, cp in robot.indices(one_based=True):
            fh[fe, cp].fixed = False
            fh[fe, cp].setlb(0)
            grfz[fe, cp].fixed = False
            grfz[fe, cp].setlb(0)

    for constraint in [c for c in dir(m) if 'straight_leg' in c]:
        utils.remove_constraint_if_exists(m, constraint)


def add_costs(robot: 'System3D',
              include_transport_cost: bool,
              include_torque_cost: bool,
              transport_axis: str = 'x',
              scale: float = 0.001,
              **other_costs) -> dict:  # time cost? distance cost?
    from pyomo.environ import Objective
    m = utils.get_pyomo_model_or_error(robot)
    utils.remove_constraint_if_exists(m, 'cost')

    body = robot.links[0]
    nfe = len(robot.m.fe)
    ncp = len(robot.m.cp)

    torque_cost = (
        motor.torque_squared_penalty(robot)*scale
        if include_torque_cost else 0
    )

    transport_cost = (
        torque_cost / body['q'][nfe, ncp, transport_axis]*scale
        if include_transport_cost else 0
    )

    pen_cost = feet_penalty(robot)
    robot.m.cost = Objective(expr=pen_cost + transport_cost + torque_cost
                             + sum(v for v in other_costs.values()))

    return {'penalty': pen_cost, 'transport': transport_cost, 'torque': torque_cost, **other_costs}


def increase_motor_limits(robot: 'System3D', *, torque_bound: float, no_load_speed: float):
    """
    >>> robot.make_pyomo_model(nfe=10,  collocation='implicit_euler', total_time=0.3, scale_forces_by=30.)
    >>> increase_motor_limits(robot, torque_bound=5., no_load_speed=100.)
    >>> ol.motor.torques(robot)[0]['Tc'].pprint()
    """
    # make sure it has a pyomo model
    utils.get_pyomo_model_or_error(robot)

    for motor_ in motor.torques(robot):
        for Tc in motor_['Tc'][:, :]:
            Tc.setub(+torque_bound)
            Tc.setlb(-torque_bound)

        if hasattr(motor_, 'torque_speed_limit'):
            tsp = motor_.torque_speed_limit
            tsp.torque_bounds = (-torque_bound, torque_bound)
            tsp.no_load_speed = no_load_speed


# UNTESTED
def fecp2val(nfe: int, ncp: int):
    def _(fe: int, cp: int):
        assert 1 <= fe <= nfe and 1 <= cp <= ncp, "fe and cp are one-based"
        return (fe-1 + (cp-1)/(ncp-1))*(nfe-1)
    return _


# UNTESTED
def init_with_function(var: Var,
                       func: Callable[[float], float],
                       lowerbound: Optional[float] = None,
                       otherinds: Iterable = (),
                       fixiflt0: bool = False):  # fix if less than 0
    """ func called with 0..1
    >>> init_with_function(var, lambda i: math.cos(0.2 + 2*math.pi*i)/5, lowerbound=0)
    """
    if lowerbound is None:
        lowerbound = float('-inf')

    m = var.model()
    nfe = len(m.fe)
    ncp = len(m.cp)

    for fe, cp in utils.get_indexes(nfe, ncp, one_based=True, skipfirst=False):
        val = var.__getindex__((fe, cp, *otherinds))

        i = fe/nfe + cp/ncp  # TODO: do this calculation properly
        val.value = max(lowerbound, func(i))

        if fixiflt0 is True and val.value <= 0:
            val.fixed = True
