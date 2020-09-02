"""
    Number of operations in EOM without simplification: 102 480. With utils.parsimp: 7776

    Example usage:
    >>> import optim_lib; utils = optim_lib.utils
    >>> robot, add_pyomo_constraints = optim_lib.models.monoped3D()
    >>> robot.calc_eom(simp_func = lambda x: utils.parsimp(x, nprocs=8))
    >>> robot.make_pyomo_model(nfe=30, collocation='euler', total_time=0.7, vary_timestep_within=(0.8, 1.2))
    >>> add_pyomo_constraints(robot)
    >>> costs = optim_lib.models.tasks.drop_test(robot, z_rot=0.2, min_torque=False, initial_height=1.0)
    >>> utils.default_solver(max_mins=10, solver='ma86', OF_hessian_approximation='limited-memory').solve(robot.m, tee=True)
    >>> robot.post_solve(costs)
    >>> robot.animate(view_along=(35, -120), t_scale=2, track='base')
"""

from typing import Tuple, Callable
from ..links import Link3D, constrain_rel_angle
from ..system import System3D
from ..foot import add_foot
from ..motor import add_torque


def model() -> Tuple[System3D, Callable[[System3D], None]]:
    # create links: body, upper leg, lower leg
    base = Link3D('base', '+x', base=True,
                  mass=5., radius=0.4, length=0.4,
                  meta=['spine'])

    upper = Link3D('upper', '-z', start_I=base.Pb_I,
                   mass=.6, radius=0.01, length=0.25,
                   meta=['leg', 'thigh'])

    lower = Link3D('lower', '-z', start_I=upper.bottom_I,
                   mass=.4, radius=0.01, length=0.25,
                   meta=['leg', 'calf'])

    add_foot(lower, at='bottom', nsides=8, friction_coeff=1.)

    # add relationships between links
    base.add_hookes_joint(upper, about='xy')
    add_torque(base, upper, about='xy',
               torque_bounds=(-2., 2.), no_load_speed=20)

    upper.add_revolute_joint(lower, about='y')
    add_torque(upper, lower, about='y', torque_bounds=(-2., 2.),
               no_load_speed=20)

    # combine into a robot
    robot = System3D('3D monoped', [base, upper, lower])

    return robot, add_pyomo_constraints


def add_pyomo_constraints(robot: System3D):
    assert robot.m is not None,\
        'robot does not have a pyomo model defined on it'

    from math import pi as π
    body, thigh, calf = [link['q'] for link in robot.links]

    constrain_rel_angle(robot.m, 'hip',
                        -π/2, body[:, :, 'theta'], thigh[:, :, 'theta'], π/2)

    constrain_rel_angle(robot.m, 'knee',
                        0, thigh[:, :, 'theta'], calf[:, :, 'theta'], π)
