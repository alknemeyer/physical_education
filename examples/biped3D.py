"""
    Number of operations in EOM without simplification: 186 903. With utils.parsimp: 

    Example usage:
    >>> import physical_education as pe
    >>> robot, add_pyomo_constraints = pe.models.biped3D()
    >>> robot.calc_eom(simp_func = lambda x: utils.parsimp(x, nprocs=12))
    >>> robot.make_pyomo_model(nfe=30, collocation='euler', total_time=0.7, vary_timestep_within=(0.8, 1.2))
    >>> add_pyomo_constraints(robot)
    >>> costs = pe.models.tasks.drop_test(robot, z_rot=0.2, min_torque=False, initial_height=1.0)
    >>> oe.utils.default_solver(max_mins=10, solver='ma86', OF_hessian_approximation='limited-memory').solve(robot.m, tee=True)
    >>> robot.post_solve(costs)
    >>> robot.animate(view_along=(35, -120), t_scale=2, track='base')
"""

from typing import Callable, Tuple
from ..links import Link3D, constrain_rel_angle
from ..system import System3D
from sympy import Matrix as Mat
from ..foot import add_foot
from ..motor import add_torque


def model() -> Tuple[System3D, Callable[[System3D], None]]:
    body = Link3D('base', '+y', base=True)

    # for stabilization
    tail = Link3D('tail', '-x', start_I=body.Pb_I)
    body.add_hookes_joint(tail, about='xy')
    add_torque(body, tail, about='xy')

    # upper left, lower left
    L_thigh = Link3D('UL', '-z', start_I=body.Pb_I +
                     body.Rb_I @ Mat([0, -body.length/2, 0]))
    body.add_hookes_joint(L_thigh, about='xy')
    add_torque(body, L_thigh, about='xy')

    L_calf = Link3D('LL', '-z', start_I=L_thigh.bottom_I)
    add_foot(L_calf, at='bottom', nsides=8, friction_coeff=1.)
    L_thigh.add_revolute_joint(L_calf, about='y')
    add_torque(L_thigh, L_calf, about='y')

    # upper right, lower right
    R_thigh = Link3D('UR', '-z', start_I=body.Pb_I +
                     body.Rb_I @ Mat([0, body.length/2, 0]))
    body.add_hookes_joint(R_thigh, about='xy')
    add_torque(body, R_thigh, about='xy')

    R_calf = Link3D('LR', '-z', start_I=R_thigh.bottom_I)
    add_foot(R_calf, at='bottom', nsides=8, friction_coeff=1.)
    R_thigh.add_revolute_joint(R_calf, about='y')
    add_torque(R_thigh, R_calf, about='y')

    # combine into a robot
    robot = System3D(
        '3D biped', [body, tail, L_thigh, L_calf, R_thigh, R_calf])

    return robot, add_pyomo_constraints


def add_pyomo_constraints(robot: System3D):
    assert robot.m is not None,\
        'robot does not have a pyomo model defined on it'

    from math import pi as π
    body, tail, L_thigh, L_calf, R_thigh, R_calf = [
        link['q'] for link in robot.links]

    # tail
    constrain_rel_angle(robot.m, 'tail_pitch',
                        -π/2, body[:, :, 'theta'], tail[:, :, 'theta'], π/2)

    constrain_rel_angle(robot.m, 'tail_taw',
                        -π/2, body[:, :, 'psi'], tail[:, :, 'psi'], π/2)

    # left leg
    constrain_rel_angle(robot.m, 'left_hip',
                        -π/2, body[:, :, 'theta'], L_thigh[:, :, 'theta'], π/2)

    constrain_rel_angle(robot.m, 'left_knee',
                        0, L_thigh[:, :, 'theta'], L_calf[:, :, 'theta'], π)

    # right leg
    constrain_rel_angle(robot.m, 'right_hip',
                        -π/2, body[:, :, 'theta'], R_thigh[:, :, 'theta'], π/2)

    constrain_rel_angle(robot.m, 'right_knee',
                        0, R_thigh[:, :, 'theta'], R_calf[:, :, 'theta'], π)
