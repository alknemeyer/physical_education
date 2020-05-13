"""
    Number of operations in EOM without simplification: 186 903. With utils.parsimp: 

    Example usage:
    >>> import optim_lib; utils = optim_lib.utils
    >>> robot, add_pyomo_constraints = optim_lib.models.biped3D()
    >>> robot.calc_eom(simp_func = lambda x: utils.parsimp(x, nprocs=12))
    >>> robot.make_pyomo_model(nfe=30, collocation='euler', total_time=0.7, vary_timestep_within=(0.8, 1.2))
    >>> add_pyomo_constraints(robot)
    >>> costs = optim_lib.models.tasks.drop_test(robot, z_rot=0.2, min_torque=False, initial_height=1.0)
    >>> utils.default_solver(max_mins=10, solver='ma86', OF_hessian_approximation='limited-memory').solve(robot.m, tee=True)
    >>> robot.post_solve(costs)
    >>> robot.animate(view_along=(35, -120), t_scale=2, track='base')
"""

from typing import Callable, Tuple
from ..links import Link3D, constrain_rel_angle
from ..system import System3D
from sympy import Matrix as Mat

# TODO: add meta information?

def biped3D() -> Tuple[System3D,Callable[[System3D],None]]:
    from ..utils import warn
    warn('this probably needs fixing! especially the x/y/z and top_I stuff!')
    warn('also, need to set masses and torques!')
    warn('also, would probably be more useful as a model of a human')
    raise NotImplementedError('complete biped3D!!')

    body = Link3D('base', '+y', base=True)

    # upper left, lower left
    L_thigh = Link3D('UL', '-z', start_I=body.Pb_I + body.Rb_I @ Mat([0, -body.length/2, 0]))
    L_calf = Link3D('LL', '-z', start_I=L_thigh.bottom_I)\
        .add_foot(at='bottom', nsides=8, friction_coeff=1.)

    # upper right, lower right
    R_thigh = Link3D('UR', '-z', start_I=body.Pb_I + body.Rb_I @ Mat([0, body.length/2, 0]))
    R_calf = Link3D('LR', '-z', start_I=R_thigh.bottom_I)\
        .add_foot(at='bottom', nsides=8, friction_coeff=1.)

    # add relationships between links
    body.add_hookes_joint(L_thigh, about='xy')\
             .add_input_torques_at(L_thigh, about='xy')

    L_thigh.add_revolute_joint(L_calf, about='y')\
           .add_input_torques_at(L_calf, about='y')

    body.add_hookes_joint(R_thigh, about='xy')\
             .add_input_torques_at(R_thigh, about='xy')

    R_thigh.add_revolute_joint(R_calf, about='y')\
           .add_input_torques_at(R_calf, about='y')

    # combine into a robot
    robot = System3D('3D biped', [body, L_thigh, L_calf, R_thigh, R_calf])
    
    return robot, add_pyomo_constraints

def add_pyomo_constraints(robot: System3D):
    from math import pi as π
    body, L_thigh, L_calf, R_thigh, R_calf = [link['q'] for link in robot.links]

    constrain_rel_angle(robot.m, 'left_hip',
                        -π/2, body[:,:,'theta'], L_thigh[:,:,'theta'], π/2)
    
    constrain_rel_angle(robot.m, 'left_knee',
                        0, L_thigh[:,:,'theta'], L_calf[:,:,'theta'], π)
    
    constrain_rel_angle(robot.m, 'right_hip',
                        -π/2, body[:,:,'theta'], R_thigh[:,:,'theta'], π/2)
    
    constrain_rel_angle(robot.m, 'right_knee',
                        0, R_thigh[:,:,'theta'], R_calf[:,:,'theta'], π)
