"""
Some tasks for the robots in the models folder.
"""
from typing import Dict, Any
from .. import utils
from pyomo.environ import Objective
from ..foot import feet_penalty
from ..motor import torque_squared_penalty


def drop_test(robot, *, z_rot: float, min_torque: bool, initial_height: float = 1.) -> Dict[str, Any]:
    """Params which have been tested for this task:
    nfe = 20, total_time = 1.0, vary_timestep_with=(0.8,1.2), 5 mins for solving

    if min_torque is True, quite a bit more time is needed as IPOPT refines things
    """
    nfe = len(robot.m.fe)
    ncp = len(robot.m.cp)

    tested_models = ('3D monoped', '3D biped',
                     '3D quadruped', '3D prismatic monoped')
    if not robot.name in tested_models:
        utils.warn(f'This robot configuration ("{robot.name}") hasn\'t been tested!')

    body = robot['base_B'] if robot.name == '3D quadruped' else robot['base']

    # start at the origin
    body['q'][1, ncp, 'x'].fix(0)
    body['q'][1, ncp, 'y'].fix(0)
    body['q'][1, ncp, 'z'].fix(initial_height)

    # fix initial angle
    for link in robot.links:
        for ang in ('phi', 'theta'):
            link['q'][1, ncp, ang].fix(0)

        link['q'][1, ncp, 'psi'].fix(z_rot)

    # start stationary
    for link in robot.links:
        for q in link.pyomo_sets['q_set']:
            link['dq'][1, ncp, q].fix(0)

    # init to y plane
    for link in robot.links:
        for ang in ('phi', 'theta'):
            link['q'][:, :, ang].value = 0

        link['q'][:, :, 'psi'].value = z_rot

    # legs slightly forward at the end
    uplopairs = (('upper', 'lower'),) if robot.name == '3D monoped' \
        else (('UL', 'LL'), ('UR', 'LR')) if robot.name == '3D biped' \
        else (('UFL', 'LFL'), ('UFR', 'LFR'), ('UBL', 'LBL'), ('UBR', 'LBR')) if robot.name == '3D quadruped' \
        else tuple()  # <- iterating over this will result in the body not being evaluated

    for upper, lower in uplopairs:
        ang = 0.01 if not (
            robot.name == '3D quadruped' and upper[1] == 'B') else -0.01
        robot[upper]['q'][nfe, ncp, 'theta'].setlb(ang)
        robot[lower]['q'][nfe, ncp, 'theta'].setub(-ang)

    # but not properly fallen over
    body['q'][nfe, ncp, 'z'].setlb(0.2)

    # objective: reduce CoT, etc
    utils.remove_constraint_if_exists(robot.m, 'cost')

    torque_cost = torque_squared_penalty(robot)
    pen_cost = feet_penalty(robot)
    robot.m.cost = Objective(expr=(torque_cost if min_torque else 0)
                             + 1000*pen_cost)

    return {'torque': torque_cost, 'penalty': pen_cost}
