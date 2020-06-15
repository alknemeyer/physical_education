from typing import Tuple, Callable
from ..links import Link3D, constrain_rel_angle
from ..system import System3D
from sympy import Matrix as Mat
from math import pi as π

def quadruped3D() -> Tuple[System3D,Callable[[System3D],None]]:
    """
    A model of cheetah 6 from Functional anatomy of the cheetah (Acinonyx jubatus) forelimb and hindlimb
    doi: 10.1111/j.1469-7580.2011.01344.x and 10.1111/j.1469-7580.2010.01310.x

    Roughly 500 000 operations in the equations of motion without simplification, and 142 520 if simplified with
    >>> robot.calc_eom(simp_func = lambda x: utils.parsimp(x, nprocs = 14))
    """
    # create front and back links of body and tail
    body_B = Link3D('base_B', '+x', base=True, mass=17., radius=0.08, length=0.41,
                    meta=['spine', 'back'])
    body_F = Link3D('base_F', '+x', start_I=body_B.bottom_I, mass=8., radius=0.08, length=0.21,
                    meta=['spine', 'front'])

    body_B.add_hookes_joint(body_F, about='xy')\
        .add_input_torques_at(body_F, about='xy')

    tail0 = Link3D('tail0', '-x', start_I=body_B.top_I, mass=0.4, radius=0.005, length=0.38,
                   meta=['tail'])
    tail1 = Link3D('tail1', '-x', start_I=tail0.bottom_I, mass=0.2, radius=0.005, length=0.38,
                   meta=['tail'])\
        .add_foot(at='bottom', nsides=8, friction_coeff=1.3)  # TODO: maybe rather friction = 0 ?

    body_B.add_hookes_joint(tail0, about='xy')\
        .add_input_torques_at(tail0, about='xy')
    tail0.add_hookes_joint(tail1, about='xy')\
        .add_input_torques_at(tail1, about='xy')\

    def def_leg(body: Link3D, front: bool, right: bool) -> Tuple[Link3D,Link3D]:
        """Define a leg and attach it to the front/back right/left of `body`.
            Only really makes sense when `body` is aligned along the `x`-axis"""
        # maybe flip x (or y)
        mfx = lambda x: x if front else -x
        mfy = lambda y: y if right else -y  # typo... these should be swapped. But test straight afterwards in case it breaks something!

        thigh_mas, calf_mas = (0.171, 0.068) if front else (0.210, 0.160)
        thigh_len, calf_len = (0.254, 0.247) if front else (0.281, 0.287)
        thigh_rad, calf_rad = (0.012, 0.005) if front else (0.010, 0.011)

        start_I = body.Pb_I + body.Rb_I @ Mat([mfx(body.length/2), mfy(body.radius), 0])
        suffix = ('F' if front else 'B') + ('R' if right else 'L')

        thigh = Link3D('U'+suffix, '-z', start_I=start_I, mass=thigh_mas, length=thigh_len, radius=thigh_rad,
                       meta=['leg', 'thigh', 'front' if front else 'back', 'right' if right else 'left'])

        calf  = Link3D('L'+suffix, '-z', start_I=thigh.bottom_I, mass=calf_mas, length=calf_len, radius=calf_rad,
                       meta=['leg', 'calf', 'front' if front else 'back', 'right' if right else 'left'])\
            .add_foot(at='bottom', nsides=8, friction_coeff=1.3)
        
        body.add_hookes_joint(thigh, about='xy')\
            .add_input_torques_at(thigh, about='xy')

        thigh.add_revolute_joint(calf, about='y')\
            .add_input_torques_at(calf, about='y')
        
        return thigh, calf
    
    ufl, lfl = def_leg(body_F, front=True, right=False)
    ufr, lfr = def_leg(body_F, front=True, right=True)
    ubl, lbl = def_leg(body_B, front=False, right=False)
    ubr, lbr = def_leg(body_B, front=False, right=True)

    # combine into a robot
    robot = System3D('3D quadruped', [body_B, body_F, tail0, tail1,
                                      ufl, lfl, ufr, lfr,
                                      ubl, lbl, ubr, lbr])
    return robot, add_pyomo_constraints

def add_pyomo_constraints(robot: System3D) -> None:
    link_body_B, link_body_F, link_tail0, link_tail1, \
    link_UFL, link_LFL, link_UFR, link_LFR, \
    link_UBL, link_LBL, link_UBR, link_LBR = [link['q'] for link in robot.links]
    
    # spine can't bend too much:
    constrain_rel_angle(robot.m, 'spine_pitch',
                             -π/3, link_body_B[:,:,'theta'], link_body_F[:,:,'theta'], π/3)
    constrain_rel_angle(robot.m, 'spine_yaw',
                             -π/3, link_body_B[:,:,'psi'], link_body_F[:,:,'psi'], π/3)
    #constrain_rel_angle(robot.m, 'spine_roll',
    #                         -π/3, link_body_B[:,:,'phi'], link_body_F[:,:,'phi'], π/3)

    # tail can't go too crazy:
    constrain_rel_angle(robot.m, 'tail_body_pitch',
                        -π/3, link_body_B[:,:,'theta'], link_tail0[:,:,'theta'], π/3)
    constrain_rel_angle(robot.m, 'tail_body_yaw',
                        -π/3, link_body_B[:,:,'psi'], link_tail0[:,:,'psi'], π/3)
    #constrain_rel_angle(robot.m, 'tail_body_roll',
    #                    -π/3, link_body_B[:,:,'phi'], link_tail0[:,:,'phi'], π/3)
    
    constrain_rel_angle(robot.m, 'tail_tail_pitch',
                        -π/2, link_tail0[:,:,'theta'], link_tail1[:,:,'theta'], π/2)
    constrain_rel_angle(robot.m, 'tail_tail_yaw',
                        -π/2, link_tail0[:,:,'psi'], link_tail1[:,:,'psi'], π/2)
    #constrain_rel_angle(robot.m, 'tail_tail_roll',
    #                    -π/2, link_tail0[:,:,'psi'], link_tail1[:,:,'psi'], π/2)

    # legs: hip abduction and knee
    for body, thigh, calf, name in ((link_body_F, link_UFL, link_LFL, 'FL'),
                                    (link_body_F, link_UFR, link_LFR, 'FR'),
                                    (link_body_B, link_UBL, link_LBL, 'BL'),
                                    (link_body_B, link_UBR, link_LBR, 'BR')):
        constrain_rel_angle(robot.m, name + '_hip_pitch',
                            -π/2, body[:,:,'theta'], thigh[:,:,'theta'], π/2)
        constrain_rel_angle(robot.m, name + '_hip_aduct',
                            -π/4, body[:,:,'psi'], thigh[:,:,'psi'], π/4)
        #constrain_rel_angle(robot.m, name + '_hip_roll',
        #                    -π/4, body[:,:,'phi'], thigh[:,:,'phi'], π/4)
        
        lo, up = (-π, 0) if name.startswith('B') else (0, π)
        constrain_rel_angle(robot.m, name + '_knee',
                            lo, thigh[:,:,'theta'], calf[:,:,'theta'], up)
