from typing import Any, Dict, Iterable, Optional, Tuple, Callable
from math import pi as π
from sympy import Matrix as Mat
from ..links import Link3D, constrain_rel_angle
from ..system import System3D
from ..foot import add_foot, feet
from ..motor import add_torque
from ..drag import add_drag
from ..spring import add_torquespring


parameters = {
    'model-6': {
        'source': """
            A model of cheetah 6 from Functional anatomy of the cheetah (Acinonyx jubatus) forelimb and hindlimb
            doi: 10.1111/j.1469-7580.2011.01344.x and 10.1111/j.1469-7580.2010.01310.x
        """,
        'body_B': {'mass': 17., 'radius': 0.08, 'length': 0.41},
        'body_F': {'mass': 8., 'radius': 0.08, 'length': 0.21},
        'tail0': {'mass': 0.4, 'radius': 0.005, 'length': 0.38},
        'tail1': {'mass': 0.2, 'radius': 0.005, 'length': 0.38},
        'front': {
            'thigh': {'mass': 0.171, 'radius': 0.012, 'length': 0.254},
            'calf':  {'mass': 0.068, 'radius': 0.005, 'length': 0.247},
        },
        'back': {
            'thigh': {'mass': 0.210, 'radius': 0.010, 'length': 0.281},
            'calf':  {'mass': 0.160, 'radius': 0.011, 'length': 0.287},
        },
        'friction_coeff': 1.3,
        'motor_params': {'torque_bounds': (-2., 2.), 'no_load_speed': 50.},
    },
    'mean-male': {
        'source': """
        Parameters for the 'mean' (X) cheetah from
        Morphology, Physical Condition, and Growth of the Cheetah (Acinonyx jubatus jubatus)
        https://academic.oup.com/jmammal/article/84/3/840/905900

            body mass = 45.6 kg  ---> majority (42kg?) in body
            chest girth = 71.7 cm  ---> front radius = 0.717m / (2*pi)
            abdomen girth = 59.4 cm  ---> back radius = 0.594m / (2*pi)
            skull length = 23.4 cm
            body length = 125.5 cm  ---> body - skull - neck = 125.5 - 23.4 - (20?) = 80cm => front = 0.5m, back = 0.3m
            tail length = 76.7 cm  ---> 38cm per half
            total length = 202.2 cm
            total foreleg length = 77 cm
            total hind leg length = 81.1 cm
            front foot length = 8.2 cm
            front foot width = 6.1 cm
            hind foot length = 9.2 cm
            hind foot width = 6.2 cm

        From "Quasi-steady state aerodynamics of the cheetah tail"
            fur length on tail = 10mm on average
            average tail diameter (no fur) = 31mm
                ---> radius = 31/2 + 10 = 25.5mm = 0.0255m

        Friction coeff of 1.3 from
        "Locomotion dynamics of hunting in wild cheetahs"

        NOTE: leg measurements mostly cribbed from 'model-6' above. Find proper values!
              lengths = same
              masses = same * 1.2
              radii = same

        NOTE: the motor_params values are mostly made up. In any case, different muscle
              groups would need different values
        """,
        'body_B': {'mass': 28., 'radius': 0.594/(2*π), 'length': 0.3},
        'body_F': {'mass': 14., 'radius': 0.717/(2*π), 'length': 0.5},
        'tail0': {'mass': 0.4, 'radius': 0.0255, 'length': 0.38},
        'tail1': {'mass': 0.2, 'radius': 0.0255, 'length': 0.38},
        'front': {
            'thigh': {'mass': 0.171*1.2, 'radius': 0.012, 'length': 0.254},
            'calf':  {'mass': 0.068*1.2, 'radius': 0.005, 'length': 0.247},
        },
        'back': {
            'thigh': {'mass': 0.210*1.2, 'radius': 0.010, 'length': 0.281},
            'calf':  {'mass': 0.160*1.2, 'radius': 0.011, 'length': 0.287},
        },
        'friction_coeff': 1.3,
        'motor_params': {'torque_bounds': (-3., 3.), 'no_load_speed': 50.},
        'tail_motor_params': {'torque_bounds': (-1., 1.), 'no_load_speed': 50.}
    },
}


def model(params: Dict[str, Any]) -> Tuple[System3D, Callable[[System3D], None]]:
    """
    Roughly 400 000 operations in the equations of motion without simplification,
    and 142 520 if simplified with
    >>> robot.calc_eom(simp_func = lambda x: utils.parsimp(x, nprocs = 14))
    """
    # create front and back links of body and tail
    body_B = Link3D('base_B', '+x', base=True, **params['body_B'],
                    meta=['spine', 'back'])

    body_F = Link3D('base_F', '+x', start_I=body_B.bottom_I, **params['body_F'],
                    meta=['spine', 'front'])

    body_B.add_hookes_joint(body_F, about='xyz')
    add_torque(body_B, body_F, about='xyz', **params['motor_params'])

    tail0 = Link3D('tail0', '-x', start_I=body_B.top_I,
                   **params['tail0'], meta=['tail'])
    tail1 = Link3D('tail1', '-x', start_I=tail0.bottom_I,
                   **params['tail1'], meta=['tail'])

    # TODO: maybe rather friction = 0 ?
    add_foot(tail1, at='bottom', nsides=8, friction_coeff=0.1,
             GRFxy_max=0.1, GRFz_max=0.1)

    # input torques to tail
    body_B.add_hookes_joint(tail0, about='xy')
    add_torque(body_B, tail0, about='xy', **params['tail_motor_params'])

    tail0.add_hookes_joint(tail1, about='xy')
    add_torque(tail0, tail1, about='xy', **params['tail_motor_params'])

    # drag on body and tail
    add_drag(body_F, at=body_F.bottom_I, name='body_F-drag-head',
             use_dummy_vars=True, cylinder_top=True)
    add_drag(body_F, at=body_F.Pb_I, name='body_F-drag-body',
             use_dummy_vars=True)
    add_drag(body_B, at=body_B.Pb_I, use_dummy_vars=True)
    add_drag(tail0, at=tail0.Pb_I, use_dummy_vars=True)
    add_drag(tail1, at=tail1.Pb_I, use_dummy_vars=True)

    # spring forces on spine
    phi_b, th_b, psi_b = body_B.q[3:]
    phi_f, th_f, psi_f = body_F.q[:3]
    add_torquespring(body_B, body_F, phi_b - phi_f, spring_coeff=2.0,
                     rest_angle=0, name='spine-torquespring-roll')
    add_torquespring(body_B, body_F, th_b - th_f, spring_coeff=0.5,
                     rest_angle=0, name='spine-torquespring-pitch')
    add_torquespring(body_B, body_F, psi_b - psi_f, spring_coeff=2.0,
                     rest_angle=0, name='spine-torquespring-yaw')

    def def_leg(body: Link3D, front: bool, right: bool) -> Tuple[Link3D, Link3D]:
        """Define a leg and attach it to the front/back right/left of `body`.
            Only really makes sense when `body` is aligned along the `x`-axis"""
        # maybe flip x (or y)
        # the model is considered to face along the x axis (so front/back
        # refers to changes in the y value).
        def mfx(x): return x if front else -x
        def mfy(y): return y if right else -y

        start_I = body.Pb_I + \
            body.Rb_I @ Mat([mfx(body.length/2), mfy(body.radius), 0])
        suffix = ('F' if front else 'B') + ('R' if right else 'L')

        p = params['front'] if front else params['back']

        thigh = Link3D('U'+suffix, '-z', start_I=start_I, **p['thigh'],
                       meta=['leg', 'thigh', 'front' if front else 'back', 'right' if right else 'left'])

        calf = Link3D('L'+suffix, '-z', start_I=thigh.bottom_I, **p['calf'],
                      meta=['leg', 'calf', 'front' if front else 'back', 'right' if right else 'left'])
        add_foot(calf, at='bottom', nsides=8,
                 friction_coeff=params['friction_coeff'])

        body.add_hookes_joint(thigh, about='xy')
        add_torque(body, thigh, about='xy', **params['motor_params'])

        thigh.add_revolute_joint(calf, about='y')
        add_torque(thigh, calf, about='y', **params['motor_params'])

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
    # π/3 = 60 degrees
    # π/2 = 90 degrees
    # π/4 = 45 degrees

    assert robot.m is not None,\
        'robot does not have a pyomo model defined on it'

    link_body_B, link_body_F, link_tail0, link_tail1, \
        link_UFL, link_LFL, link_UFR, link_LFR, \
        link_UBL, link_LBL, link_UBR, link_LBR = [
            link['q'] for link in robot.links]

    # spine can't bend too much:
    # it only has pitch and roll relative degrees of freedom. No need to constrain yaw
    constrain_rel_angle(robot.m, 'spine_pitch',
                        -π/4, link_body_B[:, :, 'theta'], link_body_F[:, :, 'theta'], π/4)
    constrain_rel_angle(robot.m, 'spine_roll',
                        -π/4, link_body_B[:, :, 'phi'], link_body_F[:, :, 'phi'], π/4)

    # tail can't go too crazy:
    constrain_rel_angle(robot.m, 'tail_body_pitch',
                        -π/3, link_body_B[:, :, 'theta'], link_tail0[:, :, 'theta'], π/3)
    constrain_rel_angle(robot.m, 'tail_body_yaw',
                        -π/3, link_body_B[:, :, 'phi'], link_tail0[:, :, 'phi'], π/3)

    constrain_rel_angle(robot.m, 'tail_tail_pitch',
                        -π/2, link_tail0[:, :, 'theta'], link_tail1[:, :, 'theta'], π/2)
    constrain_rel_angle(robot.m, 'tail_tail_yaw',
                        -π/2, link_tail0[:, :, 'phi'], link_tail1[:, :, 'phi'], π/2)

    # legs: hip abduction and knee
    for body, thigh, calf, name in ((link_body_F, link_UFL, link_LFL, 'FL'),
                                    (link_body_F, link_UFR, link_LFR, 'FR'),
                                    (link_body_B, link_UBL, link_LBL, 'BL'),
                                    (link_body_B, link_UBR, link_LBR, 'BR')):
        constrain_rel_angle(robot.m, name + '_hip_pitch',
                            -π/2, body[:, :, 'theta'], thigh[:, :, 'theta'], π/2)
        constrain_rel_angle(robot.m, name + '_hip_aduct',
                            -π/8, body[:, :, 'phi'], thigh[:, :, 'phi'], π/8)

        lo, up = (-π, 0) if name.startswith('B') else (0, π)
        constrain_rel_angle(robot.m, name + '_knee',
                            lo, thigh[:, :, 'theta'], calf[:, :, 'theta'], up)

# common functions


def high_speed_stop(robot: System3D, initial_vel: float, minimize_distance: bool,
                    gallop_data: Optional[dict] = None, offset: int = 0):
    import math
    import random
    from ..utils import copy_state_init
    from ..init_tools import add_costs

    nfe = len(robot.m.fe)
    ncp = len(robot.m.cp)
    total_time = float((nfe-1)*robot.m.hm0.value)

    body = robot['base_B']

    # start at the origin
    body['q'][1, ncp, 'x'].fix(0)
    body['q'][1, ncp, 'y'].fix(0)

    if gallop_data is not None:
        for fed, cpd in robot.indices(one_based=True):
            robot.init_from_dict_one_point(
                gallop_data, fed=fed, cpd=cpd, fes=(fed-1 + offset) % nfe, cps=0,
                skip_if_fixed=True, skip_if_not_None=False, fix=False)

        for link in robot.links:
            for q in link.pyomo_sets['q_set']:
                link['q'][1, ncp, q].fixed = True
                link['dq'][1, ncp, q].fixed = True
    else:
        # init to y plane
        body['q'][:, :, 'y'].value = 0
        for link in robot.links:
            for ang in ('phi', 'psi'):
                link['q'][:, :, ang].value = 0
                link['dq'][:, :, ang].value = 0
                link['ddq'][:, :, ang].value = 0

        # roughly bound to y plane
        for fe, cp in robot.indices(one_based=True):
            body['q'][fe, cp, 'y'].setub(0.2)
            body['q'][fe, cp, 'y'].setlb(-0.2)

        for link in robot.links:
            for ang in ('phi', 'psi'):
                for fe, cp in robot.indices(one_based=True):
                    link['q'][fe, cp, ang].setub(math.pi/4)
                    link['q'][fe, cp, ang].setlb(-math.pi/4)

        # bound theta
        for fe, cp in robot.indices(one_based=True):
            for link in robot.links[4:]:  # all leg segments - no tail or body
                link['q'][fe, cp, 'theta'].setub(math.radians(60))
                link['q'][fe, cp, 'theta'].setlb(math.radians(-60))
            for link in robot.links[:2]:  # two body segments
                link['q'][fe, cp, 'theta'].setub(math.radians(45))
                link['q'][fe, cp, 'theta'].setlb(math.radians(-45))

        for link in robot.links:
            for fe, cp in robot.indices(one_based=True):
                link['q'][fe, cp, 'theta'].value = (
                    math.radians(random.gauss(0, 15)))

        body['q'][1, ncp, 'z'].fix(0.6)

        # both sides mirrored
        for src, dst in (('UFL', 'UFR'), ('LFL', 'LFR'), ('UBL', 'UBR'), ('LBL', 'LBR')):
            copy_state_init(robot[src]['q'], robot[dst]['q'])

        # init tail to flick?
        for link in robot.links[2:4]:
            for fe, cp in robot.indices(one_based=True):
                link['q'][fe, cp, 'theta'].value = (
                    math.radians(random.random()*60))

        # stop weird local minimum where it bounces
        for fe, cp in robot.indices(one_based=True):
            if fe in range(10): continue
            # if fe > nfe/2: continue

            height = body['q'][fe, cp, 'z']
            height.setub(0.6)  # approx. leg height
        
            for foot in feet(robot):
                foot['foot_height'][fe,cp].setub(0.01)

    # start at speed
    body['dq'][1, ncp, 'x'].fix(initial_vel)

    # end at rest
    for link in robot.links:
        for q in link.pyomo_sets['q_set']:
            link['dq'][nfe, ncp, q].fix(0)

    # end in a fairly standard position
    for link in robot.links[:2]:  # two body segments
        link['q'][nfe, ncp, 'theta'].setub(math.radians(10))
        link['q'][nfe, ncp, 'theta'].setlb(math.radians(-10))
    for link in robot.links[4:]:  # leaving out tail - it might flail, which is good
        link['q'][nfe, ncp, 'theta'].setub(math.radians(20))
        link['q'][nfe, ncp, 'theta'].setlb(math.radians(-20))

    for link in robot.links:
        for ang in ('phi', 'psi'):
            link['q'][nfe, ncp, ang].setub(math.radians(5))
            link['q'][nfe, ncp, ang].setlb(math.radians(-5))

    # position and velocity over time
    for fe in robot.m.fe:
        pos = total_time * (initial_vel/2) * (fe-1)/(nfe-1)
        vel = initial_vel * (1 - (fe-1)/(nfe-1))
        # print('pos', pos, 'vel', vel)
        body['q'][fe, :, 'x'].value = pos
        body['dq'][fe, :, 'x'].value = vel

    # objective
    distance_cost = body['q'][nfe, ncp, 'x'] if minimize_distance else 0
    return add_costs(robot, include_transport_cost=False, include_torque_cost=False,
                     distance_cost=0.0001*distance_cost)


def periodic_gallop_test(robot,
                         avg_vel: float,
                         feet: Iterable,
                         foot_order_vals: Iterable,
                         init_from_dict: Optional[dict] = None,
                         at_angle_d: Optional[float] = None
                         ):
    """
    foot_order_vals = ((1, 7), (6, 13), (31, 38), (25, 32))  # 14 m/s
    """
    import math
    from math import radians
    import random
    from ..utils import constrain_total_time
    from ..foot import prescribe_contact_order
    from ..init_tools import sin_around_touchdown, add_costs
    from ..constrain import straight_leg, periodic

    nfe = len(robot.m.fe)
    ncp = len(robot.m.cp)
    total_time = float((nfe-1)*robot.m.hm0.value)

    constrain_total_time(robot.m, total_time=total_time)

    body = robot['base_B']

    # start at the origin
    body['q'][1, ncp, 'x'].fix(0)
    body['q'][1, ncp, 'y'].fix(0)

    if init_from_dict is None:
        if at_angle_d is None:
            # init to y plane
            body['q'][:, :, 'y'].value = 0

        # running in a straight line
        for link in robot.links:
            for ang in ('phi', 'psi'):
                link['q'][:, :, ang].value = (
                    radians(at_angle_d or 0) if ang == 'psi' else 0
                )
                link['dq'][:, :, ang].value = 0
                link['ddq'][:, :, ang].value = 0

        for fe, cp in robot.indices(one_based=True):
            var = robot.links[0]['q'][fe, cp, 'psi']
            var.setub(radians((at_angle_d or 0) + 10))
            var.setlb(radians((at_angle_d or 0) - 10))

        # init theta
        def rand(mu, sigma, offset=0):
            return radians(random.gauss(mu, sigma)+offset)

        for fe, cp in robot.indices(one_based=True):
            # body
            robot.links[0]['q'][fe, cp, 'theta'].value = rand(0, 15)
            robot.links[1]['q'][fe, cp, 'theta'].value = rand(0, 15, +10)
            # tail
            robot.links[2]['q'][fe, cp, 'theta'].value = rand(0, 15, -10)
            robot.links[3]['q'][fe, cp, 'theta'].value = rand(0, 15, -10)

        for link in robot.links[4:]:  # legs
            for fe, cp in robot.indices(one_based=True):
                link['q'][fe, cp, 'theta'].value = rand(0, 30)

        # body height
        body['q'][:, :, 'z'].value = 0.55

        # the feet:
        prescribe_contact_order(feet, foot_order_vals)  # type: ignore
        for (touchdown, liftoff), foot in zip(foot_order_vals, [foot.name.rstrip('_foot') for foot in feet]):
            lower, upper = foot, 'U' + foot[1:]
            straight_leg(robot[upper]['q'], robot[lower]['q'],
                         [touchdown], state='theta')

            angles = sin_around_touchdown(int((touchdown + liftoff)/2),
                                          len(robot.m.fe))
            for fe, val in zip(robot.m.fe, angles):
                robot[upper]['q'][fe, :, 'theta'].value = val
                robot[lower]['q'][fe, :, 'theta'].value = val + \
                    radians(-15 if upper[1] == 'F' else 15)

        # get timestep bounds ready
        # [long/short] timesteps in the air
        robot.m.hm[:].value = robot.m.hm[1].lb
        for start, stop in foot_order_vals:
            for fe in range(start, stop+1):
                # but [short/long] timesteps while on the ground
                robot.m.hm[fe].value = robot.m.hm[fe].ub
    else:
        #ol.utils.info('Using an init from a dictionary file -- assuming its an Euler init')
        for fed, cpd in robot.indices(one_based=True):
            robot.init_from_dict_one_point(init_from_dict, fed=fed, cpd=cpd, fes=fed-1,
                                           cps=0, skip_if_fixed=True, skip_if_not_None=False, fix=False)
        if not (at_angle_d == 0 or at_angle_d is None):
            raise ValueError(
                f'TODO: rotate init! Got at_angle_d = {at_angle_d}')

    for link in robot.links:
        for fe, cp in robot.indices(one_based=True):
            phi = link['q'][fe, cp, 'phi']
            phi.setub(radians(+15))
            phi.setlb(radians(-15))

            psi = link['q'][fe, cp, 'psi']
            psi.setub(radians(+10 + (at_angle_d or 0)))
            psi.setlb(radians(-10 + (at_angle_d or 0)))

    # bound theta
    for link in robot.links[:2]:  # body
        for fe, cp in robot.indices(one_based=True):
            link['q'][fe, cp, 'theta'].setub(radians(+60))
            link['q'][fe, cp, 'theta'].setlb(radians(-60))

    for link in robot.links[2:]:  # everything else
        for fe, cp in robot.indices(one_based=True):
            link['q'][fe, cp, 'theta'].setub(radians(+90))
            link['q'][fe, cp, 'theta'].setlb(radians(-90))

    # never fallen over
    for fe, cp in robot.indices(one_based=True):
        body['q'][fe, cp, 'z'].setlb(0.3)
        body['q'][fe, cp, 'z'].setub(0.7)

    if at_angle_d is None:
        # roughly bound to y plane
        for fe, cp in robot.indices(one_based=True, skipfirst=False):
            body['q'][fe, cp, 'y'].setub(0.2)
            body['q'][fe, cp, 'y'].setlb(-0.2)

        # average velocity init (overwrite the init!)
        for fe, cp in robot.indices(one_based=True, skipfirst=False):
            body['q'][fe, cp, 'x'].value = avg_vel * \
                total_time * (fe-1 + (cp-1)/ncp)/(nfe-1)
            body['dq'][fe, cp, 'x'].value = avg_vel

        body['q'][nfe, ncp, 'x'].fix(total_time*avg_vel)

        # periodic
        periodic(robot, but_not=('x',))
    else:
        θᵣ = radians(at_angle_d)

        # average velocity init (overwrite the init!)
        for fe, cp in robot.indices(one_based=True, skipfirst=False):
            scale = total_time * (fe-1 + (cp-1)/ncp)/(nfe-1)
            body['q'][fe, cp, 'x'].value = avg_vel * scale * math.cos(θᵣ)
            body['dq'][fe, cp, 'x'].value = avg_vel * math.cos(θᵣ)
            body['q'][fe, cp, 'y'].value = avg_vel * scale * math.sin(θᵣ)
            body['dq'][fe, cp, 'y'].value = avg_vel * math.sin(θᵣ)

        #ol.visual.warn('Should probably also bound x, y!')

        body['q'][nfe, ncp, 'x'].fix(total_time * avg_vel * math.cos(θᵣ))
        body['q'][nfe, ncp, 'y'].fix(total_time * avg_vel * math.sin(θᵣ))

        # periodic
        periodic(robot, but_not=('x', 'y'))

    return add_costs(robot, include_transport_cost=False, include_torque_cost=False)


# def set_quad_motor_limits(robot: System3D):
#     """
#     >>> robot.make_pyomo_model(nfe=10,  collocation='implicit_euler', total_time=0.3)
#     >>> increase_motor_limits(robot, torque_bound=5., no_load_speed=100.)
#     >>> ol.motor.torques(robot)[0]['Tc'].pprint()
#     """
#     assert robot.m is not None, \
#         'robot.make_pyomo_model() must be called before calling this function'

#     motors = {motor.name: motor for motor in ol.motor.torques(robot)}

#     def set_lims(name, torque_bound, no_load_speed):
#         motor = motors[name]
#         for Tc in motor_['Tc'][:, :]:
#             Tc.setub(+torque_bound)
#             Tc.setlb(-torque_bound)
#             if hasattr(motor, 'torque_speed_limit'):
#                 tsp = motor.torque_speed_limit
#                 tsp.torque_bounds = (-torque_bound, torque_bound)
#                 tsp.no_load_speed = no_load_speed

#     for name in ("base_B_base_F_torque", "base_B_UBL_torque", "base_B_UBR_torque"):
#         set_lims(name, 2.5, 75.)
#     for name in ("base_F_UFL_torque", "base_F_UFR_torque"):
#         set_lims(name, 2., 150.)
#     # for name in ("base_B_tail0_torque", "tail0_tail1_torque"):
#     #     set_lims(name, TORQUE, SPEED)
#     for name in ("UFL_LFL_torque", "UFR_LFR_torque"):
#         set_lims(name, 1., 75.)
#     for name in ("UBL_LBL_torque", "UBR_LBR_torque"):
#         set_lims(name, 0.75, 50.)

def theoretical_peak_power(*,
                           mass: float,
                           pct_mass_for_actuation: float = 0.5,
                           watts_per_kg: float = 600.,
                           disp: bool = True):
    """
    >>> theoretical_peak_power(mass=sum(link.mass for link in robot.links))
    """
    peak_power = mass*pct_mass_for_actuation*watts_per_kg

    if disp:
        print(f'Expected total power of a {mass:.2f} kg cheetah with '
              f'{100*pct_mass_for_actuation:.2f}% of mass for actuation '
              f'and {watts_per_kg:.2f} W/kg: mass*actuation*watts_per_kg = '
              f'{int(peak_power)} W')

    return peak_power


def theoretical_peak_angle_velocity(stride_freq_Hz: float = 3.,
                                    total_angle_deg: float = 180.,
                                    disp: bool = True):
    """Cheetah leg moves from 0⁰ -> 90⁰ -> 0⁰ in about 1/3 of a second. Ie, follows the shape:
         position(t) = 90/2 * sin(radians(t/0.3 * 360))
    where t = 0..0.3

    Differentiating with respect to time:
        velocity(t) = 90/2 * cos(radians(t/0.3 * 360)) * 360/0.3
    Giving a max velocity of
        velocity(0) -> 90/2 * 360/0.3 = 

    Example code:

    ```python
        from math import pi as π
        total_angle_deg = 180.
        stride_freq_Hz = 3.

        t = np.linspace(0, 1/stride_freq_Hz)
        pos = lambda t: total_angle_deg/2 * np.sin(t*stride_freq_Hz * 2*π)
        plt.plot(t, 10*pos(t), label='position [deg] scaled by 10')

        vel = lambda t: total_angle_deg/2 * np.cos(t*stride_freq_Hz * 2*π) * stride_freq_Hz * 2*π
        plt.plot(t, vel(t), label='velocity [deg]')

        max_ω_deg = total_angle_deg/2 * stride_freq_Hz * 2*π
        plt.title(f'total angle change = {total_angle_deg} deg\nmax angular velocity = {max_ω_deg:.1f} deg/s = {np.radians(max_ω_deg):.1f} rad/s')
        plt.legend(); plt.show()
    ```
    """
    from math import pi as π, radians
    peak = total_angle_deg/2 * stride_freq_Hz * 2*π

    if disp:
        print(f'Expected peak angular velocity of a leg moving though '
              f'{total_angle_deg} degrees at {stride_freq_Hz} Hz:\n'
              f'total_angle_deg/2 * stride_freq_Hz * 2*π '
              f'= {peak:.2f} deg/s = {radians(peak):.2f} rad/s')

    return peak


# def plot_power_values(robot: System3D, power_arr: List[np.ndarray]):
#     import matplotlib.pyplot as plt

#     peaks = np.sum(
#         np.hstack(power_arr),
#         axis=1
#     )

#     total_time = sum(
#         robot.m.hm[fe].value for fe in robot.m.fe if fe != 1)*robot.m.hm0.value
#     nfe = len(robot.m.fe)

#     plt.plot(np.linspace(0, total_time, num=nfe), peaks)
#     plt.title(
#         f'Total power output of cheetah.\nPeak power: {int(np.max(peaks))} W')
#     plt.ylabel('Total power [W]')
#     plt.xlabel('time [s]')
#     plt.show()
