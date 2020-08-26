from typing import Any, Dict, Tuple, Callable
from math import pi as π
from ..argh import Mat
from ..links import Link3D, constrain_rel_angle
from ..system import System3D
from ..foot import add_foot
from ..motor import add_torque
from ..drag import add_drag


cheetah_params = {
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
        'body_B': {'mass': 28., 'radius': 0.594/(2*π), 'length': 0.5},
        'body_F': {'mass': 14., 'radius': 0.717/(2*π), 'length': 0.3},
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
        'motor_params': {'torque_bounds': (-3., 3.), 'no_load_speed': 100.},
    },
}


def quadruped3D(params: Dict[str, Any]) -> Tuple[System3D, Callable[[System3D], None]]:
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

    body_B.add_hookes_joint(body_F, about='xy')
    add_torque(body_B, body_F, about='xy', **params['motor_params'])

    tail0 = Link3D('tail0', '-x', start_I=body_B.top_I,
                   **params['tail0'], meta=['tail'])
    tail1 = Link3D('tail1', '-x', start_I=tail0.bottom_I,
                   **params['tail1'], meta=['tail'])

    # TODO: maybe rather friction = 0 ?
    add_foot(tail1, at='bottom', nsides=8, friction_coeff=0.1)

    body_B.add_hookes_joint(tail0, about='xy')
    add_torque(body_B, tail0, about='xy', **params['motor_params'])

    tail0.add_hookes_joint(tail1, about='xy')
    add_torque(tail0, tail1, about='xy', **params['motor_params'])

    add_drag(body_F, at=body_F.bottom_I, name='body_F-drag-head',
             use_dummy_vars=True, cylinder_top=True)
    add_drag(body_F, at=body_F.Pb_I,
             name='body_F-drag-body', use_dummy_vars=True)
    add_drag(body_B, at=body_B.Pb_I, use_dummy_vars=True)
    add_drag(tail0, at=tail0.Pb_I, use_dummy_vars=True)
    add_drag(tail1, at=tail1.Pb_I, use_dummy_vars=True)

    def def_leg(body: Link3D, front: bool, right: bool) -> Tuple[Link3D, Link3D]:
        """Define a leg and attach it to the front/back right/left of `body`.
            Only really makes sense when `body` is aligned along the `x`-axis"""
        # maybe flip x (or y)
        def mfx(x): return x if front else -x
        # typo... these should be swapped, since the model is considered to face
        # along the y axis (so front/back refers to changes in the y value).
        # TODO: fix, but test straight afterwards in case it breaks something!
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
    link_body_B, link_body_F, link_tail0, link_tail1, \
        link_UFL, link_LFL, link_UFR, link_LFR, \
        link_UBL, link_LBL, link_UBR, link_LBR = [
            link['q'] for link in robot.links]

    # spine can't bend too much:
    # it only has pitch and yaw degrees of freedom. No need to constrain roll
    constrain_rel_angle(robot.m, 'spine_pitch',
                        -π/3, link_body_B[:, :, 'theta'], link_body_F[:, :, 'theta'], π/3)
    constrain_rel_angle(robot.m, 'spine_yaw',
                        -π/3, link_body_B[:, :, 'psi'], link_body_F[:, :, 'psi'], π/3)

    # tail can't go too crazy:
    constrain_rel_angle(robot.m, 'tail_body_pitch',
                        -π/3, link_body_B[:, :, 'theta'], link_tail0[:, :, 'theta'], π/3)
    constrain_rel_angle(robot.m, 'tail_body_yaw',
                        -π/3, link_body_B[:, :, 'psi'], link_tail0[:, :, 'psi'], π/3)

    constrain_rel_angle(robot.m, 'tail_tail_pitch',
                        -π/2, link_tail0[:, :, 'theta'], link_tail1[:, :, 'theta'], π/2)
    constrain_rel_angle(robot.m, 'tail_tail_yaw',
                        -π/2, link_tail0[:, :, 'psi'], link_tail1[:, :, 'psi'], π/2)

    # legs: hip abduction and knee
    for body, thigh, calf, name in ((link_body_F, link_UFL, link_LFL, 'FL'),
                                    (link_body_F, link_UFR, link_LFR, 'FR'),
                                    (link_body_B, link_UBL, link_LBL, 'BL'),
                                    (link_body_B, link_UBR, link_LBR, 'BR')):
        constrain_rel_angle(robot.m, name + '_hip_pitch',
                            -π/2, body[:, :, 'theta'], thigh[:, :, 'theta'], π/2)
        constrain_rel_angle(robot.m, name + '_hip_aduct',
                            -π/4, body[:, :, 'psi'], thigh[:, :, 'psi'], π/4)

        lo, up = (-π, 0) if name.startswith('B') else (0, π)
        constrain_rel_angle(robot.m, name + '_knee',
                            lo, thigh[:, :, 'theta'], calf[:, :, 'theta'], up)
