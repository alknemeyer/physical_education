from typing import Dict, List, Tuple
import sympy as sp
from .links import Link3D
from . import utils
from pyomo.environ import Constraint, Var

# TODO: add meta information!


def def_leg(body: Link3D, front: bool, right: bool,
            thigh_params: Dict[str, float] = {}, calf_params: Dict[str, float] = {}
            ) -> Tuple[Link3D, Link3D]:
    """Define a leg and attach it to the front/back right/left of `body`.
        Only really makes sense when `body` is aligned along the `x`-axis"""
    # maybe flip x (or y)
    def mfx(x): return x if front else -x
    def mfy(y): return y if right else -y

    start_I = (
        body.Pb_I +
        body.Rb_I @ sp.Matrix([mfx(body.length/2), mfy(body.radius), 0])
    )
    suffix = ('F' if front else 'B') + ('R' if right else 'L')
    thigh = Link3D('U'+suffix, '-z', start_I=start_I, **thigh_params)
    calf = Link3D('L'+suffix, '-z', start_I=thigh.bottom_I, **calf_params)\
        .add_foot(at='bottom', nsides=8)

    body.add_hookes_joint(thigh, about='xy')
    body.add_torque_input()
    body.torque.add_input_torques_at(thigh, about='xy')

    thigh.add_revolute_joint(calf, about='y')
    thigh.add_torque_input()
    thigh.torque.add_input_torques_at(calf, about='y')

    return thigh, calf


def prescribe_straight_leg(upper: Var, lower: Var, fes: List[int], state: str):
    """
    >>> prescribe_straight_leg(robot['thigh']['q'], robot['calf']['q'], [4, 15], 'theta')
    which does the equivalent of,
    >>> m.straight_leg_thigh_calf = Constraint([4, 15],
    ...     rule=lambda m, fe: robot['thigh']['q'][fe,ncp,'theta'] == robot['thigh']['q'][fe,ncp,'theta'])

    or for all legs of a quadruped:

    >>> for (touchdown, liftoff), (upper, lower) in zip(foot_order_vals, (('UFL', 'LFL'), ('UFR', 'LFR'), ('UBL', 'LBL'), ('UBR', 'LBR'))):  # keep in sync with feet!
    ...     prescribe_straight_leg(robot[upper]['q'], robot[lower]['q'], [touchdown, liftoff+1])
    """
    m = upper.model()
    ncp = len(m.cp)

    name = f'straight_leg_{upper.name}_{lower.name}'

    utils.remove_constraint_if_exists(m, name)

    def rule(m, fe): return lower[fe, ncp, state] == upper[fe, ncp, state]
    constr = Constraint(fes, rule=rule)

    setattr(m, name, constr)
