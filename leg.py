from typing import Dict, List, Tuple
import sympy as sp
from .links import Link3D
from . import utils
from pyomo.environ import Constraint, Var

# TODO: add meta information!

def def_leg(body: Link3D, front: bool, right: bool,
            thigh_params: Dict[str,float] = {}, calf_params: Dict[str,float] = {}) -> Tuple[Link3D,Link3D]:
    """Define a leg and attach it to the front/back right/left of `body`.
        Only really makes sense when `body` is aligned along the `x`-axis"""
    # maybe flip x (or y)
    mfx = lambda x: x if front else -x
    mfy = lambda y: y if right else -y

    start_I = body.Pb_I + body.Rb_I @ sp.Matrix([mfx(body.length/2), mfy(body.radius), 0])
    suffix = ('F' if front else 'B') + ('R' if right else 'L')
    thigh = Link3D('U'+suffix, '-z', start_I=start_I, **thigh_params)
    calf  = Link3D('L'+suffix, '-z', start_I=thigh.bottom_I, **calf_params)\
        .add_foot(at='bottom', nsides=8)
    
    body.add_hookes_joint(thigh, about='xy')\
        .add_input_torques_at(thigh, about='xy')

    thigh.add_revolute_joint(calf, about='y')\
        .add_input_torques_at(calf, about='y')
    
    return thigh, calf

def prescribe_straight_leg(upper: Var, lower: Var, fes: List[int], state: str = 'theta'):
    """
    >>> prescribe_straight_leg(robot['thigh']['q'], robot['calf']['q'], [4, 15])

    or for all legs of a quadruped:

    >>> for (touchdown, liftoff), (upper, lower) in zip(foot_order_vals, (('UFL', 'LFL'), ('UFR', 'LFR'), ('UBL', 'LBL'), ('UBR', 'LBR'))):  # keep in sync with feet!
    ...     prescribe_straight_leg(robot[upper]['q'], robot[lower]['q'], [touchdown, liftoff+1])
    """
    m = upper.model()
    ncp = len(m.cp)

    name = f'straight_leg_{upper.name}_{lower.name}'

    if hasattr(m, name) or hasattr(m, name + '_index'):
        utils.debug(f'Deleting previous straight leg constraint: {name}')
        m.del_component(name)
        m.del_component(name + '_index')
    
    constr = Constraint(fes, rule=lambda m,fe: lower[fe,ncp,'theta'] == upper[fe,ncp,'theta'])
    
    setattr(m, name, constr)
