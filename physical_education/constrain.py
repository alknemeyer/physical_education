from typing import List, Tuple, TYPE_CHECKING
from . import utils
from pyomo.environ import Constraint, Var

if TYPE_CHECKING:
    from .system import System3D


def periodic(robot: 'System3D', but_not: Tuple[str, ...], but_not_vel: Tuple[str, ...] = tuple()):
    """
    Make all position and velocity states in `robot` periodic, except for the
    positions (q) defined in `but_not` and the velocities (dq) in `but_not_vel`

    ## Usage

    >>> periodic(robot, but_not=('x',))
    >>> periodic(robot, but_not=('x', 'y'), but_not_vel=('x', 'y'))
    """
    pyo_model = utils.get_pyomo_model_or_error(robot)

    nfe, ncp = len(pyo_model.fe), len(pyo_model.cp)

    utils.remove_constraint_if_exists(pyo_model, 'periodic_q')
    utils.remove_constraint_if_exists(pyo_model, 'periodic_dq')

    # periodic positions
    qs = [
        (link['q'][1, ncp, q], link['q'][nfe, ncp, q])
        for link in robot.links
        for q in link.pyomo_sets['q_set']
        if q not in but_not
    ]
    pyo_model.add_component(
        'periodic_q', Constraint(
            range(len(qs)), rule=lambda m, i: qs[i][0] == qs[i][1])
    )

    # periodic velocities
    dqs = [
        (link['dq'][1, ncp, q], link['dq'][nfe, ncp, q])
        for link in robot.links
        for q in link.pyomo_sets['q_set']
        if q not in but_not_vel
    ]
    pyo_model.add_component(
        'periodic_dq', Constraint(
            range(len(dqs)), rule=lambda m, i: dqs[i][0] == dqs[i][1])
    )


def straight_leg(upper: Var, lower: Var, fes: List[int], state: str):
    """
    Constrain `state` in `upper` to be the same as `lower` at finite elements `fes`

    ## Usage

    >>> straight_leg(robot['thigh']['q'], robot['calf']['q'], [4, 15], 'theta')

    which does the equivalent of,

    >>> thigh = robot['thigh']['q']
    >>> calf = robot['calf']['q']
    >>> m.straight_leg_thigh_calf = Constraint([4, 15],
    ...     rule=lambda m, fe: thigh[fe,ncp,'theta'] == calf[fe,ncp,'theta'])

    or for all legs of a quadruped:

    >>> feet = (('UFL', 'LFL'), ('UFR', 'LFR'), ('UBL', 'LBL'), ('UBR', 'LBR'))
    >>> for (touchdown, liftoff), (upper, lower) in zip(foot_order_vals, feet):
    ...     straight_leg(robot[upper]['q'], robot[lower]['q'], [touchdown, liftoff])
    """
    m = upper.model()
    ncp = len(m.cp)

    name = f'straight_leg_{upper.name}_{lower.name}'

    utils.remove_constraint_if_exists(m, name)

    setattr(m, name, Constraint(fes,
                                rule=lambda m, fe: lower[fe, ncp, state] == upper[fe, ncp, state]))
