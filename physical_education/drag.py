"""
The *_in_air functions return the coefficient 'coeff' equal to:

    coeff = 1/2 * Cd * rho * A

It is meant to be used in the drag force equation:

    Fdrag = coeff * cos(angle) * velocity

where `angle` is the angle between the norm of the area of the object and the velocity vector

Source: https://en.wikipedia.org/wiki/Drag_equation
"""
from typing import Union
import sympy as sp
import numpy as np
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from typing_extensions import TypedDict
from pyomo.environ import (
    ConcreteModel, Param, Set, Var, Constraint,
)
from sympy import Matrix as Mat

from .links import Link3D
from .system import System3D
from .utils import norm, get_name
from . import utils
from . import visual

if TYPE_CHECKING:
    from .variable_list import VariableList


PlotConfig = TypedDict('PlotConfig', plot_forces=bool, force_scale=float)


# From https://en.wikipedia.org/wiki/Density#Air at T = 25 C
DENSITY_OF_AIR = 1.184


def cylinder_top_in_air(A: float, *, Cd: float = 0.82, rho: float = DENSITY_OF_AIR) -> float:
    """
    Default `Cd` of 0.82 is for a "long cylinder" from the table in
    https://en.wikipedia.org/wiki/File:14ilf1l.svg#/media/File:14ilf1l.svg

    via https://en.wikipedia.org/wiki/Drag_coefficient
    """
    return 1/2 * Cd * rho * A


def cylinder_in_air(A: float, *, Cd: float = 1.1, rho: float = DENSITY_OF_AIR) -> float:
    """
    Default `Cd` of 1.1 is for a human/cables from
    https://en.wikipedia.org/wiki/Drag_coefficient#General
    """
    return 1/2 * Cd * rho * A


def _angle_between(veca: Mat, vecb: Mat, eps: float):
    assert veca.shape == vecb.shape == (3, 1)

    return sp.acos(
        veca.dot(vecb) / norm(veca, eps) / norm(vecb, eps)
    )


class Drag3D:
    def __init__(self, name: str, r: Mat, area_norm: Mat, coeff: float, *,
                 use_dummy_vars: bool, cylinder_top: bool = False):
        """
        - name: unique name to identify the drag force
        - r: location where the drag force acts in 3D
        - area_norm: vector TODO: how to describe this, and cylinder_top?
        - coeff: all the constant stuff in a drag equation: 1/2 * Cd * rho * A

        ```
        >>> Drag3D('tail', r=[x, y, z], Rb_I=euler321(), coeff=cylinder_in_air(A))
        ```
        """
        self.name = name
        self.r = r
        self.Fmag = sp.Symbol('F_{d/%s}' % name)
        self.coeff = coeff
        self.area_norm = area_norm
        self.use_dummy_vars = use_dummy_vars
        self.cylinder_top = cylinder_top
        self.deactivated = False

        self._plot_config: PlotConfig = {
            'plot_forces': True,
            'force_scale': 1,
        }

    def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
        # angle between area and drag force (a scalar)
        # used to get the effective area
        dr_eqn = Mat(self.r.jacobian(q) * dq)

        if self.use_dummy_vars:
            from .symdef import make_xyz_syms

            self.area_norm_eqn = self.area_norm
            self.area_norm = make_xyz_syms(self.name + '-area-norm')[1]

            self.dr_eqn = dr_eqn
            self.dr = make_xyz_syms(self.name + '-dr')[1]
        else:
            self.dr = dr_eqn

        gamma = _angle_between(self.area_norm, self.dr, eps=1e-6)

        # magnitude of the drag force (a scalar) which is proportional to the velocity
        dx, dy, dz = self.dr
        if self.cylinder_top:
            self.Fmag_rhs = self.coeff * \
                (1 - sp.sin(gamma))**2 * (dx**2 + dy**2 + dz**2)
        else:
            self.Fmag_rhs = self.coeff * \
                sp.sin(gamma)**2 * (dx**2 + dy**2 + dz**2)

        # drag force (a vector) opposite to the velocity of the link
        self.f = - self.Fmag * self.dr / norm(self.dr, eps=1e-6)

        # then, project the force onto the plane defined by the norm of the area
        # n = self.area_norm
        # # this assumes area_norm is a unit vector
        # self.f = self.f - n.dot(self.f) * n
        # # self.f = self.f - n.dot(self.f)/n.dot(n) * n  # this normalizes area_norm

        # input force mapping (a vector)
        self.Q = Mat((self.f.T @ self.r.jacobian(q)).T)

        return self.Q

    def add_vars_to_pyomo_model(self, m: ConcreteModel):
        Fmag = Var(m.fe, m.cp, name='Fmag', bounds=(0, None))

        xyz_set = Set(initialize=['x', 'y', 'z'] if self.use_dummy_vars else [],
                      name='xyz_set', ordered=True)
        dr = Var(m.fe, m.cp, xyz_set, name='dr')
        area_norm = Var(m.fe, m.cp, xyz_set, name='area_norm')

        self.pyomo_params: Dict[str, Param] = {}
        self.pyomo_sets: Dict[str, Set] = {'xyz_set': xyz_set}
        self.pyomo_vars: Dict[str, Var] = {
            'Fmag': Fmag,
            'dr': dr,
            'area_norm': area_norm,
        }

        utils.add_to_pyomo_model(m, self.name, [
            self.pyomo_params.values(),
            self.pyomo_sets.values(),
            self.pyomo_vars.values(),
        ])

    def get_pyomo_vars(self, fe: int, cp: int):
        """fe, cp are one-based!"""
        # NB: keep in sync with get_sympy_vars()!!
        v = self.pyomo_vars
        return [v['Fmag'][fe, cp], *v['dr'][fe, cp, :], *v['area_norm'][fe, cp, :]]

    def get_sympy_vars(self):
        # NB: keep in sync with get_pyomo_vars()!!
        return [self.Fmag] + ([*self.dr, *self.area_norm] if self.use_dummy_vars else [])

    def save_data_to_dict(self) -> Dict[str, Any]:
        m = self.pyomo_vars['Fmag'].model()
        xyz_set = self.pyomo_sets['xyz_set']
        return {
            'name': self.name,
            'Fmag': utils.get_vals(self.pyomo_vars['Fmag'], (m.cp,)),
            'dr': utils.get_vals(self.pyomo_vars['dr'], (xyz_set,)),
            'area_norm': utils.get_vals(self.pyomo_vars['area_norm'], (xyz_set,)),
            'coeff': self.coeff,
        }

    def init_from_dict_one_point(self, data: Dict[str, Any],
                                 fed: int, cpd: int,
                                 fes: Optional[int] = None, cps: Optional[int] = None,
                                 **kwargs
                                 ) -> None:
        if fes is None:
            fes = fed - 1
        if cps is None:
            cps = cpd - 1

        assert self.name == data['name']
        for attr in ('coeff',):
            if getattr(self, attr) != data[attr]:
                visual.warn(
                    f'Attribute "{attr}" of link "{self.name}" is not the same as the data: {getattr(self, attr)} != {data[attr]}')

        v = self.pyomo_vars
        utils.maybe_set_var(v['Fmag'][fed, cpd],
                            data['Fmag'][fes, cps], **kwargs)

        for axi, ax in enumerate(self.pyomo_sets['xyz_set']):
            utils.maybe_set_var(v['dr'][fed, cpd, ax],
                                data['dr'][fes, cps, axi], **kwargs)
            utils.maybe_set_var(v['area_norm'][fed, cpd, ax],
                                data['area_norm'][fes, cps, axi], **kwargs)

    def add_equations_to_pyomo_model(self,
                                     sp_variables: List[sp.Symbol],
                                     pyo_variables: 'VariableList',
                                     collocation: str):
        Fmag = self.pyomo_vars['Fmag']
        m = Fmag.model()

        # if the force is deactivated, set the force magnitudes
        # to zero, don't add other constraints, and early return
        if self.deactivated:
            for fe in m.fe:
                for cp in m.cp:
                    Fmag[fe, cp].fix(0)
            return

        from pyomo.environ import atan
        func_map = {
            'sqrt': lambda x: (x + 1e-6)**(1/2),
            'atan': atan,
        }
        Fmag_rhs_func = utils.lambdify_EOM(
            self.Fmag_rhs, sp_variables, func_map=func_map)[0]

        ncp = len(m.cp)

        def def_Fmag(m, fe, cp):
            if fe == 1 and cp < ncp:
                return Constraint.Skip
            else:
                return Fmag[fe, cp] == Fmag_rhs_func(pyo_variables[fe, cp])

        setattr(m, self.name + '_Fmag_constr',
                Constraint(m.fe, m.cp, rule=def_Fmag))

        if self.use_dummy_vars:
            xyz_set = self.pyomo_sets['xyz_set']
            dr = self.pyomo_vars['dr']
            dr_rhs_funcs = utils.lambdify_EOM(
                self.dr_eqn, sp_variables, func_map=func_map
            )

            def def_dr_dummy(m, fe, cp, ax):
                if fe == 1 and cp < ncp:
                    return Constraint.Skip
                else:
                    return dr[fe, cp, ax] == dr_rhs_funcs['xyz'.index(ax)](pyo_variables[fe, cp])

            setattr(m, self.name + '_dr_dummy',
                    Constraint(m.fe, m.cp, xyz_set, rule=def_dr_dummy))

            area_norm = self.pyomo_vars['area_norm']
            area_norm_rhs_funcs = utils.lambdify_EOM(
                self.area_norm_eqn, sp_variables, func_map=func_map
            )

            def def_area_norm_dummy(m, fe, cp, ax):
                if fe == 1 and cp < ncp:
                    return Constraint.Skip
                else:
                    return area_norm[fe, cp, ax] == area_norm_rhs_funcs['xyz'.index(ax)](pyo_variables[fe, cp])

            setattr(m, self.name + '_area_norm_dummy',
                    Constraint(m.fe, m.cp, xyz_set, rule=def_area_norm_dummy))

        # for animating
        self.r_func = utils.lambdify_EOM(
            self.r, sp_variables, func_map=func_map)
        self.f_func = utils.lambdify_EOM(
            self.f, sp_variables, func_map=func_map)

    def __getitem__(self, varname: str) -> Var:
        return self.pyomo_vars[varname]

    def plot_config(self, *, plot_forces: Optional[bool] = None,
                    force_scale: Optional[float] = None) -> 'Drag3D':
        if plot_forces is not None:
            self._plot_config['plot_forces'] = plot_forces

        if force_scale is not None:
            self._plot_config['force_scale'] = force_scale

        return self

    def animation_setup(self, fig, ax, data: List[List[float]]):
        if self.deactivated:
            return

        if self._plot_config['plot_forces'] is False:
            return

        self.has_line = False
        self.plot_data = np.empty((len(data), 6))
        scale = self._plot_config['force_scale']

        for fe0, d in enumerate(data):
            x, y, z = [f(d) for f in self.r_func]
            dx, dy, dz = [f(d)*scale for f in self.f_func]
            self.plot_data[fe0, :] = (x, y, z, dx, dy, dz)

    def animation_update(self, fig, ax,
                         fe: Optional[int] = None,
                         t: Optional[float] = None,
                         t_arr: Optional[np.ndarray] = None,
                         track: bool = False):
        if self.deactivated:
            return

        if self._plot_config['plot_forces'] is False:
            return

        if self.has_line:
            self.line.remove()
            self.has_line = False

        if fe is not None:
            x, y, z, dx, dy, dz = self.plot_data[fe-1]
        else:
            assert t is not None and t_arr is not None
            x, y, z, dx, dy, dz = [
                np.interp(t, t_arr, self.plot_data[:, i]) for i in range(6)
            ]

        self.line = ax.quiver(
            x, y, z,    # <-- starting point of vector
            dx, dy, dz,  # <-- directions of vector
            arrow_length_ratio=0.05,
            color='red', alpha=.8, lw=1.5,
        )
        self.has_line = True

    def cleanup_animation(self, fig, ax):
        if self.deactivated:
            return

        try:
            del self.line
        except:
            pass

    def plot(self):
        visual.warn('Drag3D.plot() not implemented!', once=True)

    def __repr__(self) -> str:
        return f'Drag3D(name="{self.name}", coeff={self.coeff}, active={not self.deactivated})'


def add_drag(link, at: Mat, name: Optional[str] = None, **kwargs):
    name = get_name(name, [link], 'drag')

    v = sp.zeros(3, 1)
    v['xyz'.index(link.aligned_along[1])] = 1

    if kwargs.get('cylinder_top', False):
        from math import pi
        coeff = cylinder_top_in_air(A=pi * link.radius**2)
    else:
        coeff = cylinder_in_air(A=link.length * (2 * link.radius))

    drag = Drag3D(name,
                  r=at,
                  area_norm=link.Rb_I * v,
                  coeff=coeff,
                  **kwargs)
    link.nodes[name] = drag
    return drag


def drag_forces(robot_or_link: Union[System3D, Link3D]) -> List[Drag3D]:
    from typing import cast
    if isinstance(robot_or_link, System3D):
        robot = robot_or_link
        return [cast(Drag3D, node)
                for link in robot.links
                for node in link.nodes.values()
                if isinstance(node, Drag3D)]
    else:
        link = robot_or_link
        return [cast(Drag3D, node)
                for node in link.nodes.values()
                if isinstance(node, Drag3D)]
