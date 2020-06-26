import sympy as sp, numpy as np
from sympy import Matrix as Mat
from typing import Dict, Any, Optional, List
from . import utils
from .variable_list import VariableList

from pyomo.environ import (
    ConcreteModel, Param, Set, Var, Constraint,
)

def cylinder_in_air(A: float, *, Cd: float = 0.8, rho: float = 10.) -> float:
    utils.warn('These are made-up default values! Fix!'
               ' They\'re just placeholders!')
    return 1/2 * Cd * rho * A

def _norm(vec: Mat, eps: float):
    assert vec.shape == (3, 1)
    assert eps >= 0

    x, y, z = vec
    return sp.sqrt(x**2 + y**2 + z**2 + eps)

def _angle_between(veca: Mat, vecb: Mat, eps: float):
    assert veca.shape == vecb.shape == (3, 1)

    return sp.acos(
        veca.dot(vecb) / _norm(veca, eps) / _norm(vecb, eps)
    )

class Drag3D:
    def __init__(self, name: str, r: Mat, area_norm: Mat, coeff: float,
                 dummy_dr: bool = False, cylinder_top: bool = False):
        """
        - name: unique name to identify the drag force
        - r: location where the drag force acts in 3D
        - area_norm: vector TODO: how to describe this, and cylinder_top?
        - coeff: all the constant stuff in a drag equation: 1/2 * Cd * rho * A

        ```
        >>> Drag3D('tail', r=[x, y, z], Rb_I=euler321(),
                   coeff=cylinder_in_air(A))
        ```
        """
        self.name = name 
        self.r = r
        self.Fmag = sp.Symbol('F_{d/%s}' % name)
        self.coeff = coeff
        self.area_norm = area_norm
        self.dummy_dr = dummy_dr
        self.cylinder_top = cylinder_top

        self._plot_config: Dict[str] = {'plot_forces': True, 'force_scale': 1/10}

    def calc_eom(self, q, dq, ddq, ang_vel, Ek, Ep, M, C, G) -> Mat:
        # angle between area and drag force (a scalar)
        # used to get the effective area
        self.dr = Mat(self.r.jacobian(q) * dq)
        gamma = _angle_between(self.area_norm, self.dr, eps=1e-6)

        # magnitude of the drag force (a scalar)
        dx, dy, dz = self.dr
        if self.cylinder_top:
            self.Fmag_rhs = self.coeff * (1 - sp.sin(gamma))**2 * (dx**2 + dy**2 + dz**2)
        else:
            self.Fmag_rhs = self.coeff * sp.sin(gamma)**2 * (dx**2 + dy**2 + dz**2)

        # self.Fmag_rhs = self.coeff * (dx**2 + dy**2 + dz**2)
        # utils.warn('not taking into account the angle of the drag!')

        # drag force (a vector) opposite to the velocity of the link
        self.f = - self.Fmag * self.dr / _norm(self.dr, eps=1e-6)

        # input force mapping (a vector)
        self.Q = sp.Matrix((self.f.T @ self.r.jacobian(q)).T)

        return self.Q

    def add_vars_to_pyomo_model(self, m: ConcreteModel):
        Fmag = Var(m.fe, m.cp, name='Fmag', bounds=(0, None))

        self.pyomo_params: Dict[str,Param] = {}
        self.pyomo_sets: Dict[str,Set] = {}
        self.pyomo_vars: Dict[str,Var] = {
            'Fmag': Fmag
        }

        for v in self.pyomo_vars.values():
            newname = f'{self.name}_{v}'
            assert not hasattr(m, newname), f'The pyomo model already has a variable with the name "{newname}"'
            setattr(m, newname, v)
    
    def get_pyomo_vars(self, fe: int, cp: int):
        """fe, cp are one-based!"""
        # NB: keep in sync with get_sympy_vars()!!
        v = self.pyomo_vars
        return [v['Fmag'][fe,cp]]#, v['dr'][fe,cp,:]]
    
    def get_sympy_vars(self):
        # NB: keep in sync with get_pyomo_vars()!!
        return [self.Fmag]#, *self.dr]

    def save_data_to_dict(self) -> Dict[str,Any]:
        return {
            'name': self.name,
            'Fmag': utils.get_vals(self.pyomo_vars['Fmag']),
            'coeff': self.coeff,
        }

    def init_from_dict_one_point(self, data: Dict[str,Any], fed: int, cpd: int, fes: Optional[int] = None, cps: Optional[int] = None, **kwargs) -> None:
        if fes is None: fes = fed - 1
        if cps is None: cps = cpd - 1

        assert self.name == data['name']
        for attr in ('coeff',):
            if getattr(self, attr) != data[attr]:
                utils.warn(f'Attribute "{attr}" of link "{self.name}" is not the same as the data: {getattr(self, attr)} != {data[attr]}')

        v = self.pyomo_vars
        utils.maybe_set_var(v['Fmag'][fed,cpd], data['Fmag'][fes,cps], **kwargs)

    def add_equations_to_pyomo_model(self, sp_variables, pyo_variables: VariableList, collocation: str):
        Fmag = self.pyomo_vars['Fmag']
        m = Fmag.model()

        from pyomo.environ import atan
        func_map = {
            'sqrt': lambda x: (x + 1e-6)**(1/2),
            'atan': atan,
        }
        Fmag_rhs_func = utils.lambdify_EOM(self.Fmag_rhs, sp_variables, func_map=func_map)[0]

        ncp = len(m.cp)

        def def_Fmag(m, fe, cp):
            if fe == 1 and cp < ncp:
                return Constraint.Skip
            else:
                return Fmag[fe,cp] == Fmag_rhs_func(*pyo_variables[fe,cp])
        
        setattr(m, self.name + '_Fmag_constr', Constraint(m.fe, m.cp, rule=def_Fmag))

        # for animating
        self.r_func = utils.lambdify_EOM(self.r, sp_variables, func_map=func_map)
        self.f_func = utils.lambdify_EOM(self.f, sp_variables, func_map=func_map)

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
        if self._plot_config['plot_forces'] is False:
            return
        
        self.has_line = False
        self.plot_data = np.empty((len(data), 6))
        scale = self._plot_config['force_scale']

        for fe0, d in enumerate(data):
            x, y, z = [f(*d) for f in self.r_func]
            dx, dy, dz = [f(*d)*scale for f in self.f_func]
            self.plot_data[fe0, :] = (x, y, z, dx, dy, dz)

    def animation_update(self, fig, ax,
                         fe: Optional[int] = None,
                         t: Optional[float] = None,
                         t_arr: Optional[np.ndarray] = None,
                         track: bool = False):
        if self._plot_config['plot_forces'] is False:
            return
        
        if self.has_line:
            self.line.remove()
            self.has_line = False

        if fe is not None:
            x, y, z, dx, dy, dz = self.plot_data[fe-1]
        else:
            assert t is not None and t_arr is not None
            x, y, z, dx, dy, dz = [np.interp(t, t_arr, self.plot_data[:,i]) for i in range(6)]

        self.line = ax.quiver(
            x, y, z,    # <-- starting point of vector
            dx, dy, dz, # <-- directions of vector
            arrow_length_ratio=0.05,
            color = 'red', alpha = .8, lw = 1.5,
        )
        self.has_line = True

    def cleanup_animation(self, fig, ax):
        try:
            del self.line
        except:
            pass

    def plot(self):
        utils.warn('Drag3D.plot() not implemented!', once=True)

    def __repr__(self) -> str:
        return f'Drag3D(name="{self.name}", coeff={self.coeff})'
