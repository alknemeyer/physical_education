from dataclasses import dataclass, field
import sympy as sp
from sympy import Matrix as Mat
# import numpy as np
from typing import List, Optional, TYPE_CHECKING, Tuple

from .base import SimpleForceBase3D
# from .visual import LineAnimation
from .utils import norm, get_name

if TYPE_CHECKING:
    # from .variable_list import VariableList
    from .links import Link3D


# https://docs.python.org/3/library/dataclasses.html#post-init-processing


# @dataclass
# class LinearSpring3D(SimpleForceBase3D):
#     name: str
#     pos1: Mat = field(repr=False)
#     pos2: Mat = field(repr=False)
#     spring_coeff: float
#     rest_length: float

#     def __post_init__(self) -> None:
#         assert self.pos1.shape == (3, 1)
#         assert self.pos2.shape == (3, 1)

#     def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
#         relposvec = self.pos1 - self.pos2
#         r = (self.pos1 + self.pos2)/2

#         Δ = sp.sqrt(relposvec.T @ relposvec)[0, 0]  # + 1e-3?

#         fmag = self.spring_coeff * (Δ - self.rest_length)
#         f = fmag * relposvec / norm(relposvec, eps=1e-3)
#         Q = Mat((f.T @ r.jacobian(q)).T)
#         return Q

#     def add_equations_to_pyomo_model(self,
#                                      sp_variables: List[sp.Symbol],
#                                      pyo_variables: 'VariableList',
#                                      collocation: str):
#         super().add_equations_to_pyomo_model(
#             sp_variables, pyo_variables, collocation
#         )
#         self.lineanimation = LineAnimation(self.pos1, self.pos2, sp_variables)

#     def animation_setup(self, fig, ax, data: List[List[float]]):
#         self.lineanimation.animation_setup(fig, ax, data)

#     def animation_update(self, fig, ax,
#                          fe: Optional[int] = None,
#                          t: Optional[float] = None,
#                          t_arr: Optional['np.ndarray'] = None,
#                          track: bool = False):
#         self.lineanimation.animation_update(fig, ax, fe, t, t_arr, track)
#         # self.lineanimation.line.set_color('depends on force')

#     def cleanup_animation(self, fig, ax):
#         self.lineanimation.cleanup_animation(fig, ax)


@dataclass
class TorqueSpring3D(SimpleForceBase3D):
    name: str
    relative_angle: 'sp.Expression' = field(repr=False)
    spring_coeff: float
    spring_coeff_lims: Optional[Tuple[float, float]]
    rest_angle: float

    def __post_init__(self):
        super().__init__()
        if self.spring_coeff_lims is not None:
            self.spring_coeff_sym = self._dummyparam(
                'springcoeff',
                self.spring_coeff,
                self.spring_coeff_lims,
                symname='k',
            )

    def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
        k = self.spring_coeff_sym if self.spring_coeff_lims is not None else self.spring_coeff

        Δ = self.relative_angle
        Δ0 = self.rest_angle
        Ef = 1/2 * k * (Δ - Δ0)**2
        Q = - Mat([Ef]).jacobian(q).T
        return Q

        # from IPython.display import display
        # f_dummy = self._dummify(expr=f, forcename=f'Fspring_{self.name}')
        # # all zeros because f_dummy doesn't get jacobian'ed properly :/
        # Q = - Mat([f_dummy]).jacobian(q).T
        # # would have to make Q the dummy vector?
        # display(f_dummy, Q)


# def add_linearspring(link: 'Link3D', otherlink: 'Link3D',
#                      pos1: Mat, pos2: Mat,
#                      spring_coeff: float,
#                      rest_length: float,
#                      name: Optional[str] = None):

#     name = get_name(name, [link, otherlink], 'linearspring')

#     spring = LinearSpring3D(name, pos1=pos1, pos2=pos2,
#                             spring_coeff=spring_coeff,
#                             rest_length=rest_length)

#     link.nodes[name] = spring  # type: ignore
#     return spring


def add_torquespring(link: 'Link3D', otherlink: 'Link3D',
                     relative_angle: 'sp.Expression',
                     spring_coeff: float,
                     rest_angle: float,
                     spring_coeff_lims: Optional[Tuple[float, float]] = None,
                     name: Optional[str] = None):
    name = get_name(name, [link, otherlink], 'torquespring')

    spring = TorqueSpring3D(name, relative_angle=relative_angle,
                            spring_coeff=spring_coeff,
                            rest_angle=rest_angle,
                            spring_coeff_lims=spring_coeff_lims)

    link.nodes[name] = spring  # type: ignore
    return spring
