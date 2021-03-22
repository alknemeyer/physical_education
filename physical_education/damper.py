from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import sympy as sp
from sympy import Matrix as Mat

from .links import Link3D
from .base import SimpleForceBase3D
from .utils import norm, get_name


# @dataclass
# class LinearDamper3D(SimpleForceBase3D):
#     name: str
#     pos1: Mat = field(repr=False)
#     pos2: Mat = field(repr=False)
#     relvelvec: Mat = field(repr=False)
#     damping_coeff: float

#     def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
#         r = (self.pos1 + self.pos2)/2

#         Δ = sp.sqrt(self.relvelvec.T @ self.relvelvec)[0, 0]  # + 1e-3?

#         fmag = self.damping_coeff * Δ
#         f = fmag * self.relvelvec / norm(self.relvelvec, eps=1e-3)
#         Q = Mat((f.T @ r.jacobian(q)).T)
#         return Q


@dataclass
class TorqueDamper3D(SimpleForceBase3D):
    name: str
    relative_velocity: 'sp.Expression' = field(repr=False)
    damping_coeff: float
    damping_coeff_lims: Optional[Tuple[float, float]]

    def __post_init__(self):
        super().__init__()
        if self.damping_coeff_lims is not None:
            self.damping_coeff_sym = self._dummyparam(
                'dampingcoeff',
                self.damping_coeff,
                self.damping_coeff_lims,
                symname='b',
            )

    def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
        c = self.damping_coeff_sym if self.damping_coeff_lims is not None else self.damping_coeff

        Δ = self.relative_velocity
        Ef = 1/2 * c * Δ**2
        Q = - Mat([Ef]).jacobian(q).T
        return Q


# def add_lineardamper(link: 'Link3D', otherlink: 'Link3D',
#                      pos1: Mat, pos2: Mat,
#                      relvelvec: Mat,
#                      damping_coeff: float,
#                      name: Optional[str] = None):

#     name = get_name(name, [link, otherlink], 'lineardamper')

#     spring = LinearDamper3D(name, pos1=pos1, pos2=pos2,
#                             relvelvec=relvelvec,
#                             damping_coeff=damping_coeff)
#     link.nodes[name] = spring  # type: ignore
#     return spring


def add_torquedamper(link: 'Link3D', otherlink: 'Link3D',
                     relative_velocity: 'sp.Expression',
                     damping_coeff: float,
                     damping_coeff_lims: Optional[Tuple[float, float]] = None,
                     name: Optional[str] = None):

    name = get_name(name, [link, otherlink], 'torquedamper')

    damper = TorqueDamper3D(name,
                            relative_velocity=relative_velocity,
                            damping_coeff=damping_coeff,
                            damping_coeff_lims=damping_coeff_lims)
    link.nodes[name] = damper  # type: ignore
    return damper
