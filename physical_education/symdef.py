import sympy as sp
from sympy import Matrix as Mat
from typing import Iterable, Tuple

USE_LATEX: bool = True


def make_xyz_syms(i: str) -> Tuple[Mat, Mat, Mat]:
    """
    Define x, y, z symbols and time derivatives, with subscript `i`

    ## Usage

    >>> xyz, dxyz, ddxyz = make_xyz_syms('test')
    >>> dxyz
    Matrix([
    [\\dot{x}_{test}],
    [\\dot{y}_{test}],
    [\\dot{z}_{test}]])
    """
    global USE_LATEX
    if USE_LATEX is True:
        q = sp.symbols(r'x_{%s} y_{%s} z_{%s}' % (i, i, i))
        dq = sp.symbols(r'\dot{x}_{%s} \dot{y}_{%s} \dot{z}_{%s}' % (i, i, i))
        ddq = sp.symbols(
            r'\ddot{x}_{%s} \ddot{y}_{%s} \ddot{z}_{%s}' % (i, i, i))
    else:
        q = sp.symbols(f'    x_{i}   y_{i}   z_{i}')
        dq = sp.symbols(f'  dx_{i}  dy_{i}  dz_{i}')
        ddq = sp.symbols(f'ddx_{i} ddy_{i} ddz_{i}')

    return Mat(q), Mat(dq), Mat(ddq)


def make_ang_syms(i: str, nvars: int = 3) -> Tuple[Mat, Mat, Mat]:
    """
    Define `\\phi`, `\\theta`, `\\psi` (ϕ θ ψ) symbols and time derivatives, with subscript `i`

    ## Usage

    >>> ptp, dptp, ddptp = make_ang_syms('test')
    >>> dptp
    Matrix([
    [  \\dot{\\phi}_{test}],
    [\\dot{\\theta}_{test}],
    [  \\dot{\\psi}_{test}]])
    """
    assert 1 <= nvars <= 3

    global USE_LATEX
    if USE_LATEX is True:
        q = sp.symbols(r'\phi_{%s} \theta_{%s} \psi_{%s}' % (i, i, i))
        dq = sp.symbols(
            r'\dot{\phi}_{%s} \dot{\theta}_{%s} \dot{\psi}_{%s}' % (i, i, i))
        ddq = sp.symbols(
            r'\ddot{\phi}_{%s} \ddot{\theta}_{%s} \ddot{\psi}_{%s}' % (i, i, i))
    else:
        q = sp.symbols(f'  ϕ_{i}   θ_{i}   ψ_{i}')
        dq = sp.symbols(f' dϕ_{i}  dθ_{i}  dψ_{i}')
        ddq = sp.symbols(f'ddϕ_{i} ddθ_{i} ddψ_{i}')

    return Mat(q)[:nvars, :nvars], Mat(dq)[:nvars, :nvars], Mat(ddq)[:nvars, :nvars]


def make_ang_sym(rot_axis: str, i: str) -> Tuple[Iterable[sp.Symbol], Iterable[sp.Symbol], Iterable[sp.Symbol]]:
    """
    Define a single angle `\\phi` or `\\theta` or `\\psi` (ϕ θ ψ) symbols and time derivatives, with subscript `i`

    ## Usage

    >>> ptp, dptp, ddptp = make_ang_sym('x', 'test')
    >>> dptp
    \\dot{\\phi}_{test}
    """
    acceptable_args = ('x', 'y', 'z')
    assert rot_axis in acceptable_args,\
        f'Rotation about "xyz". Got {rot_axis}, must be one of {acceptable_args}'

    global USE_LATEX
    var_name = ''
    if rot_axis == 'x':
        var_name = 'phi' if USE_LATEX else 'ϕ'
    elif rot_axis == 'y':
        var_name = 'theta' if USE_LATEX else 'θ'
    elif rot_axis == 'z':
        var_name = 'psi' if USE_LATEX else 'ψ'

    if USE_LATEX is True:
        q = sp.symbols(r'\%s_{%s}' % (var_name, i))
        dq = sp.symbols(r'\dot{\%s}_{%s}' % (var_name, i))
        ddq = sp.symbols(r'\ddot{\%s}_{%s}' % (var_name, i))
    else:
        q = sp.symbols(f'{var_name}_{i}')
        dq = sp.symbols(f'd{var_name}_{i}')
        ddq = sp.symbols(f'dd{var_name}_{i}')

    return q, dq, ddq
