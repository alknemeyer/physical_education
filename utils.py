import sympy as sp, numpy as np
from sympy import Matrix as Mat
sp.init_printing()
from typing import Any, Dict, Iterable, Callable, Optional, Tuple, List, Union, Set


# printing ###################################################################
# https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLINK = '\33[5m'

USE_COLOURS: bool = True
DEBUG_MODE: bool = True
ONCE_SET: Set[str] = set()

# fix for windows (based on stackoverflow link above). Haven't tested it myself
import os
if os.name == 'nt':
    os.system('color')

from contextlib import contextmanager
@contextmanager
def no_colour_output():
    try:
        global USE_COLOURS
        USE_COLOURS = False
        yield
    finally:
        USE_COLOURS = True

def _printutil(prefix: str, s: str, colour: str, once: bool):
    if once is True:
        if s in ONCE_SET:
            return
        else:
            ONCE_SET.add(s)
    
    if USE_COLOURS is True:
        print(bcolors.BOLD + colour + str(prefix) + bcolors.ENDC, colour + str(s) + bcolors.ENDC)
    else:
        print(str(prefix), str(s))

def info(s: str, prefix    = '[INFO 💡]', once: bool = False):
    _printutil(prefix, s, bcolors.OKBLUE, once)

def warn(s: str, prefix    = '[WARNING]', once: bool = False):
    _printutil(prefix, s, bcolors.FAIL, once)

def error(s: str, prefix   = '[ERROR💩]', once: bool = False):
    _printutil(prefix, s, bcolors.FAIL + bcolors.BLINK, once)

def debug(s: str, prefix   = '[DEBUG🐞]', once: bool = False):
    if DEBUG_MODE is True:
        _printutil(prefix, s, bcolors.WARNING, once)

def success(s: str, prefix = '[SUCCESS]', once: bool = False):
    _printutil(prefix, s, bcolors.OKGREEN, once)

def test_coloring():
    info('using xyz parameter...')
    warn('this might not work :/')
    error('divided by zero :o')
    debug('the value is 2.5')
    success('it worked!')
    
    warn('this should print once!', once=True)
    warn('this should print once!', once=True)
    warn('this should print once!', once=True)
    warn('this should print once!', once=True)


# derivatives ################################################################
def deriv(expr, q: Mat, dq: Mat):
    """Take the time derivative of an expression `expr` with respect to time,
    handling the chain rule semi-correctly"""
    return (sp.diff(expr, q).T * dq)[0]

def full_deriv(var, q: Mat, dq: Mat, ddq: Mat):
    return deriv(var, q, dq) + deriv(var, dq, ddq)

# rotations ##################################################################
SymOrFloat = Union[sp.Symbol,float]

def rot(θ: SymOrFloat) -> Mat:
    return sp.Matrix([
        [ sp.cos(θ), sp.sin(θ)],
        [-sp.sin(θ), sp.cos(θ)],
    ])

def rot_x(θ: SymOrFloat) -> Mat:
    return sp.Matrix([
        [ 1,        0,           0],
        [ 0, sp.cos(θ),  sp.sin(θ)],
        [ 0,-sp.sin(θ),  sp.cos(θ)],
    ])

def rot_y(θ: SymOrFloat) -> Mat:
    return sp.Matrix([
        [ sp.cos(θ), 0,-sp.sin(θ)],
        [         0, 1,         0],
        [ sp.sin(θ), 0, sp.cos(θ)],
    ])

def rot_z(θ: SymOrFloat) -> Mat:
    return sp.Matrix([
        [ sp.cos(θ), sp.sin(θ), 0],
        [-sp.sin(θ), sp.cos(θ), 0],
        [        0,         0, 1],
    ])

def euler_321(phi: SymOrFloat, theta: SymOrFloat, psi: SymOrFloat) -> Mat:
    return rot_x(phi) @ rot_y(theta) @ rot_z(psi)

# utils to find equations of motion ##########################################
def skew_symmetric(Rx_I: Mat, q: Mat, dq: Mat) -> Mat:
    """Rx_I is the 3x3 rotation from body frame `x` to the inertial `I`
    `q` is a vector of symbolic variables
    `dq` is a vector of the derivates of `q`
    
    The return matrix can be simplified fairly quickly with `trigsimp` if
    it's made up of ~3 or fewer rotations
    """
    dRx_I = sp.zeros(3, 3)
    
    for i in range(3):
        dRx_I[:, i] = Rx_I[:, i].jacobian(q) * dq
    
    omega_Rx = Rx_I.T @ dRx_I
    return Mat([
        omega_Rx[2,1],
        omega_Rx[0,2],
        omega_Rx[1,0]
    ])

def manipulator_equation(Ek: Mat, Ep: Mat, q: Mat, dq: Mat) -> Tuple[Mat, Mat, Mat]:
    """Ek and Ep are the kinetic and potential energy of the system.
    They must be passed in as sp.Matrix types (or at least have a
    `.jacobian` method defined on them)"""
    M = sp.hessian(Ek, dq)
    N = M.shape[0]

    dM = sp.zeros(N, N)
    for i in range(N):
        for j in range(N):
            dM[i,j] = Mat([M[i,j]]).jacobian(q) @ dq

    C = dM @ dq - Ek.jacobian(q).T

    G = Ep.jacobian(q).T
    
    return M, C, G

def calc_velocities_and_energies(
        positions: Iterable[Mat], rotations: Iterable[Mat],
        masses: Iterable[float], inertias: Iterable[Mat],
        q: Mat, dq: Mat, g: float = 9.81) -> Tuple[Mat,Mat,List[Mat],List[Mat]]:
    """
        Calculate and return  the kinetic and potential energies of
        a system, given lists of:
        - positions of each body (in inertial)
        - rotations for each body (from body to inertial)
        - mass of each body
        - inertia of each body (given as a 3x3 matrix)
        - q (each state, such as q = [x, y, theta])
        - dq (derivative of each state, such as dq = [dx, dy, dtheta])
    """
    from functools import reduce
    
    dPs = [Mat(Px_I.jacobian(q) * dq) for Px_I in positions]
    ang_vels = [
        skew_symmetric(Rx_I, q, dq) for Rx_I in rotations
    ]
    # this should be sum(), but it inits the sum with the int 0, which can't
    # be added to matrices
    Ek = reduce(lambda a,b: a + b, [
        dPx_I.T * mx * dPx_I / 2 + dωx_I.T * Ix * dωx_I / 2
           for (dPx_I, mx, dωx_I, Ix) in zip(dPs, masses, ang_vels, inertias)
    ])
    Ep = Mat([sum(m * Mat([0, 0, g]).dot(p) for (m,p) in zip(masses, positions))])
    
    return Ek, Ep, dPs, ang_vels

def lambdify_EOM(EOM: Union[sp.Matrix,list], vars_in_EOM: List[sp.Symbol], *,
                 display_vars: bool = False, test_func: bool = True) -> List[Callable[...,float]]:
    """ Returns a list of functions which, when called with arguments which match
    `vars_in_EOM`, will evaluate the equations of motion specified in `EOM`.
    `display_vars` specifies whether to print out `vars_in_EOM` as a sanity check.
    `test_func` specifies whether to test that the function returns a float when
    called with float arguments

        >>> lambdify_EOM(EOM, vars_in_EOM)
        [<function _lambdifygenerated(_Dummy_833, _Dummy_834, ...
         <function _lambdifygenerated(_Dummy_851, _Dummy_852, ... 
         <function _lambdifygenerated(_Dummy_923, _Dummy_924, ...]

        >>> lambdify_EOM(EOM, vars_in_EOM[:-3])  # not all vars in EOM
        AssertionError: The function didn't return a float - it's likely ...
        """
    import pyomo.environ
    import random

    if display_vars is True:
        try:
            from IPython.core.display import display
            display(vars_in_EOM)
        except:
            print(vars_in_EOM)
    
    func_map = [{'sin': pyomo.environ.sin, 'cos': pyomo.environ.cos}]
    
    if isinstance(EOM, list):
        eom = sp.Matrix(EOM)
    elif EOM.is_Matrix is False:  # EOM is a sympy object, but not a Matrix
        eom = sp.Matrix([EOM])
    else:
        eom = EOM
    
    if not set(eom.free_symbols).issubset(set(vars_in_EOM)):
        raise ValueError('Some symbols in the eom aren\'t in `vars_in_EOM:'
                         + str(set(eom.free_symbols).difference(set(vars_in_EOM))))
    
    funcs = [sp.lambdify(vars_in_EOM, eqn, modules=func_map) for eqn in eom]
    
    # replace with set(EOM.free_symbols).difference(set(vars_in_EOM))?
    if test_func is True:
        vals = [random.random() for _ in range(len(vars_in_EOM))]
        for func in funcs:
            ret = func(*vals)
            assert type(ret) == float, "The function didn't return a float - it's likely "\
                                        "because there are symbolic variables in the EOM "\
                                        "which weren't specified in `vars_in_EOM`. Got: " + str(ret)
    
    return funcs

# simplification #################################################################
def parsimp_worker(_arg: Tuple, allow_recur: bool = True):
    expr, simp_func = _arg

    if expr.is_number:
        return expr

    args = []

    if hasattr(expr, 'args'):
        for arg in expr.args:
            if allow_recur is True and sp.count_ops(arg) > 200:
                arg = parsimp_worker((arg, simp_func), allow_recur=False)
            
            args.append(simp_func(arg))
    else:
        raise ValueError(f'Something is wrong here - got an object of type {type(expr)}'
                         f' with value: {expr}')

    try:
        return expr.func(*args)
    except Exception as e:
        raise ValueError(f'Got exception: {e} for value {expr}')

def parsimp(mat: Mat, nprocs: int, f = sp.trigsimp) -> Mat:
    import multiprocessing
    with multiprocessing.Pool(processes=nprocs) as p:
        return sp.Matrix(
           p.map(parsimp_worker, [(v,f) for v in mat])
        ).reshape(*mat.shape)

#     import sys
#     outvals = []
#         for i, val in enumerate(p.imap_unordered(parsimp_worker,
#                                 [(v,f,i) for (i,v) in enumerate(mat)]), 1):
#             outvals.append(val)
#             if disp_progress is True:
#                 sys.stdout.write('\rSimplifying.... {0:%} done'.format(i/len(vec)))
#     return sp.Matrix(outvals).reshape(vec.shape)

# interpolation ##############################################################
from pyomo.environ import ConcreteModel, Var

def def_var(m: ConcreteModel, name: str, indexes: tuple, func: Callable, bounds: tuple = (None, None)):
    """Define a variable, equal to the result of a function. Eg:
    
    >>> def_var(m, 'foot_height', (m.fe,), bounds=(0, None),
                func=lambda m,fe: sum(m.foot_height[fe,:]))
    
    # which is equivalent to:
    >>> m.foot_height = Var(m.fe, bounds=(0, None))
    >>> @m.Constraint(m.fe)
        def foot_height_constr(m, fe):
            return m.foot_height_dummy[fe] == sum(m.foot_height[fe,:]))
    """
    from pyomo.environ import Var, Constraint
    # define the variable. eg:
    # >>> m.my_var_name = Var(m.fe, m.otherindex, bounds=(0, None))
    setattr(m, name, Var(*indexes, bounds=bounds))
    
    # define a function to set the variable. eg:
    # >>> var[m.fe, m.otherindex] == func(m, m.fe, m.otherindex)
    var = getattr(m, name)
    setterfunc = lambda *args: var.__getitem__(args[1:]) == func(*args)
    
    # set the variable equal to the expression. eg:
    # >>> m.my_var_name_constr = Constraint(m.fe, m.otherindex, rule=setterfunc)
    setattr(m, name + '_constr', Constraint(*indexes, rule=setterfunc))

def fix_within(lo: float, var: Var, up: float):
    """
    >>> fix_within(3., link['q'][:,:,'z'], 6.)
    """
    var.setlb(lo)
    var.setub(up)

def add_guiding_trajectory(var: Var, trajectory: Iterable[float], *, tol: float):
    """
    Guide a variable by fixing its values to be within `+-tol` of `trajectory`.

    >>> trajectory = 1 + 0.2 * np.sin(np.linspace(0, 2*np.pi, num=len(robot.m.fe)))
    >>> add_guiding_trajectory(robot['body']['q'][:,1,'z'], trajectory, tol=0.3)
    """
    warn('utils.add_guiding_trajectory has not been tested!', once=True)
    for _var, val in zip(var, trajectory):
        _var.setub(val + tol)
        _var.setlb(val - tol)

def copy_state_init(src: Var, dst: Var):
    """
    >>> copy_state_init(robot['UFL']['q'], robot['UFR']['q'])
    """
    for idx, var in src.iteritems():
        dst[idx].value = var.value

def remove_constraint_if_exists(m: ConcreteModel, constraint: str):
    if hasattr(m, constraint):
        debug(f'Deleting existing constraint: {constraint}')
        m.del_component(constraint)
    
    constraint_index = constraint + '_index'
    if hasattr(m, constraint_index):
        debug(f'Deleting existing constraint index: {constraint_index}')
        m.del_component(constraint_index)

def maybe_set_var(var: Var, value: float, skip_if_fixed: bool, skip_if_not_None: bool, fix: bool) -> None:
    """
    >>> set_var(link['q'][fe,cp,'x'], 1.0, skip_if_fixed=True, skip_if_not_None=True, fix=False)
    """
    if var.fixed is True and skip_if_fixed is True:
        return
    elif var.value is not None and skip_if_not_None is True:
        return
    else:
        var.value = value
        if fix is True:
            var.fixed = True

def constrain_total_time(m: ConcreteModel, total_time: float):
    """
    >>> constrain_total_time(robot.m, total_time = (nfe-1)*robot.m.hm0.value)
    """
    from pyomo.environ import Constraint
    remove_constraint_if_exists(m, 'total_time_constr')
    m.total_time_constr = Constraint(expr = sum(m.hm[fe] for fe in m.fe if fe != 1)*m.hm0 == total_time)

# other utils for pyomo ######################################################
def default_solver(*,
                   max_mins: int,
                   ipopt_path: Optional[str] = None,
                   solver: str = 'ma86',
                   max_iter: int = 50_000,
                   OF_print_frequency_time: int = 10,
                   output_file: str = '.delete-me.txt',  # prefix with '.' so it doesn't get synced
                   warm_start_init_point: bool = True,
                   **kwargs):
    """
    Tested linear solvers include 'mumps', 'ma77', 'ma97', 'ma86' and 'pardiso'. ma86 seems
    to perform the best for the types of systems modelled with this library
    
    Some solver options include,

    OF_print_frequency_iter: pos int, default 1
        Summarizing iteration output is printed every print_frequency_iter iterations,
        if at least print_frequency_time seconds have passed since last output

    OF_print_frequency_time: pos int, default 0
        Summarizing iteration output is printed if at least print_frequency_time seconds
        have passed since last output and the iteration number is a multiple of
        print_frequency_iter
    
    OF_warm_start_init_point: default 'no'
        Indicates whether this optimization should use a warm start initialization, where
        values of primal and dual variables are given (e.g., from a previous optimization
        of a related problem.)
    
    OF_hessian_approximation: default 'exact'
        Set to 'limited-memory' for L-GBFS. Default is 'exact'
    
    OF_accept_every_trial_step: default 'no'
        Setting this option to "yes" essentially disables the line search and makes the
        algorithm take aggressive steps, without global convergence guarantees
    
    print_info_string: default 'no'
        Enables printing of additional info string at end of iteration output. This string
        contains some insider information about the current iteration. For details, see
        diagnostic tags at bottom of https://coin-or.github.io/Ipopt/OUTPUT.html

    See https://coin-or.github.io/Ipopt/OPTIONS.html for more options. Note that options
    written to the options file should be prefixed with "OF_"

    Another useful page is https://coin-or.github.io/Ipopt/OUTPUT.html, which helps
    understand IPOPT's output
    """
    import os, platform
    from datetime import datetime
    from pyomo.opt import SolverFactory
    
    if ipopt_path is None:
        host = platform.uname()[1]
        if host == 'scruffy':
            ipopt_path = '~/alknemeyer-msc/IPOPT/dist/bin/ipopt'
        elif host == 'minibeast':
            ipopt_path = '~/CoinIpoptBackup/build/bin/ipopt'
            # for pardiso, use '~/Ipopt3.13-Coinbrew/dist/bin/ipopt'

    opt = SolverFactory('ipopt', executable=ipopt_path)
    opt.options['print_level'] = 5
    opt.options['max_cpu_time'] = max_mins * 60
    opt.options['max_iter'] = max_iter
    opt.options['Tol'] = 1e-6
    opt.options['OF_print_timing_statistics'] = 'yes'
    opt.options['halt_on_ampl_error'] = 'yes'
    opt.options['OF_print_frequency_time'] = OF_print_frequency_time
    opt.options['OF_acceptable_tol'] = 1e-3  # default: 1e-6
    opt.options['OF_warm_start_init_point'] = 'yes' if warm_start_init_point else 'no'
    opt.options['output_file'] = os.getcwd() + '/' + output_file

    opt.options['linear_solver'] = solver
    if solver == 'ma86':
        opt.options['OF_ma86_scaling'] = 'none'
    else:
        warn(f'Got solver {solver} but don\'t have any specific flags for it.')
    
    for key, val in kwargs.items():
        opt.options[key] = val

    info(f'Optimization start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    return opt

def get_vals(var: Var, idxs: Optional[Tuple] = None) -> np.ndarray:
    m = var.model()
    assert isinstance(idxs, tuple) or idxs is None

    nfe = len(m.fe)
    ncp = len(m.cp)
    arr = np.array([var[idx].value for idx in var]).astype(float)
    
    if idxs is None:  # assume indexed with only fe
        assert var.dim() == 1, "This variable doesn't seem to be indexed with fe exclusively"
        return arr
    else:
        idxs = tuple(len(i) for i in idxs)
        
        # if m.cp wasn't given as an index
        idx_dims = next(iter(var))
        if len(idx_dims) != len(idxs) + 1:
            return arr.reshape(nfe, ncp, *idxs)
        else:
            return arr.reshape(nfe, *idxs)

def get_indexes(nfe: int, ncp: int, *, one_based: bool) -> List[Tuple[int,int]]:
    """ Get indices to index into variables, taking care of the funky first finite element
    
        >>> get_indexes(nfe=4, ncp=3, one_based=True)
        [(1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
        """
    x = 1 if one_based else 0
    return [(i + x, j + x) for i in range(nfe) for j in range(ncp)
                                                if (i > 0 or j == ncp-1)]

def get_indexes_(m: ConcreteModel, *, one_based: bool) -> List[Tuple[int,int]]:
    return get_indexes(m.fe[-1], m.cp[-1], one_based=one_based)

def print_data_summary(data: Dict[str,Any], indent: int = 0) -> None:
    """Print a more summarized version of the nested dictionaries and lists returned
       by calling `robot.print_data_summary()`"""
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            dims = 'x'.join(map(str, val.shape))
            print(' '*indent + f'{key}: [{dims} np.ndarray]')
        
        elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            print(' '* indent + f'{key}: [ ', end='')
            for v in val:
                print('{')
                print_data_summary(v, indent + 4)
                print(' '*(indent+2) + '} ', end='')
            print('\n' + ' '*indent + ']')
        
        elif isinstance(val, dict):
            print(' '*indent + f'{key}: ' + '{')
            print_data_summary(val, indent + 2)
            print(' '*indent + '}')
        
        else:
            print(' '*indent + f'{key}: {val}')

class MarkovBinaryWalk():
    """Generate a Markov process random walk which alternates between two states:
    `a` and `b`. If in state `a`, return 1 then stay in state `a with probability
    `a_prob`. Otherwise, go to state `b`. If in state `b`, return 0 and stay in
    state `b` with probability `b_prob"""
    def __init__(self, a_prob, b_prob):
        self.a_prob = a_prob
        self.b_prob = b_prob
        self.state = 'a'

    def step(self):
        import random
        if self.state == 'a':
            self.state = 'a' if random.random() < self.a_prob else 'b'
            return 1
        else:
            self.state = 'b' if random.random() < self.b_prob else 'a'
            return 0
    
    def walk(self, n):
        return np.array([self.step() for _ in range(n)], dtype=int)

# utils for matplotlib #######################################################
def plot3d_setup(figsize=(10, 10),
                 dpi: int = 60,
                 title: str = '',
                 lim: float = 2.0,
                 height: float = 5.0,
                 show_grid: bool = True,
                 plot_ground: bool = True,
                 ground_lims: Optional[Tuple[Tuple,Tuple]] = None,
                 scale_plot_size: bool = True):
    """Set up a fairly "standard" figure for a 3D system, including:
    - setting up the title correctly
    - setting limits on and labelling axes
    - optionally removing the grid
    - plotting the ground (at z = 0m)
    - scaling the plot to reduce surrounding whitespace
    
    Returns a figure and axis
    """
    from matplotlib import pyplot as plt
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca(projection='3d')
    fig.tight_layout()
    
    if len(title) > 0:
        fig.suptitle(title, fontsize=20)
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, height)
    ax.set_xlabel('$X$ [m]'); ax.set_ylabel('$Y$ [m]'); ax.set_zlabel('$Z$ [m]')
    
    if show_grid is False:
        ax.grid(False)     # hide grid lines
        ax.set_xticks([])  # hide axes ticks
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')    # hmmm
        plt.grid(b=None)   # also, hmm
    
    def add_ground(ground_lims, color: str = 'green'):
        """ground_lims could be something like, (-10*lim, lim), (-lim, 10*lim)"""        
        ground = ax.plot_surface(*np.meshgrid(*ground_lims),
                                 np.zeros((2,2)),
                                 alpha=0.5,
                                 color=color,
                                 zorder=1)
        return ground
    
    if scale_plot_size is True:
        fig.subplots_adjust(left=-0.25, bottom=-0.25, right=1.25, top=1.25,
                            wspace=None, hspace=None)
    
    return fig, ax, add_ground

def update_3d_line(line, pt1: List[float], pt2: List[float]):
    """Update data in a 3D line, passed as two points"""
    line.set_data([[pt1[0], pt2[0]],
                   [pt1[1], pt2[1]]])
    line.set_3d_properties([pt1[2], pt2[2]])

def set_view(ax, along: Union[Tuple[float,float],str] = (45, 150)):
    """Set the angle for the 'camera' in matplotlib"""
    if type(along) == tuple:
        ax.view_init(elev=along[0], azim=along[1])
    
    else:
        assert along in ('x', 'y', 'z')
        if along == 'x':
            ax.view_init(elev=0, azim=0)
            ax.set_xlabel('')
        elif along == 'y':
            ax.view_init(elev=0, azim=-90)
            ax.set_ylabel('')
        elif along == 'z':
            ax.view_init(elev=90, azim=0)
            ax.set_zlabel('')

def track_pt(ax, pt: Union[Tuple[float,float], Tuple[float,float,float]], lim: float):
    """Adjust the limits of a plot so as to track a 2D or 3D point"""
    ax.set_xlim(pt[0]-lim, pt[0]+lim)
    ax.set_ylim(pt[1]-lim, pt[1]+lim)
    if len(pt) > 2:
        ax.set_zlim(pt[2]-lim, pt[2]+lim)

def get_limsx(link, dy: float = 2., z: float = 2) -> Tuple[Tuple,Tuple,Tuple]:
    """Get limits for a movement along the positive `x`-axis. Use like,
    >>> robot.animate(lims=get_lims(robot.links[0],dy=4),
                      view_along=(35, -120),
                      plot3d_config={'figsize': (15, 3), 'dpi': 100})
    """
    xdata = get_vals(link['q'], (link.pyomo_sets['q_set'],))[:,0,0]
    xlim = max(xdata) + 1
    return ((0, xlim), (-dy/2, dy/2), (0, z))

def get_limsy(link, dx: float = 2., z: float = 2) -> Tuple[Tuple,Tuple,Tuple]:
    """Get limits for a movement along the positive `y`-axis. Use like,
    >>> robot.animate(lims=get_lims(robot.links[0],dx=4),
                      view_along=(35, -120),
                      plot3d_config={'figsize': (15, 3), 'dpi': 100})
    """
    ydata = get_vals(link['q'], (link.pyomo_sets['q_set'],))[:,0,1]
    ylim = max(ydata) + 1
    return ((-dx/2, dx/2), (0, ylim), (0, z))

def get_limsxy(link, z: float = 2) -> Tuple[Tuple,Tuple,Tuple]:
    xdata = get_vals(link['q'], (link.pyomo_sets['q_set'],))[:,0,0]
    xlims = (min(xdata) - 1, max(xdata) + 1)
    ydata = get_vals(link['q'], (link.pyomo_sets['q_set'],))[:,0,1]
    ylims = (min(ydata) - 1, max(ydata) + 1)
    return (xlims, ylims, (0, z))

# Slightly modified version of `delegates` from fast.ai:
# https://render.githubusercontent.com/view/ipynb?commit=d200d3e45f6a6e3bc4b2dba41242004cff065a19&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6661737461692f6661737461695f6465762f643230306433653435663661366533626334623264626134313234323030346366663036356131392f6465762f30315f636f72655f666f756e646174696f6e2e6970796e62&nwo=fastai%2Ffastai_dev&path=dev%2F01_core_foundation.ipynb&repository_id=150974871&repository_type=Repository
from types import FunctionType
def delegates(to_func: FunctionType):
    import inspect
    """Decorator: replace `**kwargs` in signature with params from `to_func`

    >>> def basefoo(x: int = 1, y: str = '2'):
    ...     print(x, y)
    >>> @delegates(basefoo)
    ... def foo(a, b=1, **kwargs):
    ...     # delegate to `basefoo` inside `foo`
    ...     basefoo(**kwargs)
    >>> help(foo)
    Help on function foo in module __main__:

    foo(a, b=1, x: int = 1, y: str = '2')
    """
    def _f(f):
        from_f = getattr(f, '__func__', f)
        
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)

        sigd.update({
            k: v for k, v in inspect.signature(to_func).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        })
        
        from_f.__signature__ = sig.replace(parameters=sigd.values()) # type: ignore
        from_f.__delwrap__ = to_func

        return f
    
    return _f

########## ATTIC
# def prune_small(expr, eta=1e-13):
#     """Prunes small elements (ie, less than `eta=1e-15`) from an expression"""
#     new_args = []
#     for val in expr.args:
#         if val.is_number and abs(val) < eta:
#             new_args.append(sp.Integer(0))
#         else:
#             new_args.append(prune_small(val))
    
#     return expr.func(*new_args) if len(new_args) > 0 else expr
