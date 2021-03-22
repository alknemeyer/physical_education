from typing import TYPE_CHECKING
from types import FrameType
from typing import Optional, Tuple, List, Set, Union, Dict, Any
from contextlib import contextmanager
import os
import inspect
import numpy as np
import sympy as sp


# printing

USE_COLOURS: bool = True
DEBUG_MODE: bool = True
ONCE_SET: Set[str] = set()


class bcolors:
    # https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLINK = '\33[5m'


# fix for windows (based on stackoverflow link above). Haven't tested it myself
if os.name == 'nt':
    os.system('color')  # type: ignore


@contextmanager
def no_colour_output():
    try:
        global USE_COLOURS
        USE_COLOURS = False
        yield
    finally:
        USE_COLOURS = True


def get_useful_info(f: FrameType) -> str:
    # s is something like:
    # <code object remove_constraint_if_exists at 0x000000000aeeb100, file "/home/alex/alknemeyer-msc/python/physical_education/visual.py", line 301>
    s = str(f.f_back.f_back.f_code)

    # the indexing removes the leading '<code object' and trailing '>'
    # the first three underscores receive 'at 0xabcd, file'
    # the final underscore handles 'line'
    # middle is splatted this way in case there are spaces in the file path
    fn_name, _, _, _, *middle, _, lineno = s[13:-1].split()

    # indexing removes the comma at the end + surrounding quotes (regardless of if their single or double)
    filepath = ''.join(middle)[1:-2].rstrip('.py').lstrip('<').rstrip('>')

    if 'physical_education' in filepath:
        # eg: '/home/alex/alknemeyer-msc/python/physical_education/visual'
        idx = filepath.index('physical_education')
        filepath = filepath[idx:].replace('/', '.')
    elif 'ipython' in filepath:
        # eg: 'ipython-input-34-ce5fcbaca0c1'
        filepath = filepath[:filepath.rfind('-')]

    return f'{filepath}.{fn_name}(), line {lineno}: '


def _printutil(prefix: str, s: str, colour: str, once: bool):
    if once is True:
        if s in ONCE_SET:
            return
        else:
            ONCE_SET.add(s)

    try:
        # Ref: https://stackoverflow.com/a/57712700/
        # calling_func = f'{inspect.currentframe().f_back.f_back.f_code}(): '
        frame = inspect.currentframe()
        if frame is not None:
            calling_func = get_useful_info(frame)
        else:
            calling_func = ''
    except:
        calling_func = ''

    if USE_COLOURS is True:
        print(bcolors.BOLD + colour + str(prefix) + bcolors.ENDC,
              colour + calling_func + str(s) + bcolors.ENDC)
    else:
        print(str(prefix), calling_func + str(s))


def info(s: str, prefix: str = '[INFO 💡]', once: bool = False):
    _printutil(prefix, s, bcolors.OKBLUE, once)


def warn(s: str, prefix: str = '[WARNING]', once: bool = False):
    _printutil(prefix, s, bcolors.FAIL, once)


def error(s: str, prefix: str = '[ERROR💩]', once: bool = False):
    _printutil(prefix, s, bcolors.FAIL + bcolors.BLINK, once)


def debug(s: str, prefix: str = '[DEBUG🐞]', once: bool = False):
    if DEBUG_MODE is True:
        _printutil(prefix, s, bcolors.WARNING, once)


def success(s: str, prefix: str = '[SUCCESS]', once: bool = False):
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


def print_data_summary(data: Dict[str, Any], indent: int = 0) -> None:
    """Print a more summarized version of the nested dictionaries and lists returned
       by calling `robot.print_data_summary()`"""
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            dims = 'x'.join(map(str, val.shape))
            print(' '*indent + f'{key}: [{dims} np.ndarray]')

        elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            print(' ' * indent + f'{key}: [ ', end='')
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

# matplotlib stuff


def plot3d_setup(figsize=(10, 10),
                 dpi: int = 60,
                 title: Optional[str] = None,
                 show_grid: bool = True,
                 xyz_labels: Optional[Tuple[str, str, str]] = None,
                 scale_plot_size: bool = True):
    """Set up a fairly "standard" figure for a 3D system, including:
    - setting up the title correctly
    - setting limits on and labelling axes
    - optionally removing the grid
    - plotting the ground (at z = 0m)
    - scaling the plot to reduce surrounding whitespace

    Returns a figure and axis
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca(projection='3d')
    fig.tight_layout()

    if title is not None:
        fig.suptitle(title, fontsize=20)

    if xyz_labels is None:
        ax.set_xlabel('$X$ [m]')
        ax.set_ylabel('$Y$ [m]')
        ax.set_zlabel('$Z$ [m]')
    else:
        ax.set_xlabel(xyz_labels[0])
        ax.set_ylabel(xyz_labels[1])
        ax.set_zlabel(xyz_labels[2])

    if show_grid is False:
        ax.grid(False)     # hide grid lines
        ax.set_xticks([])  # hide axes ticks
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')    # hmmm
        plt.grid(b=None)   # also, hmm

    def add_ground(ground_lims: Tuple[Tuple[float, float], Tuple[float, float]],
                   color: str = 'green',
                   alpha: float = 0.2):
        """ground_lims could be something like, (-10*lim, lim), (-lim, 10*lim)"""
        xx, yy = ground_lims
        ground = ax.plot_surface(*np.meshgrid(xx, yy),
                                 np.zeros((2, 2)),
                                 alpha=alpha,
                                 color=color,
                                 zorder=1)
        return ground

    if scale_plot_size is True:
        fig.subplots_adjust(left=-0.25, bottom=-0.25, right=1.25, top=1.25,
                            wspace=None, hspace=None)

    return fig, ax, add_ground


def plotgrass(ax, xlo: float, xup: float, ylo: float, yup: float,
              npoints: int = 30,
              color: str = 'orange',
              height: float = 0.2,
              alpha: float = 0.5):
    def getvec():
        return np.random.random((npoints, npoints, 1))

    x = getvec()*(xup-xlo) + xlo
    y = getvec()*(yup-ylo) + ylo
    z = getvec()*0

    u = (getvec() - 0.5)/5
    v = (getvec() - 0.5)/5
    w = getvec()*height

    return ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0, color=color, alpha=alpha)


def update_3d_line(line, pt1: List[float], pt2: List[float]):
    """Update data in a 3D line, passed as two points"""
    line.set_data([[pt1[0], pt2[0]],
                   [pt1[1], pt2[1]]])
    line.set_3d_properties([pt1[2], pt2[2]])


def set_view(ax, along: Union[Tuple[float, float], str] = (45, 150)):
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


def track_pt(ax, pt: Union[Tuple[float, float], Tuple[float, float, float]], lim: float):
    """Adjust the limits of a plot so as to track a 2D or 3D point"""
    ax.set_xlim(pt[0]-lim, pt[0]+lim)
    ax.set_ylim(pt[1]-lim, pt[1]+lim)
    if len(pt) > 2:
        ax.set_zlim(pt[2]-lim, pt[2]+lim)


Bounds = Tuple[float, float]


def get_limsx(link, dy: float = 2., z: float = 2) -> Tuple[Bounds, Bounds, Bounds]:
    """Get limits for a movement along the positive `x`-axis. Use like,
    >>> robot.animate(lims=get_lims(robot.links[0],dy=4),
                      view_along=(35, -120),
                      plot3d_config={'figsize': (15, 3), 'dpi': 100})
    """
    from .utils import get_vals
    xdata = get_vals(link['q'], (link.pyomo_sets['q_set'],))[:, 0, 0]
    xlim = max(xdata) + 1
    return ((0, xlim), (-dy/2, dy/2), (0, z))


def get_limsy(link, dx: float = 2., z: float = 2) -> Tuple[Bounds, Bounds, Bounds]:
    """Get limits for a movement along the positive `y`-axis. Use like,
    >>> robot.animate(lims=get_lims(robot.links[0],dx=4),
                      view_along=(35, -120),
                      plot3d_config={'figsize': (15, 3), 'dpi': 100})
    """
    from .utils import get_vals
    ydata = get_vals(link['q'], (link.pyomo_sets['q_set'],))[:, 0, 1]
    ylim = max(ydata) + 1
    return ((-dx/2, dx/2), (0, ylim), (0, z))


def get_limsxy(link, z: float = 2) -> Tuple[Bounds, Bounds, Bounds]:
    from .utils import get_vals
    xdata = get_vals(link['q'], (link.pyomo_sets['q_set'],))[:, 0, 0]
    xlims = (min(xdata) - 1, max(xdata) + 1)
    ydata = get_vals(link['q'], (link.pyomo_sets['q_set'],))[:, 0, 1]
    ylims = (min(ydata) - 1, max(ydata) + 1)
    return (xlims, ylims, (0, z))


class LineAnimation:
    def __init__(self, pos1: sp.Matrix, pos2: sp.Matrix,
                 sp_variables: List[sp.Symbol]) -> None:
        assert pos1.shape == (3, 1)
        assert pos2.shape == (3, 1)

        from .utils import lambdify_EOM
        self.pos1func = lambdify_EOM(pos1, sp_variables)
        self.pos2func = lambdify_EOM(pos2, sp_variables)

    def animation_setup(self, fig, ax, data: List[List[float]]):
        self.line = ax.plot([], [], [],
                            linewidth=1,
                            color='darkgray')[0]

        self.plot_data = [np.empty((len(data), 3)),
                          np.empty((len(data), 3))]

        for idx, d in enumerate(data):  # xyz of top and bottom of the link
            self.plot_data[0][idx, :] = [f(d) for f in self.pos1func]
            self.plot_data[1][idx, :] = [f(d) for f in self.pos2func]

    def animation_update(self, fig, ax,
                         fe: Optional[int] = None,
                         t: Optional[float] = None,
                         t_arr: Optional['np.ndarray'] = None,
                         track: bool = False):
        if fe is not None:
            pos1_xyz = self.plot_data[0][fe-1]
            pos2_xyz = self.plot_data[1][fe-1]
        else:
            pos1_xyz = [np.interp(t, t_arr, self.plot_data[0][:, i])
                        for i in range(3)]
            pos2_xyz = [np.interp(t, t_arr, self.plot_data[1][:, i])
                        for i in range(3)]

        update_3d_line(self.line, pos1_xyz, pos2_xyz)

        if track is True:
            lim = 1.0
            if pos1_xyz[2] < lim:
                pos1_xyz = (float(pos1_xyz[0]), float(pos1_xyz[1]), lim)
            track_pt(ax, pos1_xyz, lim=lim)  # type: ignore

    def cleanup_animation(self, fig, ax):
        try:
            del self.line
        except:
            pass


def data_for_3d_cylinder(top, bottom, radius: float, nsides: int):
    from numpy import sin, cos, pi, arcsin  # type: ignore

    nsides += 1
    tx, ty, tz = top = np.array(top)
    bx, by, bz = bottom = np.array(bottom)

    # zero to 2pi
    z2pi = np.linspace(0, 2*pi, nsides)

    x_grid = np.array([bx + sin(z2pi)*radius,
                       tx + sin(z2pi)*radius])
    y_grid = np.array([by + cos(z2pi)*radius,
                       ty + cos(z2pi)*radius])
#     z_grid = np.array([bz + scale*radius*0,
#                        tz + scale*radius*0])

    # adjust height of cylinder
    theta = np.linspace(0, 2*pi, nsides)
    scale = sin(z2pi)
    z = np.linspace(bz, tz, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    n = top - bottom
    b = np.array([0, 0, 1])
    th_to_xy_plane_rad = arcsin(
        b.dot(n) / np.linalg.norm(b) / np.linalg.norm(n)
    )
    z_grid = np.repeat(np.array([bz, tz]),
                       x_grid.shape[1]).reshape(x_grid.shape)
    z_grid += scale*radius * sin(th_to_xy_plane_rad)

    return x_grid, y_grid, z_grid


class CylinderAnimation:
    def __init__(self, pos1: sp.Matrix, pos2: sp.Matrix,
                 sp_variables: List[sp.Symbol],
                 radius: float,
                 nsides: int = 10) -> None:
        assert pos1.shape == (3, 1)
        assert pos2.shape == (3, 1)

        from .utils import lambdify_EOM
        self.pos1func = lambdify_EOM(pos1, sp_variables)
        self.pos2func = lambdify_EOM(pos2, sp_variables)
        self.radius = radius
        self.nsides = nsides

    def animation_setup(self, fig, ax, data: List[List[float]]):
        self.plot_data = [np.empty((len(data), 3)),
                          np.empty((len(data), 3))]

        for idx, d in enumerate(data):  # xyz of top and bottom of the link
            self.plot_data[0][idx, :] = [f(d) for f in self.pos1func]
            self.plot_data[1][idx, :] = [f(d) for f in self.pos2func]

    def animation_update(self, fig, ax,
                         fe: Optional[int] = None,
                         t: Optional[float] = None,
                         t_arr: Optional['np.ndarray'] = None,
                         track: bool = False):
        if fe is not None:
            pos1_xyz = self.plot_data[0][fe-1]
            pos2_xyz = self.plot_data[1][fe-1]
        else:
            pos1_xyz = [np.interp(t, t_arr, self.plot_data[0][:, i])
                        for i in range(3)]
            pos2_xyz = [np.interp(t, t_arr, self.plot_data[1][:, i])
                        for i in range(3)]

        if hasattr(self, 'line'):
            self.line.remove()
            del self.line

        Xc, Yc, Zc = data_for_3d_cylinder(
            pos1_xyz, pos2_xyz, radius=self.radius, nsides=self.nsides)
        self.line = ax.plot_surface(Xc, Yc, Zc, color='darkgray')

        if track is True:
            lim = 1.0
            if pos1_xyz[2] < lim:
                pos1_xyz = (float(pos1_xyz[0]), float(pos1_xyz[1]), lim)
            track_pt(ax, pos1_xyz, lim=lim)  # type: ignore

    def cleanup_animation(self, fig, ax):
        try:
            self.line.remove()
            del self.line
        except:
            pass


# TODO: refactor plotting code from drag and foot into
#       a QuiverAnimation class, similar to the one above

if TYPE_CHECKING:
    from .system import System3D


def stitch_together_animation(robot: 'System3D', data: List[dict]):
    # exclude the last finite element of each of these
    nfe = sum(d['nfe'] for d in data) - len(data)

    total_time = sum(sum(d['hm0']*d['hm'][1:]) for d in data)

    robot.make_pyomo_model(nfe=nfe, collocation='implicit_euler', total_time=total_time,
                           vary_timestep_within=(0.5, 1.5), include_dynamics=False)

    def rot2d(θ):
        return np.array([
            [np.cos(θ), np.sin(θ)],  # type: ignore
            [-np.sin(θ), np.cos(θ)],  # type: ignore
        ])

    nfe_otg = 1
    bodyq = robot.links[0]['q']

    def inc(linkq, var: str):
        val = linkq[nfe_otg-1, 1, var].value

        for fe in range(nfe_otg, nfe_otg + dnfe):
            linkq[fe, 1, var].value += val

    for idx, d in enumerate(data):
        dnfe = d['nfe'] - 1

        for fes in range(dnfe):
            robot.init_from_dict_one_point(d, fed=nfe_otg+fes, cpd=1, fes=fes, cps=0,
                                           skip_if_fixed=False, skip_if_not_None=False, fix=False)

        if idx != 0:
            for link in robot.links:
                inc(link['q'], 'psi')

            inc(bodyq, 'x')
            inc(bodyq, 'y')

            def var(fe, coord): return bodyq[nfe_otg+fe, 1, coord]
            prevx = var(0, 'x').value
            prevy = var(0, 'y').value

            # rotate the [x,y] positions and then increment from previous values
            for fe in range(dnfe):
                psi = var(fe, 'psi').value
                arr = np.array([var(fe, 'x').value - prevx,
                                var(fe, 'y').value - prevy])
                xy = rot2d(psi).T @ arr

                var(fe, 'x').value = xy[0] + prevx
                var(fe, 'y').value = xy[1] + prevy

        nfe_otg += dnfe

    return robot
