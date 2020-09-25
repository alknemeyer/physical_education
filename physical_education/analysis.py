from pyomo.environ import value as pyovalue
import numpy as np
import numbers
from .motor import torques, power
from .system import System3D


def get_power_values(robot: System3D, force_scale: float):
    nfe = len(robot.m.fe)
    power_arr = []

    for motor in torques(robot):
        _power = np.array([
            abs(pyovalue(P)) * force_scale
            for P in power(motor, robot.pyo_variables)
        ]).reshape((nfe, -1))

        power_arr.append(_power)

    return power_arr


def print_power_values(robot, force_scale: float):
    power_arr = get_power_values(robot, force_scale)

    s = 'Motor'.ljust(20) + '| Mean absolute power'
    print(s + '\n' + '-'*len(s))

    total_power = 0.
    for motor, _power in zip(torques(robot), power_arr):
        avg_power = np.sum(np.mean(_power, axis=0))
        print(f'{motor.name:20}|{avg_power:6.1f} W')
        total_power += avg_power

    print(f'\nTotal average absolute power = {total_power:.1f} W')
