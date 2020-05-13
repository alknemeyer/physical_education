"""
Running optim_lib as main currently doesn't do anything, but it could be useful
to set up as some easy way of running experiments in parallel? Eg. something like,

    $ python -m optim_lib setup_file.py --parallel-solves 4 --output-dir logs/ --until-successfull-solves 100 --save-model

which would expect a certain file structure in setup_file.py, solving 4 problems at a time until
there are 100 successfull solves, each time saving the model and IPOPT logs to logs/

setup_file.py might look like,
```
import optim_lib
# other imports...

def get_model() -> System3D:
    # define model, which could just be loading it from a pickle file
    # this gets run once
    with open('cheetah.robot', 'rb') as f:
        robot, add_pyomo_constraints = dill.load(f)
    return robot

def setup_model(robot: System3D) -> pyomo.environ.Solver:
    # make a pyomo model and add constraints
    robot.make_pyomo_model(nfe=50, collocation='euler', total_time=0.25)
    add_pyomo_constraints(robot)
    periodic_gallop_test(robot, avg_vel=10, include_cot=True)
    return solver
```

So the script would run something like,

>>> master_robot = get_model()
>>> for i in range(N):
...     robot = copy.deepcopy(master_robot)
...     solver = setup_robot(robot)
...     solver.solve(robot.m, tee=False)

except in parallel

None of that has been implemented, but it could be a useful future thing to do!
"""
