# Publishing a new version
Install [flit](https://flit.readthedocs.io/en/latest/), which makes publishing packages ridiculously easy. Next, increase the `__version__` number in [physical_education/__init__.py](physical_education/__init__.py). Then, create a (local) [tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) the commit and publish:

```bash
# make a local tag with message "release v0.0.1"
$ git tag -a v0.0.1 -m "release v0.0.1"
# push local tag to remote repo
$ git push origin v0.0.1
# generate files into dist/ and upload them to pypi
$ flit publish
```

# Development ideas
## Trailing underscores
In some python libraries, there is a convention a trailing underscore in a variable name (eg. `my_var_`) indicates that the attribute exists only after _something_ has been done (in this library, it would be after pyomo model has been made). Eg in Scikit-Learn there is a convention of using a trailing underscore for Estimator attributes that exist only after the attribute has been fit (e.g. `LinearRegression().coef_`).

## Find type stubs for third party libraries
Eg. matplotlib, pyomo, numpy from:
   https://github.com/predictive-analytics-lab/data-science-types

## Add more link types
For example, prismatic links. Through some subclassing/refactoring, you may get away very writing very little code? Otherwise, it could be written as a node?

## Add more node types
For example, hard stop joints and springs between links

## Switch to a different animation library
`matplotlib` isn't great for animations - it's quite slow, not interactive, and so on. Switching to eg. `pyqtgraph` could be nice (and might not require that much work) but we'd need to make sure it works with a remote setup and across operating systems (ie not just linux)

Other libraries for doing things in 3D include:
   * https://github.com/K3D-tools/K3D-jupyter/tree/master
   * https://github.com/rougier/matplotlib-3d
   * https://matplotlib.org/matplotblog/posts/custom-3d-engine/
