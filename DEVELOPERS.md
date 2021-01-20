# Publishing a new version
Install [flit](https://flit.readthedocs.io/en/latest/), which makes publishing packages ridiculously easy. Next, increase the `__version__` number in [`physical_education/__init__.py`](physical_education/__init__.py). Then, create a (local) [tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) the commit and publish:

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
For example, hard stop joints between links

## Switch to a different animation library
`matplotlib` isn't great for animations - it's quite slow, not interactive, and so on. Switching to eg. `pyqtgraph` could be nice (and might not require that much work) but we'd need to make sure it works with a remote setup and across operating systems (ie not just linux)

Other libraries for doing things in 3D include:
   * https://github.com/K3D-tools/K3D-jupyter/tree/master
   * https://github.com/rougier/matplotlib-3d
   * https://matplotlib.org/matplotblog/posts/custom-3d-engine/
   * https://lorensen.github.io/VTKExamples/site/Python/Utilities/Animation/ https://github.com/marcomusy/vedo/blob/master/examples/notebooks/volumetric/tensors.ipynb (also can't seem to save videos)
   * https://threejs.org/docs/index.html#examples/en/exporters/ColladaExporter (use pythreejs https://pythreejs.readthedocs.io/en/stable/introduction.html - eg: https://github.com/jupyter-widgets/pythreejs/blob/master/examples/Animation.ipynb) (can't seem to save videos)
   * or maybe pydy?
      * https://github.com/pydy/pydy-tutorial-human-standing/blob/master/notebooks/n08_visualization.ipynb
      * https://github.com/pydy/pydy/tree/master/pydy/viz
   * pyqtgraph (doesn't seem to work in notebooks)

BUT the folloing restrictions apply:
   * works using a remote computing setup (like eg JupyterLab)
   * have the ability to save videos (eg. `.mp4`)
   * not too difficult to work with (preferably has primitives for cylinders, rectangles, etc)

If using matplotlib, check out:
* https://matplotlib.org/matplotblog/posts/custom-3d-engine/
* https://github.com/rougier/matplotlib-3d/blob/master/doc/README.md

Plotly:
* https://community.plotly.com/t/basic-3d-cylinders/27990
* https://plotly.com/python/streamtube-plot/
* BUT can't seem to save videos of animations?

## Use the `logging` module instead of/in addition to my stuff
Would be very simple -- just replace some of the calls to `print` in `visual.py` with `logger.info`. See my [blog post](https://alknemeyer.github.io/technical/embedded-comms-with-python-part-2/#setting-up-logging) on how you might do this

## Remove `3D` suffix from everything
Either that, or go ahead and add 2D versions of things, possibly putting all 3D stuff in a folder like `d3` and 2D stuff in `d2`

Ie, you'd have:
```
physical_education/
   d3/
      system.py
      links.py
      drag.py
   d2/
      system.py
      links.py
   visualisation.py
   etc.py
```

Otherwise, you might decide that abstracting away 2D just is more work than it's worth, and it might even not be a good idea (see [this talk](https://www.deconstructconf.com/2019/dan-abramov-the-wet-codebase) and why you _shouldn't_ abstract everything)

## Document sources of equations, data, etc
An embarrassing amount of the equations, theory, and so on simply isn't documented properly. The maths in particular appears as equations without context. It would be good to cite sources for the complex stuff, and proviled more detailed explanations for everything else. Some of the things are quite routine (eg generalized forces) -- they could be explained in a `theory.md` document?