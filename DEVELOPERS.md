# Publishing a new version

# Development ideas
## Trailing underscores
In some python libraries, there is a convention a trailing underscore in a variable name (eg. `my_var_`) indicates that the attribute exists only after _something_ has been done (in this library, it would be after pyomo model has been made). Eg in Scikit-Learn there is a convention of using a trailing underscore for Estimator attributes that exist only after the attribute has been fit (e.g. `LinearRegression().coef_`).

## Find type stubs for third party libraries
Eg. matplotlib, pyomo, numpy from:
   https://github.com/predictive-analytics-lab/data-science-types
