"""
Helper functions used for exercises in the MHBF Computational Course
"""

import inspect
import numpy as np


def assert_var_defined(varname, func=False):
    """
    Test that the a variable with name `varname` is defined in the current
    namespace.

    Paramaters
    ----------
    varname : str
        Name of the variables.
    func : bool, optional
        If `True`, check if the variable is callable.

    Raises
    ------
    AssertionError
        If variable is not defined or not callable (for `func=True`).
    """
    defined_vars = inspect.currentframe().f_back.f_locals
    if varname not in defined_vars:
        raise AssertionError(
            f"`{varname}` is not defined. Please use the variable name used in the "
            "exercise description."
        )
    elif func and not callable(defined_vars[varname]):
        raise AssertionError(f"`{varname}` is not callable. This should be a function!")

        
def euler_maruyama(ffun, gfun, y0, t, fargs={}, gargs={}):
    """
    Solve a system of stochastic differential equations (SDE). This function behaves
    like `scipy.integrate.odeint`.
    
    Parameters
    ----------
    ffun : callable
        A function with signature `ffun(y, t, ...)` that computes the deterministic part
        f(Y(t)) of the SDE.
    gfun : callable
        A function with signature `gfun(y, t, ...)` that computes the stochastic part
        g(Y(t)) of the SDE.
    y0 : numpy.ndarray
        Same as `y0` in `scipy.integrate.odeint`.
    t : float
        Same as `t` in `scipy.integrate.odeint`.
    fargs : dict, optional
        Extra keyword arguments passed to `ffun`.
    gargs : dict, optional
        Extra keyword arguments passed to `gfun`.
        
    Returns
    -------
    y : numpy.ndarray
        Same as `y` in `scipy.integrate.odeint`.
    """

    y = np.zeros((len(t), len(y0)))
    y[0, :] = y0

    dt = np.diff(t)
    dW = np.random.standard_normal((dt.size, len(y0)))
    for i, time in enumerate(t[1:]):
        y[i + 1, :] = (
            y[i, :]
            + ffun(y[i, :], t[i], **fargs) * dt[i]
            + gfun(y[i, :], t[i], **gargs) * dW[i] * np.sqrt(dt[i])
        )

    return y


### Functions used internally for autograding
from IPython.core.ultratb import AutoFormattedTB as _AutoFormattedTB

def _run_tests(test_functions):
    """
    Run test functions and print the Exception messages without raising an Exception.
    Only raise an Exception in the end if at least one of the tests failed. Only
    catches AssertionErrors.

    Parameters
    ----------
    tests_functions : list
        List of callable test functions.

    Raises
    ------
    AssertionError
        If at least one of the test functions would have raised an AssertionError.
    """
    auto_tb = _AutoFormattedTB(mode='Context', color_scheme='Neutral', tb_offset=1)
    num_errors = 0
    for test_func in test_functions:
        try:
            test_func()
        except AssertionError:
            auto_tb()
            num_errors += 1

    if num_errors > 0:
        raise AssertionError(f"{num_errors}/{len(test_functions)} tests failed.")
