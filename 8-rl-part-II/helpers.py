"""
Helper functions used for exercises in the MHBF Computational Course
"""

import inspect
import matplotlib.pyplot as plt
from IPython.display import HTML


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


def display_animation(animation):
    """
    Create a display object that displays an animation as interactive Javascript widget
    in Jupyter.

    Parameters
    ----------
    animation : matplotlib.animation
        An animation object created with matplotlib.

    Returns
    -------
    ipython.core.HTML
        An html object that displays html code in an Jupyter notebook.
    """
    plt.close(animation._fig)
    return HTML(animation.to_jshtml())


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
