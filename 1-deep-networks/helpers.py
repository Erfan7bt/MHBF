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


class DiffuseTreeSampler:
    """
    Implementation of a branching diffusion process to generate hierarchically
    structured data, consisting of items `x` and features `y`. Items `x` will be used
    as input, features `y` as output during network training. See exercise `2.0 a)` for
    expalanation of the data generation process.

    Parameters
    ----------
    feature_dim : float
        Number of features (or properties) per item. These will be used as output
        targets during network training.
    tree_depth : int
        The depth (number of levels) of the diffusion tree
    branching_factor : int
        Number of branches at each node of the tree
    sample_epsilon : float
        Probability to switch sign at each level

    Examples
    --------
    >>> hierarchical_tree = DiffuseTreeSampler(
    >>>     features_dim, tree_depth=3, branching_factor=2, sample_epsilon=0.5
    >>> )
    >>> features, items = hierarchical_tree.sample_data()

    """

    def __init__(self, feature_dim, tree_depth, branching_factor, sample_epsilon):
        self.feature_dim = feature_dim
        self.num_examples = branching_factor ** tree_depth
        self.tree_depth = tree_depth
        self.branching_factor = branching_factor
        self.sample_epsilon = sample_epsilon

    def sample_feature(self):
        """
        Sample a single feature across items.
        """
        samples_per_tree_layer = [
            self.branching_factor ** i for i in range(1, self.tree_depth + 1)
        ]
        feature_tree = [np.random.choice([-1, 1], p=[0.5, 0.5], size=1)]
        for l in range(self.tree_depth):
            switch = np.random.choice(
                [-1, 1],
                p=[self.sample_epsilon, 1 - self.sample_epsilon],
                size=samples_per_tree_layer[l],
            )
            next_layer = np.repeat(feature_tree[-1], self.branching_factor)
            feature_tree.append(next_layer * switch)
        return feature_tree[-1]

    def sample_data(self):
        """
        Sample multiple features for multiple items where each feature for all items is
        sampled (diffuses) independently.

        Returns
        -------
        features_out : numpy.ndarray
            2D array of feature vectors for each data sample.
        items_out : numpy.ndarray
            2D array of one-hot item vectors for each datasample. This is a unit matrix.
        """
        features = []
        for tar in range(self.feature_dim):
            target_temp = self.sample_feature()
            features.append(target_temp)

        features_out = np.array(features).T
        items_out = np.diag(np.ones(self.num_examples))
        return features_out, items_out


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
        except Exception:
            auto_tb()
            num_errors += 1

    if num_errors > 0:
        raise AssertionError(f"{num_errors}/{len(test_functions)} tests failed.")
