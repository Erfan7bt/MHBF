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


def gaussian_spectrum_1D(rand_signal, epsilon, sample_period=1.0):
    """
    Low pass filter a 1D signal with power spectral density (PSD):
    PSD(k) = 1 / (- k² / 2ϵ)

    Parameters
    ----------
    rand_signal : numpy.ndarray
        1D array with random values (white noise).
    epsilon : float
        Width of the Gaussian PSD.
    sample_period : float, optional
        Sampling period of `rand_signal`. That means the time difference
        between consecutive signal measurements.

    Returns
    -------
    filtered_rand_signal : numpy.ndarray
        Low pass filtered signal (in time domain).
    """
    # The `rand_signal` signal is finite and discrete. Therefore, the Discrete Fourier
    # Transoform is used to get the Fourier coefficients for a finite set of discrete
    # Fourier frequencies (k). An efficient implementation of the Discrete Fourier
    # Transform is the Fast Fourier Transform, which we will use below (np.fft.fft). The
    # number of Fourier frequencies in your transformed signal (in frequency domain) is
    # the same as the number of discrete time points in your original signal (in time
    # domain).
    fft_rand_signal = np.fft.fft(rand_signal)

    # The np.fft.fft returned the Fourier coefficients. But for which frequencies? Now
    # that depends on the sampling period of the original signal, that means the time
    # difference between the discrete time points at which you recorded (or in your
    # case, randomly sampled) your signal. To get the correct Fourier frequencies, you
    # can use np.fft.fftfreq, which takes your signal size and its sampling period as
    # input.
    frq_rand_signal = np.fft.fftfreq(rand_signal.size, sample_period)

    # The PSD of a signal is defined as PSD(k) = |F(k)|^2, where F(k) is the Fourier
    # coefficient of frequency k. Therefore, we choose our filter in frequency domain to
    # be the sqrt of the PSD that we wan't to get.
    filter_low = np.sqrt(np.exp(-(frq_rand_signal ** 2) / (2 * epsilon)))
    # Normalize the filter
    filter_low = filter_low / np.sum(filter_low)

    # Now we apply the filter in Fourier domain (which is the same as convolving in time
    # domain, see Convolution Theorem)
    filtered_fft_rand_signal = fft_rand_signal * filter_low

    # And last, we transform the filtered signal back to the time domain, using the
    # inverse Fast Fourier Transform (np.fft.ifft).
    filtered_rand_signal = np.real(np.fft.ifft(filtered_fft_rand_signal))

    return filtered_rand_signal


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