import numpy as np


# https://vcs.ynic.york.ac.uk/ynic-debian/pymc3/-/blob/4870ca7c72250e33dfeba9f231fd4a9f9d0fce8f/pymc/stats.py
def _make_indices(dimensions):
    # Generates complete set of indices for given dimensions
    level = len(dimensions)
    if level == 1:
        return list(range(dimensions[0]))
    indices = [[]]
    while level:
        _indices = []
        for j in range(dimensions[level - 1]):
            _indices += [[j] + i for i in indices]
        indices = _indices
        level -= 1
    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices


# adaptation from krushke
# https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/master/hpd.py
def _calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0 - alpha
    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]
    if len(interval_width) == 0:
        raise ValueError("Too few elements for interval calculation")
    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max


# adaptation from krushke
# https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/master/hpd.py
def hpd(x, alpha=0.05):
    """Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)

    """
    x = x.copy()
    if x.ndim > 1:  # For multivariate node
        tx = np.transpose(
            x, list(range(x.ndim))[1:] + [0]
        )  # Transpose first, then sort
        dims = np.shape(tx)  # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))
        for index in _make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass
            sx = np.sort(tx[index])  # Sort trace
            intervals[index] = _calc_min_interval(sx, alpha)  # Append to list
        return np.array(intervals)  # Transpose back before returning
    else:
        sx = np.sort(x)  # Sort univariate node
        return np.array(_calc_min_interval(sx, alpha))
