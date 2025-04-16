import numpy as np

def make_line_data(n_samples=100, beta_0=0, beta_1=1, sd=1, X_low=-10, X_high=10, random_seed=None):
    """
    Generate linear data with noise.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    beta_0 : int
        The intercept of true line
    beta_1 : int
        The slope of true line
    sd : int
        Standard deviation of the Gaussian noise added to the true line
    X_low : int
        Lower bound for the simulated X values
    X_high : int
        Upper bound for the simulated X values
    random_seed : int, optional
        Seed for the random number generator

    Returns
    -------
    tuple of np.ndarray
        X, the input values and Y, the output values
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    X = np.random.uniform(low = X_low, high = X_high, size = (n_samples, 1))
    Y = beta_0 + beta_1 * X.ravel() + np.random.normal(scale = sd, size =n_samples)
    return X, Y

def make_sine_data(n_samples=100, sd=1, X_low=-6, X_high=6, random_seed=None):
    """
    Generate data for nonlinear regression based on a sine function.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    sd : float, default=1
        Standard deviation of the normally distributed errors.
    X_low : float, default=-6
        Lower bound for simulated X values.
    X_high : float, default=6
        Upper bound for simulated X values.
    random_seed : int or None, default=None
        Seed to control randomness.

    Returns
    -------
    tuple
        A tuple containing the x and y arrays. x is a 2D array with shape (n_samples, 1)
        and y is a 1D array with shape (n_samples, ). x contains the simulated X values
        and y contains the corresponding sine values with added normally distributed errors.
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)

    X = np.random.uniform(X_low, X_high, size=(n_samples, 1))
    y = np.sin(X).ravel() + np.random.normal(scale=sd, size=n_samples)

    return X, y

def split_data(X, y, holdout_size=0.2, random_seed=None):
    """
    Split the data into train and test sets.

    Parameters
    ----------
    X : ndarray
        The feature data to be split. A 2D array with shape (n_samples, 1).
    y : ndarray
        The target data to be split. A 1D array with shape (n_samples, ).
    holdout_size : float, default=0.2
        The proportion of the data to be used as the test set.
    random_seed : int, optional
        Seed to control randomness.

    Returns
    -------
    tuple
        The split train and test data: (X_train, X_test, y_train, y_test).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    test_size = int(holdout_size * X.shape[0])
    X_train = X[:-test_size]
    X_test = X[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    return X_train, X_test, y_train, y_test



