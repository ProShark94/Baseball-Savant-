import numpy as np


class ConformalPredictor:
    """
    A conformal predictor for regression models.

    This class wraps around a base regression model and provides conformal prediction
    intervals for the predictions. The prediction interval provides a range in which
    the true target value is expected to lie with a certain level of confidence.

    Parameters
    ----------
    regressor : object
        A regression model that has fit and predict methods.

    alpha : float, optional
        The significance level for the prediction interval, by default 0.05. The
        prediction interval will cover the true target value with a probability of
        1 - alpha.

    Attributes
    ----------
    scores : ndarray
        The absolute differences between the observed target values and the
        predictions from the base regressor, used to compute the conformal scores.

    quantile : float
        The (1 - alpha) quantile of the conformal scores, used to determine the
        width of the prediction interval.
        

    """

    def __init__(self, regressor, alpha=0.05):
        self.regressor = regressor
        self.alpha = alpha
        self.scores = None
        self.quantile = None

    def fit(self, X, y):
        """
        Fit the base regressor and compute the conformal scores and quantile.

        Parameters
        ----------
        X : ndarray
            The input features for training, shape (n_samples, n_features).
        y : ndarray
            The target values for training, shape (n_samples,).

        Returns
        -------
        self : object
            Returns self for chaining.
        """
        self.regressor.fit(X, y)
        predictions = self.regressor.predict(X)
        self.scores = np.abs(y - predictions)
        self.quantile = np.quantile(self.scores, 1 - self.alpha)

    def predict(self, X_new):
        """
        Predict target values and their conformal prediction intervals for new data.

        Parameters
        ----------
        X_new : ndarray
            The input features for prediction, shape (n_samples_new, n_features).

        Returns
        -------
        tuple of ndarray
            A tuple containing three arrays: the predicted target values, the lower
            bounds of the prediction intervals, and the upper bounds of the
            prediction intervals, each with shape (n_samples_new,).
        """
        y_pred = self.regressor.predict(X_new)
        interval_width = self.quantile
        return y_pred, y_pred - interval_width, y_pred + interval_width

    
    
