import numpy as np

def rmse(y_true, y_pred):
    """
    Compute the root mean square error (RMSE) between true and predicted values.

    The RMSE is a measure of the differences between values predicted by a model
    and the values actually observed from the environment that is being modeled.

    Parameters
    ----------
    y_true : ndarray
        True target values.
    y_pred : ndarray
        Predicted target values by the model.

    Returns
    -------
    float
        The RMSE between the true and predicted target values.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """
    Compute the mean absolute error (MAE) between true and predicted values.

    The MAE is a measure of errors between paired observations expressing the same
    phenomenon.

    Parameters
    ----------
    y_true : ndarray
        True target values.
    y_pred : ndarray
        Predicted target values by the model.

    Returns
    -------
    float
        The MAE between the true and predicted target values.
    """
    return np.mean(np.abs(y_true - y_pred))

def coverage(y_true, y_pred_lower, y_pred_upper):
    """
    Calculate the coverage of the prediction intervals.

    Coverage is defined as the proportion of times the true value falls within
    the predicted interval.

    Parameters
    ----------
    y_true : ndarray
        True target values.
    y_pred_lower : ndarray
        Lower bounds of the prediction intervals.
    y_pred_upper : ndarray
        Upper bounds of the prediction intervals.

    Returns
    -------
    float
        The proportion of true values that lie within the prediction intervals.
    """
    return np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))

def sharpness(y_pred_lower, y_pred_upper):
    """
    Calculate the sharpness of the prediction intervals.

    Sharpness is the width of the prediction intervals and can be interpreted as a
    measure of the precision of the predictions. Lower values indicate more precise
    predictions.

    Parameters
    ----------
    y_pred_lower : ndarray
        Lower bounds of the prediction intervals.
    y_pred_upper : ndarray
        Upper bounds of the prediction intervals.

    Returns
    -------
    float
        The average width of the prediction intervals.
    """
    return np.mean(y_pred_upper - y_pred_lower)
