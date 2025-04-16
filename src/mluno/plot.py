import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(X, y, regressor, conformal=False, title=''):
    """
    Plot the observed data points, the regression line, and optionally the conformal prediction intervals.

    This function creates a scatter plot of the observed data points and overlays the regression line
    predicted by the provided regressor. If conformal prediction is enabled, it also displays the 
    prediction intervals as a shaded area.

    Parameters
    ----------
    X : ndarray
        Independent variable data points, should be a 2D array for the regressor input.
    y : ndarray
        Dependent variable data points, should be a 1D array.
    regressor : object
        A fitted regression model object that must have a 'predict' method. If conformal prediction 
        is enabled, the 'predict' method should return a tuple of predicted values, lower and upper 
        prediction intervals.
    conformal : bool, optional
        Indicates whether to plot conformal prediction intervals. Default is False.
    title : str, optional
        Title for the plot. Default is an empty string.

    Returns
    -------
    fig : Figure
        The figure object that contains the plot. This object can be used to save the plot,
        modify its properties, or display it using `fig.show()`.
    ax : Axes
        The axes object of the plot. This object can be used to further customize the plot,
        such as adding labels, titles, or other plotting elements.

    """

    X_values = np.linspace(X.min(), X.max(), 500)[:, None]

    if conformal:
        y_pred, y_pred_lower, y_pred_upper = regressor.predict(X_values)
    else:
        y_pred = regressor.predict(X_values)
    
    
    #init plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, label="Data", color='blue')
    ax.plot(X_values, y_pred, label="Predicted", color='red')

    if conformal:
        ax.fill_between(
            X_values.flatten(), 
            y_pred_lower, 
            y_pred_upper, 
            color='orange',
            alpha=0.2)
        
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.grid(color='gray', linestyle='--', linewidth=0.25, alpha=0.5)

    return fig, ax