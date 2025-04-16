import numpy as np

class KNNRegressor:
    """
    A K-Nearest Neighbors regressor class.

    This class implements a basic version of the k-nearest neighbors algorithm for regression. 
    Given a set of input features and corresponding target values, it predicts the output 
    for new input features based on the mean target value of the k nearest neighbors in 
    the training set.

    Parameters
    ----------
    k : int, optional
        The number of nearest neighbors to consider for making predictions. The default 
        is 5.

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input features from the training data. Set after calling the `fit` method.
    y : ndarray of shape (n_samples,)
        The target values from the training data. Set after calling the `fit` method.

   
    
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=1, n_informative=1, noise=10, random_state=1)
    >>> knn = KNNRegressor(k=3)
    >>> knn.fit(X, y)
    >>> X_new = np.array([[0], [1]])
    >>> print(knn.predict(X_new))
    """
    
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        """
        Fit the model using X as input data and y as target values.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training data, where n_samples is the number of samples and
            n_features is the number of features.
        y : ndarray of shape (n_samples,)
            The target values (continuous outcomes) for the training samples.
        """
        
        self.X = X
        self.y = y

    def predict(self, X_new):
        """
        Predict the target for the provided data.
        
        Parameters
        ----------
        X_new : ndarray of shape (n_samples, n_features)
            New input data for which to make predictions.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted target values for the input data.
        """
        predicted_labels = [self._predict(x) for x in X_new]
        return np.array(predicted_labels)

    def _predict(self, x_new):
        """
        A helper method to predict the target for a single sample based on
        the k-nearest neighbors in the training data.
        
        Parameters
        ----------
        x_new : ndarray of shape (n_features,)
            A single sample of input data.
        
        Returns
        -------
        float
            The predicted target value for the single sample.
        """
        distances = [np.linalg.norm(x_new - x) for x in self.X]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_y = self.y[k_indices]
        return np.mean(k_nearest_y)

class LinearRegressor:
    """
    A class used to represent a Simple Linear Regressor.


    Attributes
    ----------
    weights : ndarray
        The weights of the linear regression model. Here, the weights are represented
        by the beta vector which for univariate regression is a 1D vector of length two,
        beta = [beta_0, beta_1] where beta_0 is the slope and beta_1 is the intercept.


    """

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """
        Trains the linear regression model using the given training data.
        
        Parameters
        ----------
        X : ndarray
            The training data, which is a 2D array of shape (n_samples, 1) where
            each row is a sample and each column is a feature.
        y : ndarray
            The target values, which is a 1D array of shape (n_samples, ).

        Returns
        -------
        self : returns an instance of self.
        """
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        """
        Makes predictions for input data.

        Parameters
        ----------
        X : ndarray
            Input data, a 2D array of shape (n_samples, 1), with which to make predictions.

        Returns
        -------
        y_pred : ndarray
            The predicted target values as a 1D array with the same length as X.
        """
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))  
        y_pred = X_b @ self.weights
        return y_pred


