"""Regression model for censored data."""
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import is_regressor, is_classifier
import numpy as np
from utils import name_estimators


class CensoredRegression(BaseEstimator, RegressorMixin, TransformerMixin):
    """Censored Regression.

    Regression model for the case when the target variable is only observed
    when above or below a certain threshold.

    Parameters
    ----------
    classifier: object
        Classifier that will be fitted to the derived class labels; where the
        positive class indicates that the regression target has been observed.
        Expected to be an instance of `sklearn.base.BaseEstimator` and
        `sklearn.base.ClassifierMixin`.

    regressor: object
        Regressor that will be trained on the samples of the positive class.
        Expected to be an instance of `sklearn.base.BaseEstimator` and
        `sklearn.base.RegressorMixin`.

    censored_value: float (default: 0.0)
        Threshold below or above which target is not observed.

    censored_how: string (default: 'left')
        Must be one of either 'left' or 'right'. Determines whether the data
        is left or right censored.

    Attributes
    ----------
    named_estimators: dict
        Dictionary with names for the estimators (classifier and regressor)
        as keys and the corresponding instances of classifier and regressor
        as values.

    """

    def __init__(self, classifier, regressor, censored_value=0.0,
                 censored_how='left'):
        """Initialise CensoredRegression with classifier, regressor.

        Optional parameters censored_value, censored_how can be set as well.
        """
        self._check_estimators(classifier, regressor)
        self.classifier = classifier
        self.regressor = regressor
        self.named_estimators = {key: value for key, value in name_estimators(
            [self.classifier, self.regressor])}

        self.censored_value = censored_value
        if censored_how in ['left', 'right']:
            self.censored_how = censored_how
        else:
            raise ValueError(
                f"{censored_how} must be either 'left' or 'right'.")

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params: mapping of string to any
            Parameter names mapped to their values.

        """
        if not deep:
            return super(CensoredRegression, self).get_params(deep=False)
        else:
            params = self.named_estimators.copy()
            for name, estimator in self.named_estimators.items():
                for key, value in estimator.get_params(deep=False).items():
                    params[f"{name}__{key}"] = value

            for key, value in super(
                    CensoredRegression, self).get_params(deep=False).items():
                params[key] = value

            return params

    def _get_named_estimators(self):
        return {type(estimator).__name__.lower(): estimator
                for estimator in [self.classifier, self.regressor]}

    def _check_estimators(self, classifier, regressor):

        for attribute in ['fit', 'predict_proba']:
            if not hasattr(classifier, attribute):
                raise AttributeError(
                    f"{classifier} has no attribute {attribute}")

        for attribute in ['fit', 'predict']:
            if not hasattr(regressor, attribute):
                raise AttributeError(
                    f"{regressor} has no attribute {attribute}")

        for estimator in [classifier, regressor]:
            if not isinstance(estimator, BaseEstimator):
                raise TypeError(f"{estimator} is not an estimator.")

        if not is_regressor(regressor):
            raise TypeError(f"{regressor} is not an regressor.")

        if not is_classifier(classifier):
            raise TypeError(f"{classifier} is not a classifier.")

    def _get_regression_data(self, X, y):
        y_labels = self._get_classification_labels(y)

        X_r = X[y_labels]
        y_r = y[y_labels]

        return X_r, y_r

    def _get_classification_labels(self, y):
        if self.censored_how == 'left':
            y_labels = y > self.censored_value
        else:
            y_labels = y < self.censored_value

        return y_labels

    def predict(self, X):
        """Predict censored regression value for X.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape(n_samples, n_features)
            Training vectors, where n_samples is the number samples and
            n_features is the number of features.

        Returns
        -------
        y: array of shape n_samples
            Predicted regression values.

        """
        y_pred_prob = self.classifier.predict_proba(X)
        y_pred_regr = self.regressor.predict(X)

        return np.multiply(y_pred_prob[:, 1], y_pred_regr)

    def fit(self, X, y):
        """Fit the censored regression model to the given training data.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape(n_samples, n_features)
            Training vectors, where n_samples is the number samples and
            n_features is the number of features.

        y: array-like, shape(n_samples,)
            Target values for regression.

        Returns
        -------
        self: object
            Returns self.

        """
        y_labels = self._get_classification_labels(y)
        self.classifier.fit(X, y_labels)

        X_r, y_r = self._get_regression_data(X, y)
        self.regressor.fit(X_r, y_r)

        return self
