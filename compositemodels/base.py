"""Provide base class for stacking classifiers and regressors."""

from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod


def name_estimators(estimators):
    """Return names and estimators.

    Parameters
    ----------
    estimators: Iterable of estimators of length n_estimators

    Returns
    -------
    r: zip object of names and estimators

    """
    names = [type(estimator).__name__.lower() for estimator in estimators]
    return zip(names, estimators)


class BaseStacking(ABC, BaseEstimator):
    """Base class for stacking.

    This class should not be instantiated, use subclasses instead.
    """

    @abstractmethod
    def __init__(self, base_estimators, meta_estimator,
                 use_orig_features=False):
        """Initialise BaseStacking with base and meta estimators."""
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.named_based_estimators = {name: estimator for name, estimator
                                       in name_estimators(base_estimators)}
        self.named_meta_estimator = {''.join(("meta-", name)): estimator
                                     for name, estimator
                                     in name_estimators([meta_estimator])}
        self.use_orig_features = use_orig_features

    @abstractmethod
    def _get_meta_features(self, X):
        """Return meta features on which the meta estimator will be trained."""

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
            return super(BaseStacking, self).get_params(deep=False)
        else:
            params = self.named_based_estimators.copy()
            for name, estimator in self.named_based_estimators.items():
                for key, value in estimator.get_params(deep=False).items():
                    params[f"{name}__{key}"] = value

            params.update(self.named_meta_estimator.copy())
            for name, estimator in self.named_meta_estimator.items():
                for key, value in estimator.get_params(deep=False).items():
                    params[f"{name}__{key}"] = value

            for key, value in super(
                    BaseStacking, self).get_params(deep=False).items():
                params[key] = value

            return params

    def fit(self, X, y):
        """Fit the stacking estimator to the training data.

        First fit the base estimators to the training data, then fit the meta
        estimator to the predictions of the base estimators and the targets.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape(n_samples, n_features)
            Training vectors, where n_samples is the number samples and
            n_features is the number of features.

        y: array-like, shape(n_samples, [n_classes])
            Targets for the estimation. Continuous in case of regression,
            binary or multi-class in case of classification.

        Returns
        -------
        self: object
            Returns self

        """
        for est in self.base_estimators:
            est.fit(X, y)

        X_meta = self._get_meta_features(X)

        self.meta_estimator.fit(X_meta, y)

        return self

    def predict(self, X):
        """Predict targets for X.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape(n_samples, n_features)
            Training vectors, where n_samples is the number samples and
            n_features is the number of features.

        Returns
        -------
        y: array of shape n_samples
            Predicted targets.

        """
        X_meta = self._get_meta_features(X)
        return self.meta_estimator.predict(X_meta)
