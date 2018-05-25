"""Provide classes for stacking classifiers and regressors."""

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from abc import ABC, abstractmethod
import numpy as np
from utils import name_estimators


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
        """Fit the meta classifier to the base classifiers.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape(n_samples, n_features)
            Training vectors, where n_samples is the number samples and
            n_features is the number of features.

        y: array-like, shape(n_samples,)
            Labels for classification.

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
        """Predict labels for X.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape(n_samples, n_features)
            Training vectors, where n_samples is the number samples and
            n_features is the number of features.

        Returns
        -------
        y: array of shape n_samples
            Predicted class labels.

        """
        X_meta = self._get_meta_features(X)
        return self.meta_estimator.predict(X_meta)


class StackingClassifier(BaseStacking, ClassifierMixin):
    """Simple stacking classifier.

    Parameters
    ----------
    base_classifiers: array-like of shape n_classifiers
        A list of base classifiers. They are expected to be instances of
        `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin`.

    meta_classifier: object
        The second-level classifier to be fitted to the predictions of the
        base classifiers. Expected to be an instance of
        `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin`.

    probas: boolean, optional (default True)
        If True the class probabilities as returned from `predict_proba`
        of the base classifiers will be used as meta features for training
        of the meta classifier. Otherwise the predicted classes as returned
        from `predict` of the base classifiers will be used.

    use_orig_features: boolean, optional (default False)
        If True the orginal features X along with prediction from the base
        classifiers will be used as features for training the meta classifier.

    """

    def __init__(self, base_estimators, meta_estimator,
                 use_orig_features=False, probas=True):
        """Initialise StackingClassifier with base and meta classifiers."""
        super(StackingClassifier, self).__init__(
            base_estimators=base_estimators,
            meta_estimator=meta_estimator,
            use_orig_features=use_orig_features)

        self.probas = probas

    def _get_meta_features(self, X):
        if self.probas:
            y_pred = [clf.predict_proba(X) for clf in self.base_estimators]
        else:
            y_pred = [clf.predict(X) for clf in self.base_estimators]

        meta_features = np.column_stack(y_pred)

        if self.use_orig_features:
            meta_features = np.hstack((X, meta_features))

        return meta_features

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape(n_samples, n_features)
            Training vectors, where n_samples is the number samples and
            n_features is the number of features.

        Returns
        -------
        y: array of shape n_samples
            Predicted class probabilities.

        """
        X_meta = self._get_meta_features(X)
        return self.meta_estimator.predict_proba(X_meta)


class StackingRegressor(BaseStacking, RegressorMixin):
    """Simple stacking regressor.

    Parameters
    ----------
    base_regressors: array-like of shape n_classifiers
        A list of base regressors. They are expected to be instances of
        `sklearn.base.BaseEstimator` and `sklearn.base.RegressorMixin`.

    meta_regressor: object
        The second-level regressor to be fitted to the predictions of the
        base regressors. They are expected to be instances of
        `sklearn.base.BaseEstimator` and `sklearn.base.RegressorMixin`.

    use_orig_features: boolean, optional (default False)
        If True the orginal features X along with prediction from the base
        regressors will be used as features for training the meta regressor.

    """

    def __init__(self, base_estimators, meta_estimator,
                 use_orig_features=False):
        """Initialise StackingRegressor with base and meta regressors."""
        super(StackingRegressor, self).__init__(
            base_estimators=base_estimators,
            meta_estimator=meta_estimator,
            use_orig_features=use_orig_features)

    def _get_meta_features(self, X):
        y_pred = [estimator.predict(X) for estimator in self.base_estimators]

        meta_features = np.column_stack(y_pred)

        if self.use_orig_features:
            meta_features = np.hstack((X, meta_features))

        return meta_features
