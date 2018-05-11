from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from abc import ABC, abstractmethod
import numpy as np


class BaseStacking(ABC, BaseEstimator):

    @abstractmethod
    def __init__(self, base_estimators, meta_estimator,
                 use_orig_features=False):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.use_orig_features = use_orig_features

    # @abstractmethod
    def _get_meta_features(self, X):
        """Return meta features on which the meta estimator will be trained."""

        y_pred = [regr.predict(X) for regr in self.base_estimators]

        meta_features = np.column_stack(y_pred)

        if self.use_orig_features:
            meta_features = np.hstack((X, meta_features))

        return meta_features

    def fit(self, X, y):
        for est in self.base_estimators:
            est.fit(X, y)

        X_meta = self._get_meta_features(X)

        self.meta_estimator.fit(X_meta, y)

    def predict(self, X):
        X_meta = self._get_meta_features(X)
        return self.meta_estimator.predict(X_meta)


class StackingClassifier(BaseStacking, ClassifierMixin):

    def __init__(self, base_classifiers, meta_classifier,
                 use_orig_features=False, probas=True):
        super(StackingClassifier, self).__init__(
            base_estimators=base_classifiers,
            meta_estimator=meta_classifier,
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
        X_meta = self._get_meta_features(X)
        return self.meta_estimator.predict_proba(X_meta)


class StackingRegressor(BaseStacking, RegressorMixin):

    def __init__(self, base_regressors, meta_regressor,
                 use_orig_features=False):

        super(StackingRegressor, self).__init__(
            base_estimators=base_regressors,
            meta_estimator=meta_regressor,
            use_orig_features=use_orig_features)

    def _get_meta_features(self, X):
        y_pred = [regr.predict(X) for regr in self.base_estimators]

        meta_features = np.column_stack(y_pred)

        if self.use_orig_features:
            meta_features = np.hstack((X, meta_features))

        return meta_features


def _get_features(self, X):
    if self.probas:
        y_pred = [clf.predict_proba(X) for clf in self.base_estimators]
    else:
        y_pred = [clf.predict(X) for clf in self.base_estimators]

    meta_features = np.column_stack(y_pred)

    if self.use_orig_features:
        meta_features = np.hstack((X, meta_features))

    return meta_features
