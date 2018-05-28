"""Provide classes for stacking classifiers and regressors."""

from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import check_cv
from base import BaseStacking
import numpy as np


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


class StackingClassifierCV(BaseStacking, ClassifierMixin):
    """Stacking classifier using cross validation.

    Parameters
    ----------
    base_estimators:

    meta_estimator:

    """

    def __init__(self, base_estimators, meta_estimator,
                 cv=3, use_orig_features=False, probas=True):
        """Initialise StackingClassifierCV."""
        super(StackingClassifierCV, self).__init__(
            base_estimators=base_estimators,
            meta_estimator=meta_estimator,
            use_orig_features=use_orig_features)

        self.cv = cv
        self.probas = probas

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
        meta_features_list = []
        cv = check_cv(self.cv, y=y, classifier=True)

        for clf in self.base_estimators:
            # feels kind of clumsy, but we want the meta features in the
            # original ordering
            if self.probas:
                meta_features = np.zeros((y.shape[0], 2))

                pred = [(test, clf.fit(X[train], y[train]).predict_proba(
                    X[test])) for train, test in cv.split(X, y)]

            else:
                meta_features = np.zeros((y.shape[0],))

                pred = [(test, clf.fit(X[train], y[train]).predict(
                    X[test])) for train, test in cv.split(X, y)]

            for index, y_pred in pred:
                meta_features[index] = y_pred

            meta_features_list.append(meta_features)

        all_meta_features = np.column_stack(meta_features_list)

        if self.use_orig_features:
            all_meta_features = np.hstack((X, all_meta_features))

        # train base estimators with whole training data set
        for clf in self.base_estimators:
            clf.fit(X, y)

        # train meta estimators
        self.meta_estimator.fit(all_meta_features, y)

        return self

    # def predict(self, X):
    #     """Predict labels for X.
    #
    #     Parameters
    #     ----------
    #     X: {array-like, sparse matrix}, shape(n_samples, n_features)
    #         Training vectors, where n_samples is the number samples and
    #         n_features is the number of features.
    #
    #     Returns
    #     -------
    #     y: array of shape n_samples
    #         Predicted class labels.
    #
    #     """
    #     meta_features = self._get_meta_features(X)
    #     return self.meta_estimator.predict(meta_features)

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
        meta_features = self._get_meta_features(X)
        return self.meta_estimator.predict_proba(meta_features)

    def _get_meta_features(self, X):
        if self.probas:
            y_pred = [clf.predict_proba(X) for clf in self.base_estimators]
        else:
            y_pred = [clf.predict(X) for clf in self.base_estimators]

        meta_features = np.column_stack(y_pred)

        if self.use_orig_features:
            meta_features = np.hstack((X, meta_features))

        return meta_features
