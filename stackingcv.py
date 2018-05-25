"""Stacking estimators using cv."""

from base import BaseStacking
from sklearn.base import ClassifierMixin
from sklearn.model_selection import check_cv
import numpy as np


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
        if self.probas:
            y_pred = [clf.predict_proba(X) for clf in self.base_estimators]
        else:
            y_pred = [clf.predict(X) for clf in self.base_estimators]

        meta_features = np.column_stack(y_pred)

        if self.use_orig_features:
            meta_features = np.hstack((X, meta_features))

        return self.meta_estimator.predict(meta_features)

    def _get_meta_features(self, X):
        pass
