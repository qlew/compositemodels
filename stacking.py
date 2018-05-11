"""Simple models to stack estimators"""

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import is_classifier
import numpy as np


class StackingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Simple stacking classifier

    Parameters
    ----------
    base_classifiers: array-like of shape n_classifiers
        A list of base classifiers.

    meta_classifier: object
        The second-level classifier to be fitted to the predictions of the
        base classifiers.


    """

    def __init__(self, base_classifiers, meta_classifier, probas=True,
                 use_orig_features=False):
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier
        self._check_classifiers()
        self.probas = probas
        self.use_orig_features = use_orig_features

    def _check_classifiers(self):
        for clf in self.base_classifiers:
            if not is_classifier(clf):
                raise TypeError(f"Base classifier {clf} is not a classifier.")

        if not is_classifier(self.meta_classifier):
            raise TypeError(
                f"Meta classifier {self.meta_classifier} is not a classifier.")

    def fit(self, X, y):
        """Fit the meta classifier to the base classifiers.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape(n_samples, n_features)
            Training vectors, where n_samples is the number samples and
            n_features is the number of features.

        y: array-like, shape(n_samples,)
            Labels for classification.

        probas: boolean, optional (default True)
            If True the class probabilities as returned from `predict_proba`
            of the base classifiers will be used as meta features for training
            of the meta classifier. Otherwise the predicted classes as returned
            from `predict` of the base classifiers will be used.

        use_orig_features: boolean, optional (default False)
            If True the orginal features X along with the meta features will be
            used as features for training the meta classifier.

        Returns
        -------
        self: object
            Returns self
        """

        for clf in self.base_classifiers:
            clf.fit(X, y)

        X_meta = self._get_meta_features(X)

        if not self.use_orig_features:
            self.meta_classifier.fit(X_meta, y)
        else:
            self.meta_classifier.fit(np.hstack((X, X_meta)), y)

        return self

    def _get_meta_features(self, X):
        if self.probas:
            y_pred = [clf.predict_proba(X) for clf in self.base_classifiers]
        else:
            y_pred = [clf.predict(X) for clf in self.base_classifiers]

        meta_features = np.concatenate(y_pred, axis=1)

        if self.use_orig_features:
            meta_features = np.hstack((X, meta_features))

        return meta_features

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
        return self.meta_classifier.predict(X_meta)

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
        return self.meta_classifier.predict_proba(X_meta)
