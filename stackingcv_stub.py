"""Stub for stacking with CV."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from stackingcv import StackingClassifierCV
from sklearn.datasets import make_classification

X, y = make_classification()

base_clfs = [RandomForestClassifier(), LogisticRegression()]
meta_clf = LogisticRegression()

stacking = StackingClassifierCV(
    base_estimators=base_clfs, meta_estimator=meta_clf, cv=3)

stacking.fit(X, y)
stacking.predict(X)
stacking.get_params()
