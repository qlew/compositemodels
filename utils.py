"""Provide utils for composite models."""


def name_estimators(estimators):
    """Return names and estimators"""
    names = [type(estimator).__name__.lower() for estimator in estimators]
    return zip(names, estimators)
