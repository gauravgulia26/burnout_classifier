"""
Registry of all supported machine learning algorithms.

The registry maps a unique model name to its corresponding estimator class.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from typing import Type, List

MODEL_REGISTRY: dict[str, Type[BaseEstimator]] = {
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "logistic_regression": LogisticRegression,
}


def list_all_registered_models() -> List:
    """Get a list of all availaible models"""
    return list(MODEL_REGISTRY.keys())
