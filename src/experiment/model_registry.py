"""
Registry of all supported machine learning algorithms.

The registry maps a unique model name to its corresponding estimator class.
"""

from typing import List, Type

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

MODEL_REGISTRY: dict[str, Type[BaseEstimator]] = {
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "logistic_regression": LogisticRegression,
    "xgboost": XGBClassifier,
}


def list_all_registered_models() -> List:
    """Get a list of all availaible models"""
    return list(MODEL_REGISTRY.keys())
