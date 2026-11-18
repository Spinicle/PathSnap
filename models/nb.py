"""Gaussian Naive Bayes baseline (Owner: A). Same protocol as models/logreg.py.
Run: python -m models.nb
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.logreg import run_baseline

if __name__ == "__main__":
    run_baseline("nb", Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]))
