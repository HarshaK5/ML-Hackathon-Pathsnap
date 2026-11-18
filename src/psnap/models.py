"""Model zoo (10 models from 8 algorithm families) + dataset loading for model experiments."""
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from .config import ART, SEED
from .features import FEATURES


def load_split(split):
    """Label-valid rows only. Adds gid = contiguous id per (trip, GPS point)."""
    pairs = pd.read_parquet(ART / "data" / f"{split}_pairs.parquet")
    pairs = pairs[pairs.valid].reset_index(drop=True)
    gid = pairs.groupby(["trip", "pt"], sort=False).ngroup().values
    pairs["gid"] = gid
    return pairs


def xy(df):
    return df[FEATURES].values.astype(np.float32), df["y"].values.astype(np.int8)


class SubsampledKNN:
    """KNN is O(n_train) per query, so the reference set is a random subsample of the training rows."""
    def __init__(self, n=60000, k=25, seed=SEED):
        self.n, self.seed = n, seed
        self.m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="brute", n_jobs=-1))

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(X), min(self.n, len(X)), replace=False)
        self.m.fit(X[idx], y[idx]); return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def zoo():
    """name -> (family, config dict (for the report), factory)"""
    return {
        "LogReg": ("Linear / probabilistic", dict(C=1.0, solver="lbfgs", max_iter=300, scaling="standard"),
                   lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))),
        "GaussianNB": ("Linear / probabilistic", dict(var_smoothing=1e-9),
                       lambda: GaussianNB()),
        "KNN": ("Instance-based", dict(k=25, weights="distance", reference_rows=60000, scaling="standard"),
                lambda: SubsampledKNN()),
        "DecisionTree": ("Tree-based", dict(max_depth=14, min_samples_leaf=20),
                         lambda: DecisionTreeClassifier(max_depth=14, min_samples_leaf=20, random_state=SEED)),
        "RandomForest": ("Ensemble (bagging)", dict(n_estimators=100, max_depth=16, min_samples_leaf=10, max_samples=0.3),
                         lambda: RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=10, max_samples=0.3,
                                                        n_jobs=-1, random_state=SEED)),
        "ExtraTrees": ("Ensemble (bagging)", dict(n_estimators=100, max_depth=18, min_samples_leaf=10, max_samples=0.3),
                       lambda: ExtraTreesClassifier(n_estimators=100, max_depth=18, min_samples_leaf=10, max_samples=0.3,
                                                    bootstrap=True, n_jobs=-1, random_state=SEED)),
        "HistGB": ("Boosting", dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31),
                   lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)),
        "LightGBM": ("Boosting", dict(n_estimators=300, learning_rate=0.1, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8),
                     lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, subsample=0.8, subsample_freq=1,
                                                colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)),
        "XGBoost": ("Boosting", dict(n_estimators=300, learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8),
                    lambda: xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                                              tree_method="hist", random_state=SEED, n_jobs=-1, verbosity=0)),
        "MLP": ("Neural network", dict(hidden=(64, 32), activation="relu", alpha=1e-4, batch_size=1024, early_stopping=True, max_iter=60, scaling="standard"),
                lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                                                                      early_stopping=True, n_iter_no_change=6, max_iter=60, random_state=SEED))),
    }
