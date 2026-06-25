"""
Train pIC50 regression model on TLR2 ChEMBL data.

Models: Random Forest, XGBoost, Gradient Boosting
Validation: 80/20 split + 5-fold CV
Saves best model to model/tlr2_model.pkl
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEATURES_CSV = "data/chembl_tlr2_features.csv"
MODEL_OUT = "model/tlr2_model.pkl"
META_OUT = "model/model_meta.json"

FP_BITS = 2048
PHYS_COLS = [
    "mw", "logp", "hbd", "hba", "tpsa", "rot_bonds",
    "aromatic_rings", "heavy_atoms", "rings", "stereo_centers",
    "acyl_chain_count", "longest_chain"
]


def load_features(path=FEATURES_CSV):
    df = pd.read_csv(path)
    fp_cols = [f"fp_{i}" for i in range(FP_BITS)]
    X_fp = df[fp_cols].values.astype(np.float32)
    X_phys = df[PHYS_COLS].fillna(0).values.astype(np.float32)
    X = np.hstack([X_phys, X_fp])
    y = df["pIC50"].values.astype(np.float32)
    return X, y, df


def eval_model(name, model, X_train, X_test, y_train, y_test, X_all, y_all):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv = cross_val_score(model, X_all, y_all, cv=5, scoring="r2")
    print(f"\n{name}")
    print(f"  Test R2:   {r2:.4f}")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  CV R2:     {cv.mean():.4f} ± {cv.std():.4f}")
    return r2, rmse, cv.mean()


def train():
    print("Loading features...")
    X, y, df = load_features()
    print(f"Dataset: {X.shape[0]} compounds, {X.shape[1]} features")
    print(f"pIC50 range: {y.min():.2f} - {y.max():.2f}, mean {y.mean():.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1
        )

    best_name, best_r2, best_model = None, -np.inf, None
    results = {}

    for name, model in models.items():
        r2, rmse, cv_r2 = eval_model(
            name, model, X_train, X_test, y_train, y_test, X, y
        )
        results[name] = {"r2": float(r2), "rmse": float(rmse), "cv_r2": float(cv_r2)}
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, model

    print(f"\nBest model: {best_name} (R2={best_r2:.4f})")

    joblib.dump(best_model, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

    meta = {
        "best_model": best_name,
        "test_r2": best_r2,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(X)),
        "feature_order": PHYS_COLS + [f"fp_{i}" for i in range(FP_BITS)],
        "fp_bits": FP_BITS,
        "fp_radius": 2,
        "phys_cols": PHYS_COLS,
        "results": results,
        "pic50_mean": float(y.mean()),
        "pic50_std": float(y.std()),
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {META_OUT}")
    return best_model, meta


if __name__ == "__main__":
    train()
