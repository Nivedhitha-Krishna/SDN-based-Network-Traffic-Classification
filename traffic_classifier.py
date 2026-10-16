"""
SDN-style network traffic classification with classic ML
Course: 24CYS304 Computer Networks - Group 1

Usage:
  python traffic_classifier.py --demo
  python traffic_classifier.py --data "TimeBasedFeatures-Dataset-15s-NO-VPN.arff" --classes BROWSING VOIP STREAMING
  python traffic_classifier.py --data file.csv --label-col class1 --classes BROWSING VOIP STREAMING CHAT
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ---------------------------------------------------------------- STEP 1: DATA
def synthetic_flows(n_per_class=600, seed=42):
    """Fake flow records ONLY to test the pipeline (not for the report)."""
    rng = np.random.default_rng(seed)
    # (duration, mean_iat, flow_pkts_per_s, flow_bytes_per_s, mean_active, mean_idle)
    profiles = {
        "VIDEO":    (60, 0.02, 400, 4.0e5, 8.0, 1.0),
        "BROWSING": (15, 0.30, 40,  2.0e4, 1.5, 4.0),
        "VOIP":     (40, 0.02, 50,  8.0e3, 6.0, 0.5),
    }
    rows = []
    for label, p in profiles.items():
        d = rng.lognormal(np.log(p[0]), 0.7, n_per_class)
        iat = rng.lognormal(np.log(p[1]), 0.8, n_per_class)
        pps = rng.lognormal(np.log(p[2]), 0.8, n_per_class)
        bps = rng.lognormal(np.log(p[3]), 0.9, n_per_class)
        act = rng.lognormal(np.log(p[4]), 0.8, n_per_class)
        idle = rng.lognormal(np.log(p[5]), 0.8, n_per_class)
        rows.append(pd.DataFrame({
            "duration": d, "mean_flowiat": iat, "flowPktsPerSecond": pps,
            "flowBytesPerSecond": bps, "mean_active": act, "mean_idle": idle,
            "class1": label}))
    return pd.concat(rows, ignore_index=True)


def load_iscx(path, label_col, classes):
    if path.lower().endswith(".arff"):
        from scipy.io import arff
        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        for c in df.columns:                       # nominal/string columns come as bytes
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if label_col not in df.columns:
        raise SystemExit(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()
    print("Classes available in file:\n", df[label_col].value_counts(), "\n")
    return df


# --------------------------------------------------------------- STEP 2: CLEAN
def prepare(df, label_col, classes):
    df = df[df[label_col].isin([c.upper() for c in classes])].copy()   # keep chosen 3-4 classes
    y = df.pop(label_col)
    X = df.select_dtypes(include=[np.number])                         # numeric flow features only
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)                                      # drop broken rows
    X, y = X[keep], y[keep]
    X = X.loc[:, X.nunique() > 1]                                     # drop constant columns
    return X, y


# -------------------------------------------------------------- STEP 3: TRAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--list", action="store_true", help="only list the classes in the file and exit")
    ap.add_argument("--data", "--csv", dest="data", help=".arff or .csv file")
    ap.add_argument("--label-col", default="class1")
    ap.add_argument("--classes", nargs="+", default=["VIDEO", "BROWSING", "VOIP"])
    a = ap.parse_args()

    if a.demo:
        df, label_col = synthetic_flows(), "class1"
        print("DEMO MODE: synthetic data, results are NOT for the report\n")
    elif a.data:
        label_col = a.label_col
        df = load_iscx(a.data, label_col, a.classes)
        if a.list:
            return
    else:
        raise SystemExit("Give --demo or --data <file>")

    X, y = prepare(df, label_col, a.classes)
    print(f"Using {X.shape[0]} flows, {X.shape[1]} features, classes: {sorted(y.unique())}")
    print(y.value_counts(), "\n")

    # stratified split keeps class proportions equal in train and test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        results[name] = (m, pred)
        print(f"=== {name} ===  accuracy = {accuracy_score(y_te, pred):.3f}")
        print(classification_report(y_te, pred, digits=3))

    # ------------------------------------------ 5-FOLD CROSS-VALIDATION (all flows)
    print("=== 5-fold stratified cross-validation (whole dataset) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, m in models.items():
        sc = cross_val_score(m, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"{name}: mean accuracy = {sc.mean():.4f} +/- {sc.std():.4f}   (folds: {np.round(sc, 4)})")
    print()

    # ------------------------------------------------ STEP 4: EVIDENCE FOR REPORT
    rf, rf_pred = results["RandomForest"]
    labels = sorted(y.unique())
    for name, fname in [("RandomForest", "confusion_matrix.png"),
                        ("SVM (RBF)", "confusion_matrix_svm.png")]:
        _, pred = results[name]
        ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=labels),
                               display_labels=labels).plot(cmap="Blues")
        plt.title(f"{name} - confusion matrix")
        plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
    imp.plot.barh(title="Top features (Random Forest)")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()
    print("Saved confusion_matrix.png, confusion_matrix_svm.png and feature_importance.png")

    # ----------------------------------------- STEP 5: SAVE MODEL + SAMPLE PREDICTIONS
    joblib.dump({"model": rf, "features": list(X.columns), "test_index": list(X_te.index)}, "flow_classifier.joblib")
    print("Saved trained model to flow_classifier.joblib\n")

    print("=== Example: classifying individual flows ===")
    sample = X_te.sample(5, random_state=1)
    probs = rf.predict_proba(sample)
    for (idx, row), pr in zip(sample.iterrows(), probs):
        pred = rf.classes_[pr.argmax()]
        print(f"flow #{idx}: duration={row.get('duration', float('nan')):.2f}  "
              f"bytes/s={row.get('flowBytesPerSecond', float('nan')):.0f}  "
              f"-> predicted {pred} ({pr.max()*100:.1f}% confident) | actual {y_te[idx]}")


if __name__ == "__main__":
    main()
