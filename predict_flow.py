"""
Classify flows with the trained model (run traffic_classifier.py first).

  python predict_flow.py --data "TimeBasedFeatures-Dataset-15s.arff" --rows 0 5 100
  python predict_flow.py --data "TimeBasedFeatures-Dataset-15s.arff" --pick VOIP 3
  python predict_flow.py --data "TimeBasedFeatures-Dataset-15s.arff" --pick-each 2
"""
import argparse
import joblib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help=".arff or .csv file containing flows")
ap.add_argument("--rows", nargs="+", type=int, default=[], help="row numbers to classify")
ap.add_argument("--pick", nargs=2, metavar=("CLASS", "N"), help="classify N random flows of CLASS")
ap.add_argument("--pick-each", type=int, metavar="N", help="classify N random flows of each trained class")
ap.add_argument("--label-col", default="class1")
a = ap.parse_args()

bundle = joblib.load("flow_classifier.joblib")
model, features = bundle["model"], bundle["features"]
test_index = set(bundle.get("test_index", []))

if a.data.lower().endswith(".arff"):
    from scipy.io import arff
    df = pd.DataFrame(arff.loadarff(a.data)[0])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
else:
    df = pd.read_csv(a.data)
df.columns = [c.strip() for c in df.columns]

lab = a.label_col
rows = list(a.rows)
pool = df[df.index.isin(test_index)] if test_index else df   # only flows the model never trained on
if not test_index:
    print("WARNING: retrain with the latest traffic_classifier.py so picks come from unseen test flows only")
if a.pick:
    cls, n = a.pick[0].upper(), int(a.pick[1])
    rows += pool[pool[lab] == cls].sample(n, random_state=7).index.tolist()
if a.pick_each:
    for cls in model.classes_:
        rows += pool[pool[lab] == cls].sample(a.pick_each, random_state=7).index.tolist()

if test_index:
    for r in a.rows:
        if r not in test_index:
            print(f"note: row {r} may have been used for training, so a correct answer there proves little")
for r in rows:
    flow = df.loc[[r], features].replace([np.inf, -np.inf], np.nan)
    if flow.isna().any(axis=None):
        print(f"row {r}: has missing values, skipped"); continue
    probs = model.predict_proba(flow)[0]
    print(f"row {r}: predicted {model.classes_[probs.argmax()]} ({probs.max()*100:.1f}% confident)", end="")
    if a.label_col in df.columns:
        actual = df.loc[r, a.label_col]
        print(f"  | actual label: {actual}", end="")
        if actual not in model.classes_:
            print("  (NOT one of the trained classes, so this answer cannot be right)", end="")
        else:
            print("  OK" if actual == model.classes_[probs.argmax()] else "  WRONG", end="")
    print("\n   probabilities:", {c: round(float(p), 3) for c, p in zip(model.classes_, probs)})
