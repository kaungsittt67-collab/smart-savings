"""Turn the two saved scikit-learn models into JSON files that a web page can read.

Run in Colab after Part C (or on any computer with scikit-learn), in the folder that has
savings_model.joblib, saver_model.joblib and customers_with_groups.csv:
    python export_for_web.py
It writes savings_model.json, saver_model.json and customers.json for GitHub Pages.
"""
import json
import joblib
import pandas as pd

# ---------- 1. Fixed-deposit model (Random Forest inside a pipeline) ----------
savings = joblib.load("savings_model.joblib")
prep = savings["model"].named_steps["columntransformer"]
forest = savings["model"].steps[-1][1]
scaler = prep.named_transformers_["standardscaler"]
onehot = prep.named_transformers_["onehotencoder"]
num_cols, cat_cols = prep.transformers_[0][2], prep.transformers_[1][2]

trees = []
for est in forest.estimators_:
    t = est.tree_
    value = t.value[:, 0, :]
    yes = (value[:, 1] / value.sum(axis=1)).round(6)  # chance of "yes" in each leaf
    trees.append({"f": t.feature.tolist(), "t": [float(x) for x in t.threshold],
                  "l": t.children_left.tolist(), "r": t.children_right.tolist(), "p": yes.tolist()})

json.dump({"model_name": savings["model_name"], "num": list(num_cols),
           "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
           "cat": list(cat_cols), "categories": [list(c) for c in onehot.categories_],
           "choices": savings["choices"], "scores": savings["scores"], "trees": trees},
          open("savings_model.json", "w"), separators=(",", ":"))

# ---------- 2. Saver-group model (StandardScaler + K-Means) ----------
saver = joblib.load("saver_model.joblib")
sc, km = saver["model"].named_steps["standardscaler"], saver["model"].named_steps["kmeans"]
json.dump({"features": list(saver["features"]), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
           "centers": km.cluster_centers_.tolist(),
           "names": {str(k): v for k, v in saver["names"].items()},
           "tips": {str(k): v for k, v in saver["tips"].items()},
           "profile": {str(g): row.round(1).tolist() for g, row in saver["profile"].iterrows()}},
          open("saver_model.json", "w"), indent=1)

# ---------- 3. Customers for the chart ----------
customers = pd.read_csv("customers_with_groups.csv")
customers[["Annual Income (k$)", "Spending Score (1-100)", "group", "group_name"]].to_json(
    "customers.json", orient="records")
print("wrote savings_model.json, saver_model.json, customers.json")
