import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATS = ["vector_sim", "trigram_sim", "hybrid_score"]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", default="research/models/ltr_logreg.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    X = df[FEATS].values
    y = df["label"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    model.fit(Xtr, ytr)

    p = model.predict_proba(Xte)[:, 1]
    print("AUC:", roc_auc_score(yte, p))

    joblib.dump(model, args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()