import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATS = ["vector_sim", "trigram_sim", "hybrid_score"]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", default="research/models/ltr_xgb.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    X = df[FEATS].values
    y = df["label"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(Xtr, ytr)

    p = model.predict_proba(Xte)[:, 1]
    print("AUC:", roc_auc_score(yte, p))

    joblib.dump(model, args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()