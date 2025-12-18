import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

FEATS = ["vector_sim", "trigram_sim", "hybrid_score"]


def split_by_group(df: pd.DataFrame, group_col: str = "group_key", test_size: float = 0.2, seed: int = 42):
    """
    Split so that the same group never appears in both train and test.
    This prevents leakage for ranking-style datasets.
    """
    groups = df[group_col].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    n_test = max(1, int(len(groups) * test_size))
    test_g = set(groups[:n_test])

    tr = df[~df[group_col].isin(test_g)].copy()
    te = df[df[group_col].isin(test_g)].copy()
    return tr, te


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", default="research/models/ltr_xgb.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)

    # group by (query, gold_answer_id) to avoid leakage
    df["group_key"] = df["query"].astype(str) + "||" + df["gold_answer_id"].astype(str)

    tr_df, te_df = split_by_group(df, group_col="group_key", test_size=0.2, seed=42)

    Xtr = tr_df[FEATS].values
    ytr = tr_df["label"].values
    Xte = te_df[FEATS].values
    yte = te_df["label"].values

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