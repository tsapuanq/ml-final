import os
import pandas as pd
import numpy as np
from joblib import load

K_VALUES = [1, 3, 5, 10, 20]

# ---------- metrics ----------
def recall_at_k(ranked_ids, gold_id, k):
    return int(gold_id in ranked_ids[:k])

def mrr_at_k(ranked_ids, gold_id, k):
    topk = ranked_ids[:k]
    if gold_id not in topk:
        return 0.0
    rank = topk.index(gold_id) + 1
    return 1.0 / rank

def eval_one_ranking(group: pd.DataFrame, score_col: str):
    # group contains rows for ONE query with candidates
    gold = int(group["gold_answer_id"].iloc[0])
    # sort candidates by score desc
    g = group.sort_values(score_col, ascending=False)
    ranked = [int(x) for x in g["cand_answer_id"].tolist()]

    out = {}
    for k in K_VALUES:
        out[f"Recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"MRR@{k}"] = mrr_at_k(ranked, gold, k)
    return out

def summarize(method_name: str, per_query_rows: list[dict]):
    df = pd.DataFrame(per_query_rows)
    summary = {"method": method_name}
    for k in K_VALUES:
        summary[f"Recall@{k}"] = float(df[f"Recall@{k}"].mean())
        summary[f"MRR@{k}"] = float(df[f"MRR@{k}"].mean())
    return summary

# ---------- main ----------
def main():
    # paths
    ltr_path = "data/exports/ltr_train.csv"  # у тебя так лежит
    logreg_path = "ml/models/ltr_logreg.joblib"
    xgb_path = "ml/models/ltr_xgb.joblib"

    df = pd.read_csv(ltr_path)

    # ожидаемые колонки (по твоему примеру):
    # query,gold_answer_id,cand_answer_id,vector_sim,trigram_sim,hybrid_score,label
    required = {"query","gold_answer_id","cand_answer_id","vector_sim","trigram_sim","hybrid_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in ltr_train.csv: {missing}")

    # --- baselines on SAME candidate pool ---
    results = []

    # Vector baseline (rank by vector_sim)
    perq = []
    for _, g in df.groupby("query", sort=False):
        perq.append(eval_one_ranking(g, "vector_sim"))
    results.append(summarize("vector_only_rank", perq))

    # Trigram baseline
    perq = []
    for _, g in df.groupby("query", sort=False):
        perq.append(eval_one_ranking(g, "trigram_sim"))
    results.append(summarize("trigram_only_rank", perq))

    # Hybrid baseline
    perq = []
    for _, g in df.groupby("query", sort=False):
        perq.append(eval_one_ranking(g, "hybrid_score"))
    results.append(summarize("hybrid_rank", perq))

    # --- LTR rerank ---
    # feature columns (same ones you already have)
    feature_cols = ["vector_sim", "trigram_sim", "hybrid_score"]

    # LogReg
    if os.path.exists(logreg_path):
        model = load(logreg_path)
        df_lr = df.copy()
        # probability of label=1
        df_lr["ltr_logreg_score"] = model.predict_proba(df_lr[feature_cols].values)[:, 1]

        perq = []
        for _, g in df_lr.groupby("query", sort=False):
            perq.append(eval_one_ranking(g, "ltr_logreg_score"))
        results.append(summarize("ltr_logreg_rerank", perq))
    else:
        print(f"WARN: {logreg_path} not found, skipping logreg.")

    # XGBoost
    if os.path.exists(xgb_path):
        model = load(xgb_path)
        df_x = df.copy()
        # some xgb models expose predict_proba, some only predict
        if hasattr(model, "predict_proba"):
            df_x["ltr_xgb_score"] = model.predict_proba(df_x[feature_cols].values)[:, 1]
        else:
            df_x["ltr_xgb_score"] = model.predict(df_x[feature_cols].values)

        perq = []
        for _, g in df_x.groupby("query", sort=False):
            perq.append(eval_one_ranking(g, "ltr_xgb_score"))
        results.append(summarize("ltr_xgb_rerank", perq))
    else:
        print(f"WARN: {xgb_path} not found, skipping xgb.")

    summary = pd.DataFrame(results)
    summary = summary.sort_values(["Recall@1","MRR@10","Recall@5"], ascending=False)

    os.makedirs("data/exports", exist_ok=True)
    out_path = "data/exports/final_eval_summary.csv"
    summary.to_csv(out_path, index=False)

    print("\n=== FINAL SUMMARY (same candidate pool) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()