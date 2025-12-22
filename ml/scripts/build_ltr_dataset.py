# research/scripts/build_ltr_dataset.py
import os
import csv
import time
import random
import argparse
from typing import Dict, List, Any, Tuple, Set

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=OPENAI_API_KEY)

def make_supabase():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

sb = make_supabase()

def with_retry(fn, *, tries: int = 6, base_sleep: float = 1.0, label: str = ""):
    last_err = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            sleep_s = min(30.0, base_sleep * (2 ** attempt))
            print(f"[WARN] {label} failed attempt={attempt+1}/{tries}: {type(e).__name__}: {e}")
            time.sleep(sleep_s)
    raise last_err


def embed(text: str) -> List[float]:
    def _call():
        r = client.embeddings.create(model=EMBED_MODEL, input=text, encoding_format="float")
        return r.data[0].embedding
    return with_retry(_call, tries=6, base_sleep=1.0, label="embed")


def rpc_with_retry(fn_name: str, payload: dict, *, tries: int = 6, base_sleep: float = 1.0):
    """
    RPC retry + recreate supabase client if transport dies (HTTP2/SSL EOF).
    """
    global sb
    last_err = None
    for attempt in range(tries):
        try:
            res = sb.rpc(fn_name, payload).execute()
            return res.data or []
        except Exception as e:
            last_err = e
            print(f"[WARN] RPC {fn_name} failed attempt={attempt+1}/{tries}: {type(e).__name__}: {e}")

            try:
                sb = make_supabase()
            except Exception as e2:
                print(f"[WARN] supabase recreate failed: {type(e2).__name__}: {e2}")

            sleep_s = min(30.0, base_sleep * (2 ** attempt))
            time.sleep(sleep_s)

    raise last_err


def rpc_vector(q_emb, topk):
    return rpc_with_retry("match_qa_vector", {"query_embedding": q_emb, "match_count": topk})

def rpc_trigram(q_text, topk):
    return rpc_with_retry("match_qa_trigram", {"query_text": q_text, "match_count": topk})

def rpc_hybrid(q_text, q_emb, topk):
    return rpc_with_retry("match_qa_hybrid", {"query_text": q_text, "query_embedding": q_emb, "match_count": topk})

def merge_feats(v, t, h):
    feats: Dict[int, Dict[str, float]] = {}

    for row in v:
        aid = int(row["answer_id"])
        feats.setdefault(aid, {})["vector_sim"] = float(row.get("similarity", 0.0))

    for row in t:
        aid = int(row["answer_id"])
        feats.setdefault(aid, {})["trigram_sim"] = float(row.get("trigram", 0.0))

    for row in h:
        aid = int(row["answer_id"])
        feats.setdefault(aid, {})["hybrid_score"] = float(row.get("score", 0.0))

    out = []
    for aid, d in feats.items():
        out.append({
            "cand_answer_id": aid,
            "vector_sim": d.get("vector_sim", 0.0),
            "trigram_sim": d.get("trigram_sim", 0.0),
            "hybrid_score": d.get("hybrid_score", 0.0),
        })
    return out


def load_done_queries(out_path: str) -> Set[Tuple[str, int]]:
    """
    Return set of (query, gold_answer_id) already written to out csv.
    So we can resume if script crashed mid-run.
    """
    done: Set[Tuple[str, int]] = set()
    if not out_path or not os.path.exists(out_path):
        return done

    with open(out_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if "query" not in rdr.fieldnames or "gold_answer_id" not in rdr.fieldnames:
            return done
        for row in rdr:
            q = row.get("query", "")
            try:
                gold = int(row.get("gold_answer_id", "0"))
            except Exception:
                continue
            if q:
                done.add((q, gold))
    return done


def ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_index_csv", required=True)
    ap.add_argument("--eval_csv", required=True, help="exclude leakage: answer_id in eval")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="data/exports/ltr_train.csv")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep between queries (seconds)")
    ap.add_argument("--resume", action="store_true", help="resume if out exists (recommended)")
    args = ap.parse_args()

    df = pd.read_csv(args.qa_index_csv)
    eval_df = pd.read_csv(args.eval_csv)

    eval_answer_ids = set(eval_df["answer_id"].astype(int).tolist())

    df = df[~df["answer_id"].astype(int).isin(eval_answer_ids)].copy()

    if args.limit and len(df) > args.limit:
        df = df.sample(args.limit, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    ensure_parent_dir(args.out)

    done = load_done_queries(args.out) if args.resume else set()
    if done:
        print(f"[INFO] Resume enabled. Found {len(done)} processed queries in {args.out}")

    fieldnames = [
        "query", "gold_answer_id",
        "cand_answer_id",
        "vector_sim", "trigram_sim", "hybrid_score",
        "label"
    ]

    out_exists = os.path.exists(args.out)
    mode = "a" if out_exists else "w"

    with open(args.out, mode, encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if not out_exists:
            writer.writeheader()

        for _, r in tqdm(df.iterrows(), total=len(df)):
            q = str(r["search_text"])
            gold = int(r["answer_id"])

            if args.resume and (q, gold) in done:
                continue

            q_emb = embed(q)

            v = rpc_vector(q_emb, args.topk)
            t = rpc_trigram(q, args.topk)
            h = rpc_hybrid(q, q_emb, args.topk)

            feats = merge_feats(v, t, h)

            for feat in feats:
                row_out = {
                    "query": q,
                    "gold_answer_id": gold,
                    **feat,
                    "label": 1 if feat["cand_answer_id"] == gold else 0,
                }
                writer.writerow(row_out)

            f_out.flush()
            done.add((q, gold))

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("[OK] Saved:", args.out)


if __name__ == "__main__":
    main()
