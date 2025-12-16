#research/scripts/eval_recall.py
import os, csv, argparse, json, time, statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

KS = [1, 3, 5, 10, 20]

FOLLOWUP_HINTS = [
    # ru
    "они", "это", "там", "про них", "подробнее", "а что", "а как", "а где", "а сколько", "нет про",
    # en
    "it", "they", "there", "more", "tell me more", "what about", "about it", "about them",
    # kk (простые подсказки)
    "ол", "олар", "сол", "толығырақ", "тағы"
]

@dataclass
class EvalRow:
    qid: str
    question: str
    answer_id: int
    lang: str
    topic: str
    qtype: str
    split: str

def load_eval(path: str) -> List[EvalRow]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(EvalRow(
                qid=r["qid"],
                question=r["question"],
                answer_id=int(r["answer_id"]),
                lang=r.get("lang","").strip(),
                topic=r.get("topic","").strip(),
                qtype=r.get("qtype","").strip(),
                split=r.get("split","").strip(),
            ))
    return rows

def rank_of(target: int, preds: List[int]) -> int:
    try:
        return preds.index(target) + 1
    except ValueError:
        return 0

def recall_at_k(ranks: List[int], k: int) -> float:
    return sum(1 for r in ranks if 1 <= r <= k) / max(1, len(ranks))

def mrr_at_k(ranks: List[int], k: int) -> float:
    s = 0.0
    for r in ranks:
        if 1 <= r <= k:
            s += 1.0 / r
    return s / max(1, len(ranks))

def hitrate(ranks: List[int]) -> float:
    return sum(1 for r in ranks if r > 0) / max(1, len(ranks))

def mean_rank(ranks: List[int]) -> float:
    rr = [r for r in ranks if r > 0]
    return float(statistics.mean(rr)) if rr else 0.0

def median_rank(ranks: List[int]) -> float:
    rr = [r for r in ranks if r > 0]
    return float(statistics.median(rr)) if rr else 0.0

def summarize(name: str, ranks: List[int]):
    print(f"\n=== {name} ===")
    print(f"N={len(ranks)} | HitRate(any@{max(KS)}): {hitrate(ranks):.3f} | MeanRank: {mean_rank(ranks):.2f} | MedianRank: {median_rank(ranks):.2f}")
    for k in KS:
        print(f"Recall@{k}: {recall_at_k(ranks, k):.3f} | MRR@{k}: {mrr_at_k(ranks, k):.3f}")

def is_followup_like(q: str) -> bool:
    t = (q or "").lower()
    if len(t.split()) <= 6:
        return True
    return any(h in t for h in FOLLOWUP_HINTS)

def group_key(row: EvalRow, by: str) -> str:
    if by == "lang":
        return row.lang or "unknown"
    if by == "topic":
        return row.topic or "unknown"
    if by == "qtype":
        return row.qtype or "unknown"
    if by == "split":
        return row.split or "unknown"
    return "all"

def print_group_breakdown(rows: List[EvalRow], ranks_map: Dict[str, List[int]], by: str):
    # ranks_map: mode -> list aligned to rows
    print(f"\n--- Breakdown by {by} ---")
    groups: Dict[str, List[int]] = {}
    idxs: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        k = group_key(r, by)
        idxs.setdefault(k, []).append(i)

    # header
    modes = list(ranks_map.keys())
    print("group\tN\t" + "\t".join([f"{m}:R@1" for m in modes]) + "\t" + "\t".join([f"{m}:R@5" for m in modes]))
    for g, id_list in sorted(idxs.items(), key=lambda x: (-len(x[1]), x[0])):
        line = [g, str(len(id_list))]
        for m in modes:
            rr = [ranks_map[m][i] for i in id_list]
            line.append(f"{recall_at_k(rr, 1):.3f}")
        for m in modes:
            rr = [ranks_map[m][i] for i in id_list]
            line.append(f"{recall_at_k(rr, 5):.3f}")
        print("\t".join(line))

class EvalClient:
    def __init__(self, rewrite_model: str, rewrite_cache_path: str, sleep_s: float = 0.0, embed_cache_path: Optional[str] = None):
        load_dotenv()
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
        self.embedding_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.rewrite_model = rewrite_model
        self.rewrite_cache_path = rewrite_cache_path
        self.sleep_s = sleep_s
        self.embed_cache_path = embed_cache_path

        # rewrite cache
        self.rewrite_cache: Dict[str, str] = {}
        if self.rewrite_cache_path and os.path.exists(self.rewrite_cache_path):
            try:
                with open(self.rewrite_cache_path, "r", encoding="utf-8") as f:
                    self.rewrite_cache = json.load(f)
            except Exception:
                self.rewrite_cache = {}

        # embedding cache
        self.embed_cache: Dict[str, List[float]] = {}
        if self.embed_cache_path and os.path.exists(self.embed_cache_path):
            try:
                with open(self.embed_cache_path, "r", encoding="utf-8") as f:
                    self.embed_cache = json.load(f)
            except Exception:
                self.embed_cache = {}

    def _save_json_atomic(self, path: str, obj):
        if not path:
            return
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _save_rewrite_cache(self):
        self._save_json_atomic(self.rewrite_cache_path, self.rewrite_cache)

    def _save_embed_cache(self):
        self._save_json_atomic(self.embed_cache_path, self.embed_cache)

    def embed(self, text: str) -> List[float]:
        key = f"{self.embedding_model}::{text}"
        if self.embed_cache_path and key in self.embed_cache:
            return self.embed_cache[key]

        r = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float",
        )
        emb = r.data[0].embedding

        if self.embed_cache_path:
            self.embed_cache[key] = emb
            self._save_embed_cache()

        return emb

    def vector_only(self, query: str, top_k: int) -> List[int]:
        q = self.embed(query)
        res = self.sb.rpc("match_qa_index", {
            "query_embedding": q,
            "match_count": top_k,
        }).execute()
        return [int(x["answer_id"]) for x in (res.data or [])]

    def hybrid(self, query: str, top_k: int) -> List[int]:
        q = self.embed(query)
        res = self.sb.rpc("match_qa_hybrid", {
            "query_text": query,
            "query_embedding": q,
            "match_count": top_k,
        }).execute()
        return [int(x["answer_id"]) for x in (res.data or [])]

    def rewrite_query(self, question: str, lang: str) -> str:
        key = f"{lang}::{question}"
        if key in self.rewrite_cache:
            return self.rewrite_cache[key]

        if lang == "ru":
            lang_hint = "Russian"
        elif lang == "kk":
            lang_hint = "Kazakh"
        else:
            lang_hint = "English"

        instructions = (
            "You are a query rewriter for an information retrieval system.\n"
            "Rewrite the user query to improve retrieval.\n"
            "Rules:\n"
            "- DO NOT answer the question.\n"
            "- DO NOT add any new facts.\n"
            "- Preserve the original meaning.\n"
            "- Keep the same language as the input.\n"
            "- Output ONLY the rewritten query text (no quotes, no extra words).\n"
        )
        inp = (
            f"Language: {lang_hint}\n"
            f"User query:\n{question}\n"
        )

        if self.sleep_s > 0:
            time.sleep(self.sleep_s)

        r = self.client.responses.create(
            model=self.rewrite_model,
            instructions=instructions,
            input=inp,
        )
        out = (r.output_text or "").strip() or question

        self.rewrite_cache[key] = out
        self._save_rewrite_cache()
        return out

def save_failures(path: str, rows: List[EvalRow], preds_by_mode: Dict[str, List[List[int]]], ranks_by_mode: Dict[str, List[int]], topk: int):
    if not path:
        return
    modes = list(preds_by_mode.keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid","question","gold_answer_id","lang","topic","qtype","split"] + [f"{m}_rank" for m in modes] + [f"{m}_top{topk}" for m in modes])
        for i, r in enumerate(rows):
            # пишем только проблемные: где хотя бы один режим не нашёл
            if all(ranks_by_mode[m][i] > 0 for m in modes):
                continue
            row = [r.qid, r.question, r.answer_id, r.lang, r.topic, r.qtype, r.split]
            for m in modes:
                row.append(ranks_by_mode[m][i])
            for m in modes:
                row.append(" ".join(map(str, preds_by_mode[m][i][:topk])))
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to eval CSV")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--rewrite_model", default="gpt-4o-mini")
    ap.add_argument("--rewrite_cache", default="data/eval/rewrite_cache.json")
    ap.add_argument("--embed_cache", default="data/eval/embed_cache.json")
    ap.add_argument("--sleep_s", type=float, default=0.0, help="sleep between rewrite calls")
    ap.add_argument("--rewrite_policy", choices=["always","followup_only","never"], default="followup_only")
    ap.add_argument("--max_rows", type=int, default=0, help="debug: limit rows (0 = all)")
    ap.add_argument("--failures_out", default="data/eval/failures.csv", help="where to save failure cases (csv)")
    ap.add_argument("--breakdown", action="store_true", help="print breakdown by lang/topic/qtype/split")
    args = ap.parse_args()

    rows = load_eval(args.csv)
    if args.max_rows and args.max_rows > 0:
        rows = rows[:args.max_rows]

    cli = EvalClient(
        rewrite_model=args.rewrite_model,
        rewrite_cache_path=args.rewrite_cache,
        sleep_s=args.sleep_s,
        embed_cache_path=args.embed_cache,
    )

    # Store per-row preds for error analysis
    preds_vec: List[List[int]] = []
    preds_hyb: List[List[int]] = []
    preds_rw:  List[List[int]] = []

    ranks_vec, ranks_hyb, ranks_hyb_rw = [], [], []

    t0 = time.time()
    for row in rows:
        # 1) vector-only
        pv = cli.vector_only(row.question, args.topk)
        preds_vec.append(pv)
        ranks_vec.append(rank_of(row.answer_id, pv))

        # 2) hybrid
        ph = cli.hybrid(row.question, args.topk)
        preds_hyb.append(ph)
        ranks_hyb.append(rank_of(row.answer_id, ph))

        # 3) hybrid + rewrite (policy-controlled)
        do_rw = False
        if args.rewrite_policy == "always":
            do_rw = True
        elif args.rewrite_policy == "followup_only":
            do_rw = is_followup_like(row.question)
        else:
            do_rw = False

        if do_rw:
            rq = cli.rewrite_query(row.question, row.lang)
        else:
            rq = row.question

        pr = cli.hybrid(rq, args.topk)
        preds_rw.append(pr)
        ranks_hyb_rw.append(rank_of(row.answer_id, pr))

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s | rows={len(rows)} | topk={args.topk} | rewrite_policy={args.rewrite_policy}")

    summarize("Vector-only", ranks_vec)
    summarize("Hybrid", ranks_hyb)
    summarize("Hybrid + Query Rewrite", ranks_hyb_rw)

    ranks_by_mode = {
        "vector": ranks_vec,
        "hybrid": ranks_hyb,
        "hybrid_rewrite": ranks_hyb_rw,
    }
    preds_by_mode = {
        "vector": preds_vec,
        "hybrid": preds_hyb,
        "hybrid_rewrite": preds_rw,
    }

    if args.breakdown:
        print_group_breakdown(rows, ranks_by_mode, by="lang")
        print_group_breakdown(rows, ranks_by_mode, by="topic")
        print_group_breakdown(rows, ranks_by_mode, by="qtype")
        print_group_breakdown(rows, ranks_by_mode, by="split")

    save_failures(args.failures_out, rows, preds_by_mode, ranks_by_mode, topk=args.topk)
    if args.failures_out:
        print(f"\nSaved failures to: {args.failures_out}")

if __name__ == "__main__":
    main()