#src/rag/rag2.py
import re
from typing import List, Dict, Tuple
from openai import OpenAI
from supabase import create_client

from bot_rag.config import EMBEDDING_MODEL, RERANK_MODEL

class RAG2:
    def __init__(self, openai_api_key: str, supabase_url: str, supabase_service_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.sb = create_client(supabase_url, supabase_service_key)
        self.embedding_model = EMBEDDING_MODEL

    def embed(self, text: str) -> List[float]:
        r = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float",
        )
        return r.data[0].embedding

    def search_hybrid(self, query: str, top_k: int = 20) -> List[Dict]:
        q_emb = self.embed(query)
        res = self.sb.rpc("match_qa_hybrid", {
            "query_text": query,
            "query_embedding": q_emb,
            "match_count": top_k,
        }).execute()
        return res.data or []
    
    def search_hybrid_with_embedding(self, query_text: str, query_embedding: List[float], top_k: int = 20) -> List[Dict]:
        """
        Same as search_hybrid(), but reuses precomputed embedding to avoid extra OpenAI calls.
        """
        res = self.sb.rpc("match_qa_hybrid", {
            "query_text": query_text,
            "query_embedding": query_embedding,
            "match_count": top_k,
        }).execute()
        return res.data or []

    def fetch_answers(self, answer_ids: List[int]) -> Dict[int, Dict]:
        if not answer_ids:
            return {}
        res = (
            self.sb.table("qa_answers")
            .select("answer_id,answer,answer_clean,lang")
            .in_("answer_id", answer_ids)
            .execute()
        )
        rows = res.data or []
        return {int(r["answer_id"]): r for r in rows}

    def pick_candidates(self, hits: List[Dict], max_unique: int = 7) -> List[Tuple[int, float]]:
        best = {}
        for h in hits:
            aid = int(h["answer_id"])
            sc = float(h.get("score", 0.0))
            if aid not in best or sc > best[aid]:
                best[aid] = sc
        items = sorted(best.items(), key=lambda x: -x[1])
        return items[:max_unique]

    def rerank_if_needed(self, question: str, candidates: List[Tuple[int, float]], answers_map: Dict[int, Dict]) -> int:
        if not candidates:
            return -1
        if len(candidates) == 1:
            return candidates[0][0]

        top1 = candidates[0][1]
        top2 = candidates[1][1]
        gap = top1 - top2

        if top1 >= 0.55 and gap >= 0.03:
            return candidates[0][0]

        lines = []
        for i, (aid, sc) in enumerate(candidates, start=1):
            ans = (answers_map.get(aid, {}) or {}).get("answer_clean") or (answers_map.get(aid, {}) or {}).get("answer", "")
            lines.append(f"{i}) id={aid} score={sc:.3f} answer={ans[:500]}")

        r = self.client.responses.create(
            model=RERANK_MODEL,
            instructions="Choose the single best answer for the user's question. Return ONLY the number (1..N).",
            input="USER QUESTION:\n" + question + "\n\nCANDIDATES:\n" + "\n\n".join(lines),
        )
        txt = (r.output_text or "").strip()
        m = re.search(r"\d+", txt)
        if not m:
            return candidates[0][0]
        idx = int(m.group())
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1][0]
        return candidates[0][0]
