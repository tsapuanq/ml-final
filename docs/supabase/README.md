# Supabase Database (QA Retrieval)

This Supabase project stores data for QA retrieval.  
Most retrieval logic is implemented inside Postgres as RPC functions (SQL functions), e.g. `match_qa_hybrid`.

## High-level idea

We have:
- **Canonical answers** in `qa_answers`
- **Search index rows** (query phrases + embeddings) in `qa_index`, each row points to an answer
- **Optional text chunks** in `qa_chunks` (chunk + embedding)
- **Paraphrase progress** in `qa_paraphrase_done` to avoid repeated paraphrase expansion

At runtime, the app calls Postgres RPC functions to find the best `answer_id`.

---

## Extensions

Enabled extensions:
- `vector` (pgvector) — vector columns and ANN indexes (ivfflat)
- `pg_trgm` — trigram similarity search (`%` operator + `similarity()`)
- `pgcrypto`, `uuid-ossp`, etc.

---

## Tables

### 1) `qa_answers` (canonical answers)

Purpose: store final answer text returned to users.

Columns:
- `answer_id` bigint (PK)
- `answer` text (NOT NULL)
- `lang` text (nullable)
- `meta` jsonb (nullable)
- `created_at` timestamptz (nullable)
- `answer_hash` text (nullable, UNIQUE)
- `answer_clean` text (nullable)

Indexes:
- PK: `qa_answers_pkey (answer_id)`
- Unique: `qa_answers_answer_hash_uq (answer_hash)`

---

### 2) `qa_index` (search phrases → answers)

Purpose: store searchable phrases / paraphrases and map them to answers.

Columns:
- `index_id` bigint (PK)
- `answer_id` bigint (NOT NULL) — references `qa_answers.answer_id`
- `lang` text (nullable)
- `search_text` text (NOT NULL)
- `embedding` vector (NOT NULL)
- `weight` real (nullable)
- `meta` jsonb (nullable)
- `created_at` timestamptz (nullable)
- `search_hash` text (nullable, UNIQUE)

Indexes:
- PK: `qa_index_pkey (index_id)`
- Unique: `qa_index_search_hash_uq (search_hash)`
- Trigram: `qa_index_search_text_trgm_idx` (GIN, `search_text gin_trgm_ops`)
- Vector ANN: `qa_index_embedding_ivfflat` (ivfflat on `embedding vector_cosine_ops`, lists=200)

Notes:
- `weight` column exists but is not used in current RPC scoring.
- Multiple `qa_index` rows can map to the same `answer_id`.

---

### 3) `qa_chunks` (text chunks + embeddings)

Purpose: store standalone chunks of text with embeddings. Can be used as a corpus or as a source to build the QA index.

Columns:
- `id` bigint (PK)
- `text_chunk` text (NOT NULL)
- `embedding` vector (NOT NULL)
- `created_at` timestamptz (nullable)

Indexes:
- PK: `qa_chunks_pkey (id)`
- Vector ANN: `qa_chunks_embedding_ivfflat` (ivfflat, lists=100)

---

### 4) `qa_paraphrase_done` (paraphrase progress)

Purpose: track which base search hashes were already expanded with paraphrases.

Columns:
- `base_search_hash` text (PK)
- `created_at` timestamptz (nullable)

Indexes:
- PK: `qa_paraphrase_done_pkey (base_search_hash)`

---

## Security / RLS

RLS is disabled (`rowsecurity = false`) on all tables and there are no policies.

This means access is controlled by DB roles / keys.  
In production, these tables should typically be accessed only by server-side code using the Service Role key.

---

## RPC Functions (Retrieval)

### `match_qa_vector(query_embedding, match_count=20)`
Vector-only retrieval from `qa_index`:
- similarity = `1 - (embedding <=> query_embedding)`
- ordered by `<=>` (closest vectors first)

Returns: `(answer_id, search_text, similarity, score)` where `score == similarity`.

---

### `match_qa_trigram(query_text, match_count=20)`
Text-only retrieval using pg_trgm:
- filters by `search_text % query_text`
- trigram = `similarity(search_text, query_text)`

Returns: `(answer_id, search_text, trigram, score)` where `score = least(1.0, trigram)`.

---

### `match_qa_hybrid(query_text, query_embedding, match_count=20)`
Hybrid retrieval combining vector and trigram candidates.

Process:
1) Top K by vector similarity (v)
2) Top K by trigram similarity (k)
3) Union candidates and take best per `answer_id`
4) Final score:

`score = 0.80 * vector_similarity + 0.20 * min(1.0, trigram_similarity)`

Returns:
- `answer_id`
- `search_text` (one representative phrase)
- `similarity` (vector)
- `trigram` (text)
- `score` (final)

---

### `get_paraphrase_candidates(max_rows=350)`
Returns candidates from `qa_index` to expand with paraphrases.

Logic:
- only rows where `meta->>'source' = 'rules'`
- only short phrases: `char_length <= 40` OR `word_count <= 2`
- excludes items already in `qa_paraphrase_done` by `search_hash`

Returns:
- `answer_id, lang, search_text, base_search_hash`

---

### Eval helpers
- `eval_match_vector`
- `eval_match_hybrid`

These return `(answer_id, score)` to evaluate ranking quality.


---

## Typical data flow

1) Insert canonical answers into `qa_answers` (dedup by `answer_hash`).
2) Insert search phrases into `qa_index` (dedup by `search_hash`) + store embeddings.
3) (Optional) Store text chunks in `qa_chunks`.
4) (Optional) Run paraphrase expansion:
   - call `get_paraphrase_candidates`
   - generate paraphrases in app code
   - insert new rows into `qa_index`
   - mark completed base hashes in `qa_paraphrase_done`
5) Runtime:
   - compute query embedding
   - call RPC `match_qa_hybrid(query_text, query_embedding)`
   - take best `answer_id` and fetch final answer from `qa_answers`

---