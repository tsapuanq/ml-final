-- Extensions assumed: vector, pg_trgm, pgcrypto

CREATE TABLE qa_answers (
    answer_id BIGSERIAL PRIMARY KEY,
    answer TEXT NOT NULL,
    lang TEXT,
    meta JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    answer_hash TEXT UNIQUE,
    answer_clean TEXT
);

CREATE TABLE qa_index (
    index_id BIGSERIAL PRIMARY KEY,
    answer_id BIGINT NOT NULL REFERENCES qa_answers(answer_id),
    lang TEXT,
    search_text TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    weight REAL,
    meta JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    search_hash TEXT UNIQUE
);

CREATE TABLE qa_chunks (
    id BIGSERIAL PRIMARY KEY,
    text_chunk TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE qa_paraphrase_done (
    base_search_hash TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX qa_index_embedding_ivfflat ON qa_index USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);
CREATE INDEX qa_index_search_text_trgm_idx ON qa_index USING gin (search_text gin_trgm_ops);
CREATE INDEX qa_chunks_embedding_ivfflat ON qa_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
