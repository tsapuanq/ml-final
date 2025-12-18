CREATE OR REPLACE FUNCTION match_qa_vector(query_embedding VECTOR, match_count INT DEFAULT 20)
RETURNS TABLE(answer_id BIGINT, search_text TEXT, similarity REAL, score REAL)
LANGUAGE SQL AS $$
    SELECT answer_id,
           search_text,
           1 - (embedding <=> query_embedding) AS similarity,
           1 - (embedding <=> query_embedding) AS score
    FROM qa_index
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION match_qa_trigram(query_text TEXT, match_count INT DEFAULT 20)
RETURNS TABLE(answer_id BIGINT, search_text TEXT, trigram REAL, score REAL)
LANGUAGE SQL AS $$
    SELECT answer_id,
           search_text,
           similarity(search_text, query_text) AS trigram,
           LEAST(1.0, similarity(search_text, query_text)) AS score
    FROM qa_index
    WHERE search_text % query_text
    ORDER BY similarity(search_text, query_text) DESC
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION match_qa_hybrid(query_text TEXT, query_embedding VECTOR, match_count INT DEFAULT 20)
RETURNS TABLE(answer_id BIGINT, search_text TEXT, similarity REAL, trigram REAL, score REAL)
LANGUAGE SQL AS $$
    WITH vector_hits AS (
        SELECT answer_id, search_text, 1 - (embedding <=> query_embedding) AS similarity
        FROM qa_index
        ORDER BY embedding <=> query_embedding
        LIMIT match_count
    ),
    trigram_hits AS (
        SELECT answer_id, search_text, similarity(search_text, query_text) AS trigram
        FROM qa_index
        WHERE search_text % query_text
        ORDER BY similarity(search_text, query_text) DESC
        LIMIT match_count
    ),
    merged AS (
        SELECT v.answer_id,
               COALESCE(v.search_text, t.search_text) AS search_text,
               COALESCE(v.similarity, 0) AS similarity,
               COALESCE(t.trigram, 0) AS trigram
        FROM vector_hits v
        FULL OUTER JOIN trigram_hits t USING (answer_id)
    )
    SELECT answer_id,
           search_text,
           similarity,
           trigram,
           0.8 * similarity + 0.2 * LEAST(1.0, trigram) AS score
    FROM merged
    ORDER BY score DESC
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION get_paraphrase_candidates(max_rows INT DEFAULT 350)
RETURNS TABLE(answer_id BIGINT, lang TEXT, search_text TEXT, base_search_hash TEXT)
LANGUAGE SQL AS $$
    SELECT answer_id,
           lang,
           search_text,
           search_hash AS base_search_hash
    FROM qa_index
    WHERE meta->>'source' = 'rules'
      AND (char_length(search_text) <= 40 OR array_length(string_to_array(search_text, ' '), 1) <= 2)
      AND search_hash NOT IN (SELECT base_search_hash FROM qa_paraphrase_done)
    ORDER BY created_at NULLS LAST
    LIMIT max_rows;
$$;

CREATE OR REPLACE FUNCTION eval_match_vector(query_embedding VECTOR, top_k INT)
RETURNS TABLE(answer_id BIGINT, score REAL)
LANGUAGE SQL AS $$
    SELECT answer_id, 1 - (embedding <=> query_embedding) AS score
    FROM qa_index
    ORDER BY embedding <=> query_embedding
    LIMIT top_k;
$$;

CREATE OR REPLACE FUNCTION eval_match_hybrid(query_text TEXT, query_embedding VECTOR, top_k INT)
RETURNS TABLE(answer_id BIGINT, score REAL)
LANGUAGE SQL AS $$
    SELECT answer_id,
           0.8 * (1 - (embedding <=> query_embedding)) +
           0.2 * LEAST(1.0, similarity(search_text, query_text)) AS score
    FROM qa_index
    WHERE search_text % query_text
    ORDER BY score DESC
    LIMIT top_k;
$$;
