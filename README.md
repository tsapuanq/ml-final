# SDU AI Assistant — Multilingual RAG Chatbot (RU/KZ/EN)

AI-ассистент для студентов SDU: отвечает на частые вопросы про учебу, MySDU, пересдачи, общежитие, правила, поступление и программы обучения.  
Проект построен на Retrieval-Augmented Generation (RAG): сначала находим релевантный ответ в базе знаний, затем (опционально) формируем финальный ответ через LLM строго на основе найденного контекста.

## Why this matters
В SDU информация часто “размазана” по сайту, чатам и устным источникам.  
Цель проекта — сделать единый канал быстрых и одинаково корректных ответов 24/7, снизив нагрузку на эдвайзеров и старшекурсников.

---

## Key Features
- **Multilingual support:** Kazakh / Russian / English
- **Hybrid retrieval (production):** Vector search + Trigram search (pg_trgm)  
- **Robust to short queries & slang:** query preprocessing + canonical term expansion (например: *“мудл” → “moodle”*)
- **Supabase PostgreSQL backend:** хранение знаний + индексы + RPC функции поиска
- **Optional ML reranking (XGBoost / Logistic Regression):** обученный Learning-to-Rank для top-N кандидатов (можно включать только “когда не уверены”)
- **Telegram bot demo:** пользовательский интерфейс через Telegram

---

## Architecture (High-Level)
1) User query → preprocessing (канонизация терминов, исправление вариантов написания)  
2) Embedding запроса (OpenAI text-embedding)  
3) **Hybrid retrieval** из Supabase:
   - Vector similarity по embedding
   - Trigram similarity по search_text
   - Итоговый score = 0.8 * vector_sim + 0.2 * trigram_sim  
4) Выбор топ-кандидатов (dedup по answer_id)  
5) (Опционально) rerank (ML/LLM) только если retrieval “не уверен”  
6) Ответ берётся из `qa_answers.answer_clean` и может быть:
   - отправлен напрямую, или
   - использован как KNOWLEDGE для генерации ответа LLM (строгий режим без галлюцинаций)

---

## Database Design (Supabase / PostgreSQL)
Мы храним знания в 3 слоя:

- **qa_chunks** — “сырой” слой: исходные Q/A чанки (после слияния источников) + embedding  
- **qa_answers** — уникальные ответы (дедупликация, хранение clean-версии)  
- **qa_index** — поисковый индекс:
  - множество вариантов формулировок (search_text) → один `answer_id`
  - embedding для vector search
  - metadata (язык, источники, оригинальная формулировка)

Это позволяет:
- не дублировать ответы
- добавлять много перефразов к одному ответу
- делать устойчивый поиск под реальную речь пользователей

---

## Retrieval Methods Compared
Мы честно сравнили несколько подходов на одном evaluation set и одинаковом candidate pool size (top-N):

1. **Vector-only** (`match_qa_vector`)
2. **Trigram-only** (`match_qa_trigram`)
3. **Hybrid** (`match_qa_hybrid`) ✅ выбран в прод
4. **LTR Logistic Regression rerank**
5. **LTR XGBoost rerank**

**Почему Hybrid в проде:** качество почти как у XGBoost, но:
- не требует переобучения и мониторинга модели,
- проще поддерживать,
- стабильнее при изменении данных.

---

## Metrics
Мы использовали retrieval-метрики, которые отвечают на вопрос:
“Нашёлся ли правильный ответ в топ-K?”

- **Recall@K** (K = 1, 3, 5, 10, 20)
- **MRR@K** — учитывает позицию правильного ответа (чем выше — тем лучше)

Пример интерпретации:
- Recall@1 = 0.97 → в 97% случаев правильный answer_id на 1 месте.

---

## Results (Final)
Итоги запуска (same candidate pool):

- **LTR XGBoost rerank:** Recall@1 ≈ 0.975 (best)
- **Hybrid Rank:** Recall@1 ≈ 0.973 (почти на уровне)
- **Vector-only:** Recall@1 ≈ 0.785
- **Trigram-only:** Recall@1 ≈ 0.506

Вывод: Hybrid даёт почти максимум качества при минимальной сложности внедрения.