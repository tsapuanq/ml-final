# SDU RAG Bot — структура проекта

Ниже — **как устроены папки и где что хранится**.  
(Без разборов каждого скрипта по строкам — только смысл папок и “что там лежит”.)

---

## Дерево проекта (с комментариями)

```text
.
├── .env                          # переменные окружения (ключи, URL’ы, токены)
├── .gitignore                    # что не коммитить в git
├── data                          # ВСЕ данные проекта: сырьё → обработка → экспорт → оценка
│   ├── eval                       # файлы для оценки качества (ручная разметка/выборки/кэш переписываний)
│   │   ├── eval_manual_curated.csv
│   │   ├── eval_random_stratified.csv
│   │   └── rewrite_cache.json
│   ├── exports                    # выгрузки/артефакты пайплайнов (таблицы, датасеты, результаты)
│   │   ├── ltr_train.csv
│   │   ├── qa_answers_rows.csv
│   │   ├── qa_chunks_rows.csv
│   │   ├── qa_index_rows.csv
│   │   └── qa_paraphrase_done_rows.csv
│   ├── logs                       # логи работы пайплайнов/бота (события, отладка)
│   │   └── events.jsonl
│   ├── processed                  # обработанные версии датасетов (очищено/нормализовано/подготовлено)
│   │   ├── embedding_ml_full.csv
│   │   └── QA_clean_base.csv
│   └── raw                        # сырые исходники: как пришло/скачалось/спарсилось
│       ├── provided_dataset.csv
│       ├── QA_raw.csv
│       ├── sdu_bachelor_programs.csv
│       └── sdu_bachelor_programs.json
├── docs                           # документация проекта и инфраструктуры
│   └── supabase                   # всё, что относится к БД Supabase/Postgres
│       ├── functions.sql          # SQL-функции/процедуры (если используете)
│       ├── README.md              # заметки по Supabase (как развёрнуть/настроить)
│       └── schema.sql             # схема таблиц (DDL), индексы, структуры БД
├── ml                             # машинное обучение (LTR/ранжирование), офлайн обучение моделей
│   ├── __init__.py
│   ├── models                     # сохранённые обученные модели (артефакты обучения)
│   │   ├── ltr_logreg.joblib
│   │   └── ltr_xgb.joblib
│   └── scripts                    # обучение/оценка/сбор датасета для LTR
│       ├── __init__.py
│       ├── build_ltr_dataset.py
│       ├── eval_recall.py
│       ├── train_ltr_logreg.py
│       └── train_ltr_xgb.py
├── notebooks                      # исследовательские ноутбуки (EDA, эксперименты)
│   └── EDA.ipynb
├── pipelines                      # конвейеры: ingestion → indexing → evaluation (производственный “ETL”)
│   ├── __init__.py
│   ├── evaluation                 # оценка retrieval/качества поиска
│   │   ├── __init__.py
│   │   └── eval_recall.py
│   ├── indexing                   # построение индексов/эмбеддингов/подготовка таблиц под поиск
│   │   ├── __init__.py
│   │   ├── build_index_from_qa_chunks.py
│   │   ├── clean_answers.py
│   │   └── expand_index_paraphrases_v2.py
│   └── ingestion                  # загрузка/подготовка фактов и Q/A из сырья в БД
│       ├── __init__.py
│       ├── cleaning_script.py
│       ├── ingest_sdu_programs_json.py
│       └── sdu_scrape.py
├── README.md                      # основной README проекта (этот файл/или общий)
├── requirements.txt               # зависимости Python
└── src                            # основной “продуктовый” код: бот + RAG
    └── bot_rag
        ├── __init__.py
        ├── add                    # утилиты для добавления/сидирования фактов (быстрый старт/наполнение)
        │   └── seed_facts.py
        ├── bot                    # Telegram-бот (точка входа, handlers, UI, callbacks)
        │   ├── __init__.py
        │   ├── app.py
        │   ├── callbacks.py
        │   ├── handlers.py
        │   └── ui.py
        ├── config.py              # конфиги бота/RAG (ключи, режимы, параметры, лимиты)
        └── rag                    # ядро RAG: поиск, подготовка запроса, LLM, память
            ├── __init__.py
            ├── lang.py
            ├── llm.py
            ├── memory.py
            ├── query_preprocess.py
            └── rag2.py