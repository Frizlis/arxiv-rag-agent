# План проекта: мультиагентный RAG чат-бот по статьям arXiv

Nov 10, 2026

## Цель проекта и обзор

Цель — построить FastAPI-сервис с чат-ботом, который отвечает на вопросы по научным статьям arXiv и возвращает ответ вместе с найденными документами. Внутри работает мультиагентный граф LangGraph из трёх агентов (guardrail → RAG → writer) и отдельного ToolNode для векторного поиска. Всё поднимается одной командой `docker compose up`.

Зачем этот проект: на одном примере отработать полный цикл LLM-приложения — данные, эмбеддинги, векторная БД, агенты, конфигурация, API и деплой.

**Данные.** [arXiv Dataset (Cornell University, Kaggle)](https://www.kaggle.com/datasets/Cornell-University/arxiv), лицензия CC0. Документ = заголовок + аннотация статьи, метаданные = id, авторы, категории, дата. Рабочий срез: категории `cs.CL`, `cs.AI`, `cs.LG` за последние 1–2 года, примерно 20–50 тыс. документов.

**Стек.**

| Слой | Выбор | Зачем |
| --- | --- | --- |
| Оркестрация агентов | LangGraph + LangChain | Граф состояний, ToolNode, условные переходы |
| LLM | OpenAI (`gpt-4o-mini`) или локальная через Ollama | Переключение только через `.env` |
| Эмбеддинги | `sentence-transformers` (например, `BAAI/bge-small-en-v1.5`) или OpenAI embeddings | Считаются заранее при загрузке в БД |
| Векторная БД | PostgreSQL + pgvector | Косинусное расстояние (`<=>`), SQL для статистик и фильтров |
| Конфигурация | `pydantic-settings` + `.env` | Все параметры вне кода |
| API | FastAPI + Uvicorn | REST-эндпоинт `/chat`, Swagger из коробки |
| Деплой | Docker + docker compose | Сервисы app, db, (ollama) одной командой |
| Тесты | pytest | Проверка инструментов, графа и API |

## Архитектура

Запрос проходит по графу: guardrail отсекает нерелевантное, RAG-агент в цикле с ToolNode ищет документы, writer собирает итоговый ответ.

&#91;embedded content: граф LangGraph · 3 агента, ToolNode, 1 развилка\]

RAG-агент может обращаться к ToolNode несколько раз подряд (например, переформулировать запрос). Когда документов достаточно, управление переходит к writer'у. Все узлы читают и пишут общее состояние `AgentState`.

## Структура репозитория

Код разбит по слоям: конфиг, БД, инструменты, агенты, граф, API и ingestion. Так каждый слой можно тестировать и менять отдельно.

```
arxiv-rag-agent/
├── app/
│   ├── main.py              # FastAPI: /chat, /health
│   ├── config.py            # Settings (pydantic-settings)
│   ├── llm.py               # фабрика LLM и эмбеддингов по провайдеру
│   ├── schemas.py           # Pydantic-модели API и Structured Output
│   ├── db/
│   │   ├── models.py        # таблица documents + vector-колонка
│   │   └── repository.py    # поиск по косинусу, статистики
│   ├── tools/
│   │   ├── vector_search.py # @tool search_documents
│   │   └── stats.py         # @tool document_stats (опционально)
│   ├── agents/
│   │   ├── guardrail.py
│   │   ├── rag.py
│   │   └── writer.py
│   ├── graph/
│   │   ├── state.py         # AgentState (TypedDict)
│   │   └── builder.py       # сборка StateGraph
│   └── prompts/
│       ├── loader.py
│       ├── guardrail.txt
│       ├── rag_agent.txt
│       └── writer.txt
├── ingestion/
│   ├── load_kaggle.py       # первичная загрузка снапшота
│   ├── fetch_arxiv_api.py   # сбор свежих статей из arXiv API
│   ├── preprocess.py        # очистка, нормализация
│   └── embed_and_store.py   # эмбеддинги + запись в БД
├── tests/
├── data/                    # сырые файлы (в .gitignore)
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── pyproject.toml / requirements.txt
└── README.md
```

## Поэтапный план работ

Работа идёт в 12 этапов снизу вверх: сначала фундамент (конфиг, данные, БД), потом агенты и граф, в конце API, Docker и проверка. Каждый этап даёт работающий кусок, который можно проверить до перехода к следующему.

### Этап 0. Подготовка окружения

- [ ] Создать репозиторий, виртуальное окружение, `pyproject.toml` или `requirements.txt` (langgraph, langchain-openai, langchain-ollama, sentence-transformers, psycopg, pgvector, sqlalchemy, fastapi, uvicorn, pydantic-settings, pytest).
- [ ] Настроить `.gitignore` (`.env`, `data/`), линтер (ruff) и форматтер.
- [ ] Получить Kaggle API-токен и, при желании, ключ OpenAI.

Зачем: воспроизводимое окружение и зафиксированные версии избавят от «у меня работает» при сборке в Docker.

### Этап 1. Конфигурация через pydantic-settings

- [ ] Описать класс `Settings` в `app/config.py`: `LLM_PROVIDER` (openai/ollama), `LLM_MODEL`, `LLM_TEMPERATURE`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `POSTGRES_HOST/PORT/DB/USER/PASSWORD`, `TOP_K`, `SIMILARITY_THRESHOLD`.
- [ ] Создать `.env.example` с теми же ключами без секретов.
- [ ] В `app/llm.py` написать фабрики `get_chat_model(settings)` и `get_embeddings(settings)`, которые по провайдеру возвращают `ChatOpenAI` или `ChatOllama` (аналогично для эмбеддингов).

Зачем: требование «сменить модель без изменения кода». Весь код работает с абстракцией `BaseChatModel`, а конкретный класс выбирает фабрика по значению из `.env`.

### Этап 2. Данные: загрузка и подготовка

- [ ] Скачать `arxiv-metadata-oai-snapshot.json` через Kaggle API.
- [ ] Читать файл построчно (это JSON Lines, несколько ГБ) и фильтровать по категориям и дате, не загружая всё в память.
- [ ] Очистить текст: убрать переносы строк, лишние пробелы, LaTeX-разметку в простых случаях.
- [ ] Сформировать документ: `content = title + "\n\n" + abstract`, метаданные — `arxiv_id`, `authors`, `categories`, `published_date`, `url`.

Зачем: качество поиска прямо зависит от чистоты текста, а метаданные нужны для ссылок в ответе и для tool статистик.

### Этап 3. Векторная БД и эмбеддинги

- [ ] Поднять PostgreSQL с pgvector (образ `pgvector/pgvector`) в docker compose.
- [ ] Создать таблицу `documents`: id, arxiv\_id (unique), title, abstract, authors, categories (array), published\_date, url, `embedding vector(EMBEDDING_DIM)`.
- [ ] Создать HNSW-индекс с `vector_cosine_ops`.
- [ ] Посчитать эмбеддинги батчами (по 64–256 документов) и записать в БД через upsert по `arxiv_id`.
- [ ] В `repository.py` реализовать `search(query_embedding, top_k)` через оператор `<=>` (косинусное расстояние), возвращая `similarity = 1 - distance`.

Зачем: эмбеддинги считаются заранее, а не на каждый запрос. При запросе считается только эмбеддинг вопроса, поиск идёт по индексу за миллисекунды.

### Этап 4. Инструмент векторного поиска

- [ ] Написать `@tool search_documents(query: str, top_k: int = 5)` с понятным docstring: агент читает его, чтобы решить, когда вызывать tool.
- [ ] Внутри: эмбеддинг запроса → поиск в БД → список документов со score.
- [ ] Возвращать компактный JSON (id, title, фрагмент abstract, score), чтобы не раздувать контекст LLM.

Зачем: этот tool будет обёрнут в отдельный `ToolNode` графа, как требует задание.

### Этап 5. Промпты в txt-файлах

- [ ] Создать `prompts/guardrail.txt`, `rag_agent.txt`, `writer.txt` с плейсхолдерами (`{question}`, `{documents}`).
- [ ] Написать `prompts/loader.py`: `load_prompt(name) -> str` с кэшированием.

Зачем: промпты меняются чаще кода. Вынесенные в файлы, они правятся без правки Python и хорошо видны в diff.

### Этап 6. Состояние графа (TypedDict)

- [ ] Описать `AgentState(TypedDict)` в `graph/state.py`:
  - `messages: Annotated[list[AnyMessage], add_messages]` — история диалога и вызовов tools;
  - `question: str` — исходный вопрос;
  - `is_relevant: bool`, `guardrail_reason: str` — результат модерации;
  - `retrieved_docs: list[dict]` — найденные документы;
  - `final_answer: str` — ответ writer'а;
  - `iterations: int` — счётчик шагов RAG-агента для защиты от бесконечного цикла.

Зачем: единое состояние — это «шина» между агентами. Каждый узел читает из него нужное и пишет свой промежуточный результат.

### Этап 7. Агенты

- [ ] **Guardrail.** LLM с `with_structured_output(GuardrailDecision)`, где модель — `is_relevant: bool`, `reason: str`, опционально `category` (off\_topic, harmful, ok). Пропускает вопросы о научных статьях и ML/AI, отсекает всё остальное и вредные запросы.
- [ ] **RAG agent.** LLM с `bind_tools([search_documents, document_stats])`. Сам решает: вызвать поиск (возможно несколько раз с переформулировкой) или передать управление writer'у. Лимит итераций — из конфига.
- [ ] **Writer.** LLM с `with_structured_output(FinalAnswer)`: `answer: str`, `used_doc_ids: list[str]`, `confidence`. Отвечает только по найденным документам, ссылается на них, честно говорит, если ответа в базе нет.
- [ ] Узел-экстрактор (или логика в RAG-узле) собирает документы из `ToolMessage` в `retrieved_docs`.

Зачем: разделение ответственности. Каждый агент делает одно дело, его промпт короче, а поведение предсказуемее и проще тестируется.

### Этап 8. Сборка графа LangGraph

- [ ] Узлы: `guardrail`, `rag_agent`, `tools` (`ToolNode`), `writer`, `reject`.
- [ ] Рёбра: `START → guardrail`; условный переход `guardrail → rag_agent | reject`; `rag_agent → tools | writer` через `tools_condition`-подобную функцию; `tools → rag_agent`; `writer → END`; `reject → END`.
- [ ] Скомпилировать граф, по желанию добавить checkpointer (`MemorySaver`) для истории по `thread_id`.
- [ ] Нарисовать граф через `graph.get_graph().draw_mermaid()` и положить картинку в README.

Зачем: граф делает поток управления явным. Цикл agent ↔ tools позволяет агенту самому решать, сколько раз искать.

### Этап 9. FastAPI-сервис

- [ ] `POST /chat`: вход `{question, thread_id?}`, выход `{answer, documents: [{arxiv_id, title, authors, url, score, snippet}], is_relevant, reason?}`.
- [ ] `GET /health`: проверка соединения с БД и доступности LLM.
- [ ] Граф и клиенты создаются один раз при старте (`lifespan`), а не на каждый запрос.
- [ ] Обработка ошибок (таймауты LLM, недоступная БД) и логирование.

Зачем: требование «ответ + сами документы». API — это контракт, через который сервисом пользуются фронтенд, бот или другие сервисы.

### Этап 10. Docker и docker compose

- [ ] `Dockerfile` для приложения (python-slim, установка зависимостей, `uvicorn app.main:app`).
- [ ] `docker-compose.yml` с сервисами: `db` (pgvector, volume, healthcheck), `app` (`depends_on: db: condition: service_healthy`, `env_file: .env`), `ingestion` (одноразовый запуск, profile), опционально `ollama`.
- [ ] Проверить, что `docker compose up --build` поднимает всё с нуля.

Зачем: требование деплоя. Любой человек клонирует репозиторий, копирует `.env.example` в `.env` и запускает одну команду.

### Этап 11. Тестирование и оценка качества

- [ ] Unit-тесты: загрузка конфига, загрузчик промптов, поиск в БД (на маленьком фикстурном наборе).
- [ ] Тесты графа с замоканной LLM (`FakeListChatModel`): нерелевантный вопрос уходит в reject, релевантный проходит через tools.
- [ ] Интеграционный тест `/chat` через `TestClient`.
- [ ] Мини-набор из 20–30 вопросов с ожидаемыми статьями: считать hit@k для поиска и вручную оценить ответы.

Зачем: без проверки нельзя понять, что именно сломалось — поиск, агент или промпт.

### Этап 12. Документация

- [ ] README: описание, схема графа, как запустить, примеры запросов `curl`, как переключить LLM, как перезалить данные.

Зачем: проект для портфолио оценивают в первую очередь по README.

## Опциональные задачи

Обе опции хорошо ложатся на arXiv: у него есть публичный API для свежих статей и богатые метаданные для статистик.

### Ingestion pipeline из первоисточника

- [ ] `fetch_arxiv_api.py`: запрос к arXiv API (`export.arxiv.org/api/query`) по категориям и диапазону дат, разбор Atom-XML (`feedparser`), пауза \~3 секунды между запросами по правилам API.
- [ ] Общий конвейер для обоих источников: сбор → валидация Pydantic-схемой `RawPaper` → очистка → эмбеддинги → upsert.
- [ ] Идемпотентность: повторный запуск не создаёт дублей (уникальный `arxiv_id`), а обновлённые версии статей перезаписываются.
- [ ] Запуск как отдельный сервис compose: `docker compose run ingestion --source api --categories cs.CL --since 2026-10-01`.
- [ ] Логирование: сколько собрано, отфильтровано, записано.

Зачем: в реальных системах база постоянно пополняется, а не загружается один раз. Pipeline показывает, как данные из «сырого» источника попадают в структурированное хранилище.

### Дополнительный tool: статистика по документам

- [ ] `@tool document_stats(category: str | None, date_from: str | None, date_to: str | None, group_by: Literal["category", "year", "month", "author"])` — агрегирующий SQL-запрос без векторного поиска.
- [ ] Примеры вопросов, на которые он отвечает: «Сколько статей по cs.CL в базе за 2025 год?», «Кто самые частые авторы по теме RAG?», «Как менялось число статей по месяцам?».
- [ ] Добавить tool в `bind_tools` RAG-агента и в тот же `ToolNode`; в промпте объяснить, когда нужен поиск, а когда статистика.

Зачем: агент учится выбирать между инструментами по смыслу вопроса — это ключевой навык tool-calling агентов. Для «сколько» векторный поиск бесполезен, нужен SQL.

## Чек-лист соответствия требованиям

Каждое требование задания закрывается конкретным этапом плана.

| Требование | Где реализуется | Этап |
| --- | --- | --- |
| Чат-бот для поиска по базе текстовых документов | `POST /chat` + граф агентов | 8, 9 |
| Векторная БД с заранее посчитанными эмбеддингами | PostgreSQL + pgvector, загрузка батчами | 3 |
| Мультиагентная система на LangGraph | `graph/builder.py` | 8 |
| Агент moderation/guardrail | `agents/guardrail.py`, structured output | 7 |
| RAG-агент сам решает, когда вызывать tools | `bind_tools` + условное ребро на `ToolNode` | 7, 8 |
| Writer формирует итоговый ответ | `agents/writer.py`, structured output | 7 |
| Поиск по эмбеддингам и косинусному расстоянию | оператор `<=>`, индекс `vector_cosine_ops` | 3, 4 |
| Поиск — отдельный ToolNode | узел `tools` в графе | 8 |
| Structured Output у агентов | `GuardrailDecision`, `FinalAnswer` | 7 |
| TypedDict для состояния | `AgentState` в `graph/state.py` | 6 |
| Промпты в отдельных txt | `app/prompts/*.txt` | 5 |
| Конфиг в `.env` через pydantic-settings | `app/config.py`, `.env.example` | 1 |
| Смена LLM (локальная ↔ OpenAI) без правки кода | фабрика `get_chat_model` | 1 |
| Результат: ответ + релевантные документы | схема ответа `/chat` | 9 |
| FastAPI в Docker, сборка через docker compose | `Dockerfile`, `docker-compose.yml` | 10 |
| Опционально: ingestion pipeline | `ingestion/` | Опции |
| Опционально: дополнительный tool | `document_stats` | Опции |

## Ожидаемый результат

На выходе — репозиторий, который после `cp .env.example .env` и `docker compose up --build` поднимает работающий сервис. Он отвечает на вопросы по статьям arXiv и возвращает ответ вместе с источниками.

**Что получится в итоге:**

1. **Работающий сервис.** Swagger на `http://localhost:8000/docs`, эндпоинты `/chat` и `/health`, база PostgreSQL + pgvector с десятками тысяч статей и заранее посчитанными эмбеддингами.
2. **Корректная маршрутизация запросов:**
   - «Какие подходы к уменьшению галлюцинаций в RAG предлагались в 2025 году?» → guardrail пропускает, RAG-агент ищет (возможно дважды, с переформулировкой), writer отвечает со ссылками на 3–5 статей;
   - «Какая погода в Москве?» → guardrail отклоняет с объяснением, поиск не вызывается;
   - «Сколько статей по cs.CL в базе за 2025 год?» → агент выбирает `document_stats`, а не векторный поиск.
3. **Ответ в предсказуемом формате:**

```json
{
  "answer": "Текст ответа со ссылками [1], [2]...",
  "is_relevant": true,
  "documents": [
    {"arxiv_id": "2501.01234", "title": "...", "authors": "...",
     "url": "https://arxiv.org/abs/2501.01234", "score": 0.83, "snippet": "..."}
  ]
}
```

4. **Гибкая конфигурация.** Смена `LLM_PROVIDER=openai` на `LLM_PROVIDER=ollama` в `.env` и перезапуск контейнера переключают модель без единой правки кода. Промпты правятся в txt-файлах.
5. **Обновляемая база.** Ingestion-сервис докачивает свежие статьи из arXiv API без дублей.
6. **Проверяемость.** Тесты на ключевые компоненты, мини-набор вопросов с метрикой hit@k, README со схемой графа и примерами запросов.

**Чему вы научитесь:** проектировать мультиагентные графы в LangGraph, писать tools и управлять tool-calling, использовать structured output, работать с векторной БД и эмбеддингами, выносить конфигурацию и промпты из кода, упаковывать LLM-сервис в Docker. Это законченный проект для портфолио, который можно показать на собеседовании и расширять дальше: стриминг ответа, гибридный поиск BM25 + векторы, reranker, веб-интерфейс.
