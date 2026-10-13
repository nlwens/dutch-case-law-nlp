# Dutch case law NLP

Classification and question answering over Dutch court rulings. Started as a student project for Saxion / JustID.

Two BERTje classifiers that read a Dutch court ruling and assign **rechtsgebieden** (law areas) and **bijzondere kenmerken** (procedure features). Both tasks are multi-label. A label is kept if its sigmoid score is at least 0.5.

Input is ruling text only. Output is two `{label: score}` maps.

A second part, [RAG](#rag-questions-over-rulings), answers questions from 2 500 indexed rulings and cites each claim by ECLI.

Weights live on the Hugging Face Hub, not in this repo:

- [`newnus/justid-rechtsgebieden`](https://huggingface.co/newnus/justid-rechtsgebieden)
- [`newnus/justid-bijzondere-kenmerken`](https://huggingface.co/newnus/justid-bijzondere-kenmerken)

## Results

Balanced `test.jsonl` is the number for the report. `natural_test.jsonl` is a second hold-out with the real year / area mix.

| Task | Split | Micro-F1 | Macro-F1 |
|---|---|---|---|
| Rechtsgebieden | test | 0.96 | 0.73 |
| Rechtsgebieden | natural_test | 0.96 | 0.75 |
| Bijzondere kenmerken | test | 0.92 | 0.76 |
| Bijzondere kenmerken | natural_test | 0.93 | 0.76 |

Micro is high because frequent labels (Strafrecht, Bestuursrecht, Hoger beroep, …) are easy. Macro is lower because rare labels barely appear in training.

On the rechtsgebieden test set, **Burgerlijk procesrecht** (22 rows) and **Materieel strafrecht** (35 rows) have F1 0.00. On bijzondere kenmerken, **Proceskostenveroordeling** (37 rows) is 0.00 and **Tussenuitspraak** is 0.16. Full per-label tables are in `results/`.

Natural_test is slightly higher than the balanced test. The models did not overfit the stratified sample.

## How to run

Python 3.12. GPU is optional (RTX 3050 / CUDA 12.1 was used for training).

```powershell
cd dutch-case-law-nlp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Predict one ruling

```powershell
.\.venv\Scripts\python.exe predict.py
```

Paste text, then a blank line. First run downloads the two Hub models if `models/rg` and `models/bk` are missing.

Notebooks: `.\.venv\Scripts\python.exe -m jupyter lab notebooks`, pick the `.venv` kernel, run `01_load` → `02_clean` → `03_train` → `04_predict`. Do not run the two `trainer.train()` cells unless you want a multi-hour job.

### API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs

- `POST /predict-text` — JSON `{"text": "..."}`
- `POST /predict-document` — PDF upload
- `POST /ask` — JSON `{"question": "..."}`, see [RAG](#rag-questions-over-rulings)
- `GET /health`

### Docker

Needs Docker Desktop. Uses CPU torch. Local `models/` is mounted when present; otherwise the container pulls from the Hub. `data/rag/` and `.env` are used when present, so `/ask` works in the container too (about 15–50 s per question on CPU).

```powershell
docker compose up --build
```

Same docs URL as above. Stop with `Ctrl+C` or `docker compose down`.

## RAG: questions over rulings

Ask in any language, get an answer in that language built only from passages of Dutch rulings, with an ECLI after every claim. If the rulings do not cover the question, the answer is `Not found in the provided rulings.` in English, and `/ask` returns `"refused": true`. The evaluation questions are in English.

Nothing is trained. All models are used as they are:

| Step | Model |
|---|---|
| Translate the question to Dutch | Gemini (`gemini-3.5-flash-lite`, API) |
| Vector search, 30 candidates each for the Dutch and the original question | `intfloat/multilingual-e5-base` + Chroma, cosine |
| Rerank, keep probability ≥ 0.5, at most 5 passages, 2 per ruling | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Answer from the kept passages | Gemini |

The original question is searched next to its translation because the translation sometimes gets a legal term wrong. In one test question, Gemini invented the Dutch word *rijonderdrinkingszaken* for drink-driving cases; with the Dutch query alone, no passage passed the reranker.

Rulings are cut into passages of about 250 words that never cross a heading. Each passage carries its section (court's assessment, party argument, decision, advisory opinion, ...), and the prompt tells the model that a party argument is not a finding of the court.

Two ways to refuse: no passage passes the reranker (Gemini is not asked to answer), or the draft answer says the passages do not address the question. Cited ECLIs are checked against the passages the model was given.

### Results

`python -m rag.evaluate` runs 29 questions in `rag/eval/questions.jsonl`: two per topic for twelve topics, and five on subjects that were left out of the corpus (flight delays, crypto, ...), which must be refused.

| Metric | Result |
|---|---|
| Topic questions with at least one on-topic passage | 24 / 24 |
| Kept passages from on-topic rulings | 95 % |
| Topic questions answered | 22 / 24 |
| Refusal questions refused | 5 / 5 |
| Cited ECLIs outside the given passages | 0 |
| Median time per question (GPU retrieval) | 4.7 s |

The two unanswered topic questions (landlord's grounds for termination, rent arrears and eviction) got on-topic passages, but the draft answers were about side issues (fixed-term leases, who to sue), and the check step judged that they did not answer the question.

"On-topic" means the ruling matches the topic's keyword rule in `rag/build_corpus.py`. That rule says a ruling is about the topic, not that the passage answers the question. So three answers were also checked by hand against their passages (`notebooks/05_rag.ipynb`): of 9 claims, 8 match the cited ruling and 1 cites a ruling that is about TBS in general for a point about compulsory treatment. One answer is correct but incomplete: it skips the passage where the court set child support to nil because of low income.

### How to run

Needs the processed ruling texts (see [Data](#data)) and a free Gemini API key from [AI Studio](https://aistudio.google.com/apikey) in `.env` in the project root:

```
GEMINI_API_KEY=...
```

Build the corpus, passages and index once:

```powershell
.\.venv\Scripts\python.exe -m rag.build_corpus
.\.venv\Scripts\python.exe -m rag.chunk
.\.venv\Scripts\python.exe -m rag.build_index
```

Then ask a question, run the evaluation, or use `POST /ask` in the API:

```powershell
.\.venv\Scripts\python.exe -m rag.answer "When do courts extend a TBS order?"
.\.venv\Scripts\python.exe -m rag.evaluate
```

`notebooks/05_rag.ipynb` walks through the two retrieval stages, the refusals and the manual check. It reads the evaluation output and makes no Gemini calls.

## Data

The ruling texts (`data/processed/*.jsonl`) are not in git. Notebooks `01` and `02` expect those files locally. Kept labels are listed in `data/processed/labels.json`. After cleaning: **65 316** rulings. Labels with fewer than 100 training examples are dropped: 20 rechtsgebieden and 25 bijzondere kenmerken remain.

The RAG corpus is 2 500 rulings from `test` and `natural_test`, so none of them were used to train the classifiers. `data/rag/` (corpus, passages, Chroma index) is gitignored.

## Layout

```
app.py                 FastAPI
predict.py             shared inference
config.py              paths, Hub ids, threshold, RAG models
notebooks/             01 load → 02 clean → 03 train → 04 eval + predict → 05 RAG
rag/                   corpus, chunking, index, retrieval, answer, evaluation
results/               eval.json + per-label CSV, rag_eval.json + rag_answers.jsonl
models/                local weights (gitignored)
```

## Limits

- BERTje truncates at 512 tokens. Long rulings lose the tail.
- Rare labels stay weak at threshold 0.5. Per-label thresholds were not searched.
- PDF upload needs a text layer. Scanned pages return 400.
- RAG: 29 questions is a small test set, and the answer prompt was adjusted after seeing false refusals on some of them, so they are not fully held out.
- RAG: Gemini's translation differs slightly between runs even at temperature 0, and a different Dutch query retrieves different passages. Between two runs of the same code, two questions flipped from answered to refused.
- RAG: a cited ECLI is checked to be among the passages, not that the claim is right. The model only knows its five passages and does not always use the best one, so answers can be incomplete.
- RAG: some section labels are wrong. Appeal rulings often restate a party's grounds under a "beoordeling" heading, which is labelled as the court's assessment.
- RAG: the question and the retrieved passages are sent to the Gemini API. The rulings are public, but do not send confidential text. On the free tier, Google may use the data to improve its models.
