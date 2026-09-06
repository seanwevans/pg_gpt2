# Deploying pg_gpt2 as a chat endpoint

This walks through the two halves of a public deployment:

1. a **pg_gpt2 instance** — PostgreSQL with the extension installed, weights
   loaded, and an HTTP endpoint in front of it;
2. a **static chat UI on GitHub Pages** that calls that endpoint.

GitHub Pages is static hosting. It cannot run PostgreSQL, so the page it serves
is a client only: it holds no model and does no inference. Generation always
happens on the instance you stand up in step 1.

```
   browser                      GitHub Pages                    your host
  ┌────────┐   loads page   ┌──────────────────┐          ┌───────────────────┐
  │  user  │ ─────────────► │ site/index.html  │          │  FastAPI  :8000   │
  │        │                │ site/chat.js     │          │        │          │
  │        │ ───── POST /v1/chat/stream ─────────────────►│        ▼          │
  │        │ ◄──── text/event-stream ────────────────────  │ PostgreSQL + pg_llm│
  └────────┘                └──────────────────┘          └───────────────────┘
```

---

## 1. Stand up an instance

### With docker compose (recommended)

```bash
git clone https://github.com/seanwevans/pg_gpt2
cd pg_gpt2

# Tokenizer assets and (optionally) a converted checkpoint. See models/README.md.
mkdir -p models
curl -Lo models/vocab.json  https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
curl -Lo models/merges.txt  https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt
python scripts/convert_gpt2_checkpoint.py --source gpt2 --output models/gpt2-small.npz

docker compose up --build -d
PG_GPT2_WEIGHTS=/mnt/models/gpt2-small.npz docker compose run --rm provision

curl localhost:8000/healthz
```

Omit `PG_GPT2_WEIGHTS` and the provisioning step creates a small randomly
initialised model instead. That is enough to exercise every stage of the
pipeline — the output is gibberish, but it is gibberish produced by a real
forward pass in SQL.

### By hand

```bash
sudo apt-get install -y postgresql-server-dev-16
make && sudo make install

createdb pg_gpt2
psql -d pg_gpt2 -c 'CREATE EXTENSION pg_llm;'

python scripts/ingest_tokenizer.py --dsn "$DSN" --model gpt2-small \
    --vocab vocab.json --merges merges.txt --truncate
psql -d pg_gpt2 -c "SELECT pg_llm_import_npz('/abs/path/gpt2-small.npz','gpt2-small');"

pip install -r server/requirements.txt
PG_GPT2_DSN="$DSN" PG_GPT2_MODEL=gpt2-small \
    uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Endpoint configuration

Every setting is an environment variable read by `server/config.py`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PG_GPT2_DSN` | `postgresql://postgres@localhost:5432/postgres` | Database to generate from |
| `PG_GPT2_MODEL` | `gpt2-small` | Model name used when a request does not name one |
| `PG_GPT2_MAX_TOKENS` | `64` | Hard cap on tokens per request |
| `PG_GPT2_MAX_PROMPT_CHARS` | `2000` | Hard cap on prompt length |
| `PG_GPT2_CORS_ORIGINS` | `*` | Comma-separated origins allowed to call the API |
| `PG_GPT2_RATE_LIMIT_PER_MINUTE` | `20` | Per-client request budget (`0` disables) |
| `PG_GPT2_MAX_CONCURRENT` | `4` | In-flight generations; the rest queue |
| `PG_GPT2_STATEMENT_TIMEOUT_MS` | `120000` | Postgres-side abort for a runaway generation |

Once you know your Pages origin, narrow CORS to it:

```bash
PG_GPT2_CORS_ORIGINS=https://seanwevans.github.io
```

### Exposing it over HTTPS

A page served from `https://…github.io` cannot call an `http://` endpoint —
browsers block the mixed content. The endpoint therefore needs TLS. Put it
behind a reverse proxy that terminates TLS (Caddy, nginx + certbot, a cloud
load balancer), or expose it through a tunnel such as `cloudflared`. Point the
public hostname at port 8000 and keep Postgres itself unpublished.

---

## 2. Deploy the chat UI to GitHub Pages

The page lives in `site/` and is deployed by
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml).

1. **Enable Pages.** Repository → *Settings* → *Pages* → *Build and deployment*
   → *Source*: **GitHub Actions**. This is a one-time repository setting; the
   workflow cannot turn it on for you.
2. **Point the page at your endpoint** (optional). Repository → *Settings* →
   *Secrets and variables* → *Actions* → *Variables* → add
   `PG_GPT2_ENDPOINT` = `https://your-endpoint.example.com`. The workflow writes
   it into `config.json` so visitors do not have to type a URL. Without it the
   page renders a setup banner and asks for one.
3. **Deploy.** Push to `main` with changes under `site/`, or run the workflow
   manually from the *Actions* tab (it accepts an `endpoint` input for one-off
   deployments).

The published page is at `https://<owner>.github.io/<repo>/`.

Visitors can always override the endpoint themselves: the settings panel writes
to `localStorage`, and `?endpoint=https://host` works as a shareable link.

---

## Endpoint API

### `GET /healthz`

```json
{"status":"ok","api_version":"1","pg_llm_version":"0.1.0",
 "default_model":"gpt2-small","max_tokens":64}
```

### `GET /v1/models`

Lists every model in `llm_model_config` with its shape, how many parameter rows
it has, and whether a tokenizer is loaded for it (`ready`).

### `POST /v1/chat`

```bash
curl -X POST https://your-endpoint/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The database that dreamed of language","max_tokens":32}'
```

```json
{"model":"gpt2-small","prompt":"…","completion":"…","text":"…",
 "tokens":32,"elapsed_ms":4120}
```

`messages` may be sent instead of `prompt`, in which case the turns are
flattened into a `Human:`/`AI:` transcript. GPT-2 has no chat fine-tuning, so
this framing is a convention, not a trained behaviour.

### `POST /v1/chat/stream`

Same request body; responds with `text/event-stream`:

```
event: start
data: {"model":"gpt2-small","prompt":"Once upon a time","max_tokens":5}

event: token
data: {"step":1,"token_id":11,"delta":",","completion":","}

event: done
data: {"completion":", there was"}
```

Tokens are emitted as they are sampled. The loop lives in the API rather than
in `llm_generate_stream` because a PL/pgSQL set-returning function fills its
whole result before the first row is visible; each iteration is still a single
call to `llm_next_token`, which runs the forward pass and sampler in SQL.

---

## Operational notes

- **Everything is CPU-bound inside Postgres.** A token is a full transformer
  forward pass expressed as SQL and C kernels. Expect seconds per token for
  GPT-2 small, not milliseconds. Keep `PG_GPT2_MAX_TOKENS` small on a public
  endpoint.
- **`llm_tensor` is a shared, cluster-wide cache.** `llm_logits` warms it via
  `llm_materialize_inference_params`, which is keyed on a single owning model.
  Serving two different models from one database will make them evict each
  other's weights on every request; run one model per database.
- **The endpoint is read-only by construction** — it only calls `llm_encode`,
  `llm_next_token`, `llm_decode` and `llm_generate`. Give it a database role
  with no write privileges if you are exposing it publicly.
