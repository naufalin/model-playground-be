# LLM Playground — Backend

Experiment deeply with one LLM or compare multiple models side-by-side. Tune prompts,
tools, skills, and reasoning settings in a durable playground session.

## Stack

- **FastAPI** — async Python web framework
- **SQLAlchemy** + **PostgreSQL** — async ORM with integer PKs + sqids encoding
- **httpx** — async HTTP client to the agent runtime service
- **JWT** — stateless auth (python-jose + pwdlib/Argon2)

## Quick Start

```bash
# 1. Install deps
uv sync

# 2. Copy env and fill in values
cp .env.example .env
# Edit .env: set SECRET_KEY, DATABASE_URL, AGENT_RUNTIME_URL, AGENT_RUNTIME_BEARER_TOKEN

# 3. Create database
createdb playground

# 4. Run migrations (includes seed data for LLM models)
uv run alembic upgrade head

# 5. Start server
uv run uvicorn playground.app:app --reload --port 8080
```

## API Overview

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/signup` | Register |
| POST | `/auth/login` | Login → JWT |
| GET | `/auth/me` | Current user |
| GET | `/models` | List available LLMs |
| POST | `/models/sync` | Sync model metadata from the agent runtime |
| GET | `/mcp/servers` | List approved MCP servers |
| GET | `/mcp/servers/{id}/tools` | Check MCP server tools |
| GET | `/pricing/models` | Normalized OpenAI and OpenRouter pricing catalog |
| POST | `/playground` | Create session |
| GET | `/playground` | List sessions |
| GET | `/playground/{id}` | Session detail + threads |
| PATCH | `/playground/{id}` | Rename session |
| DELETE | `/playground/{id}` | Delete session |
| POST | `/playground/{id}/chat` | Fan-out to N models with reasoning/perf/tool metadata (SSE) |
| POST | `/playground/{id}/chat/{thread_id}` | Continue single thread with enriched SSE |

See [docs/MCP.md](docs/MCP.md) for MCP selection, discovery, and immutable
thread snapshot semantics.

## Project Structure

```
src/playground/
├── config.py          # Settings (pydantic-settings)
├── ids.py             # sqids encode/decode
├── app.py             # FastAPI app + lifespan
├── deps.py            # Shared DI dependencies
├── auth/              # JWT auth (signup, login, me)
├── db/                # SQLAlchemy models + repos
├── models/            # GET /models endpoint
├── pricing/           # Catalog normalization + response estimates
├── sessions/          # CRUD + chat service + fanout streaming
├── mcp/               # Approved MCP catalog proxy + selection helpers
└── runtime/           # Agent runtime HTTP client
```

## Development

```bash
uv run ruff check src/       # lint
uv run pytest                # test
uv run alembic revision --autogenerate -m "description"  # new migration
```
