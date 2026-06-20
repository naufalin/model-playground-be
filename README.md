# LLM Playground — Backend

Test multiple LLMs side-by-side. Send one prompt, get responses from N models in parallel.

## Stack

- **FastAPI** — async Python web framework
- **SQLAlchemy** + **PostgreSQL** — async ORM with integer PKs + sqids encoding
- **httpx** — async HTTP client to the agent runtime service
- **JWT** — stateless auth (python-jose + passlib/bcrypt)

## Quick Start

```bash
# 1. Install deps
uv sync

# 2. Copy env and fill in values
cp .env.example .env
# Edit .env: set SECRET_KEY, DATABASE_URL

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
| POST | `/playground` | Create session |
| GET | `/playground` | List sessions |
| GET | `/playground/{id}` | Session detail + threads |
| DELETE | `/playground/{id}` | Delete session |
| POST | `/playground/{id}/chat` | Fan-out to N models (SSE) |
| POST | `/playground/{id}/chat/{thread_id}` | Continue single thread (SSE) |

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
├── playground/        # CRUD + chat + fanout streaming
└── runtime/           # Agent runtime HTTP client
```

## Development

```bash
uv run ruff check src/       # lint
uv run pytest                # test
uv run alembic revision --autogenerate -m "description"  # new migration
```
