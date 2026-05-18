# Podcast Database Updater

> **Part 1 of a two-part project.** This repository builds the searchable podcast database. Part 2 is the AI agent that queries it.

A fully automated pipeline that turns podcast audio into a searchable vector database — with zero manual intervention once deployed. Point it at any RSS feed, and every new episode becomes semantically searchable within minutes of running.

---

## What it does

```
RSS Feed → Download MP3 → Transcribe → Embed → ChromaDB
```

The pipeline runs four steps in sequence:

| Step | What happens | Tool used |
|------|-------------|-----------|
| 1. Download | New episodes are fetched from the RSS feed. Spotify metadata (title, date, link) is attached to each one. | feedparser + Spotify API |
| 2. Transcribe | The MP3 audio is converted to text. Large files are automatically split into 10-minute chunks before sending. | Groq Whisper API |
| 3. Embed | The transcript is cut into overlapping text chunks (~1000 chars), then each chunk is converted into a vector (a list of numbers that captures meaning). | HuggingFace Inference API |
| 4. Store | The vectors and their metadata are stored in a local ChromaDB database. The pipeline records the latest episode date so it only processes new episodes on the next run. | ChromaDB |

---

## Why this architecture?

### Why use APIs instead of running models locally?

The short answer: **Docker image size**.

Running Whisper and an embedding model locally requires downloading those models into the image. That pushes the image from ~100 MB to ~7 GB making cold starts slow, cloud deployments expensive, and iteration painful.

By calling external APIs instead, the Docker image stays lean (~100 MB), starts in ~5 seconds, and needs no GPU.

| Approach | Image size | Cold start | GPU needed | Cost |
|----------|-----------|------------|------------|------|
| Local models | ~7 GB | Minutes | Yes (ideally) | Hardware |
| API-based (this project) | ~100 MB | ~5 seconds | No | Free tiers |

Every external service is also a **swap point**: replace Groq with OpenAI, HuggingFace with Cohere, ChromaDB with Pinecone, the pipeline structure doesn't change.

### Why RSS?

RSS feeds are the open standard used by every podcast platform. No scraping, no fragile HTML parsing, just a clean XML feed with episode titles, dates, and direct audio links. Point the pipeline at any RSS URL and it works.

---

## API limitations and their impact on the workflow

Understanding the free tier constraints is important, they directly shape how the pipeline behaves.

### Groq Whisper API (transcription)

**Free tier limits:**
- **25 MB per file** — audio files larger than this are automatically split by the pipeline using `ffmpeg` (into 10-minute chunks). The chunks are transcribed individually and the text is reassembled.
- **20 requests per minute** — the pipeline retries with exponential backoff when rate-limited.
- **7,200 audio seconds per hour** (~2 hours/hour)
- **28,800 audio seconds per day** (~8 hours/day)
- **~20 hours of audio per month**

**Impact on the workflow:** The `MAX_DOWNLOADED_EPISODES_PER_RUN` environment variable (default: `5`) limits how many episodes are processed per run. This prevents a single run from exhausting the daily or monthly quota. For a weekly pipeline processing average-length episodes (~20–40 min each), the free tier is generally sufficient. For a large backfill of hundreds of episodes, you will hit the monthly limit and need to spread runs across multiple days.

### HuggingFace Inference API (embeddings)

**Free tier limits:**
- **~30,000 requests per month** — each text chunk sent for embedding counts as one request.
- **Rate limited** — the pipeline waits automatically when the API returns a 429 or a "model loading" (503) response.
- **Model cold start** — the first request after a period of inactivity may take ~20 seconds while the model loads on HuggingFace's servers.

**Impact on the workflow:** A single 20-minute episode produces roughly 200–230 sentences, which the pipeline groups into ~10–20 chunks of ~1000 characters. At that rate, 30,000 requests covers around 1,500–3,000 episodes per month. For a typical weekly podcast, this is not a concern. For a large catalog backfill, pace your runs.

**Model used:** `sentence-transformers/all-MiniLM-L6-v2` — a compact, fast model that produces 384-dimensional vectors. It performs well for semantic search in French despite being multilingual.

### Spotify API (metadata)

**Free tier limits:** The Spotify API is free for non-commercial use. Calls are rate-limited per application, but at one call per episode download, this is rarely a problem.

**Impact on the workflow:** The pipeline searches for an exact episode title match and verifies that the result belongs to the correct show (using `SPOTIFY_PODCAST_NAME`). This prevents false positives when another podcast shares the same episode title or when one podcast name is a substring of another. If no exact match is found, the `spotify_url` field is stored as `null`. The pipeline continues regardless — the Spotify link is enrichment, not a requirement.

`SPOTIFY_PODCAST_NAME` must match the show name exactly as it appears on Spotify (case-insensitive). If this variable is not set, the pipeline falls back to the RSS feed title.

### ChromaDB (storage)

ChromaDB runs **locally as a file-based database** — no external API, no rate limits, no cost. The database is stored in the `podcast_db_local_fr/` folder and mounted as a Docker volume so it persists across runs.

---

## Project structure

```
podcast-db-updater/
├── src/
│   ├── pipeline_update_db.py   # Entry point — orchestrates all 4 steps
│   ├── download_podcasts.py    # Step 1: RSS parsing, MP3 download, Spotify metadata
│   ├── transcribe_podcast.py   # Step 2: calls groq_transcription, saves .txt files
│   ├── groq_transcription.py   # Groq Whisper API client with auto file-splitting
│   ├── hf_embeddings.py        # HuggingFace Inference API embedding function
│   └── store_podcast.py        # Step 4: chunking, embedding, ChromaDB upsert
├── podcast_db_local_fr/        # ChromaDB database (auto-created, git-ignored)
├── mp3_downloads/              # Temporary MP3 + transcript files (auto-created)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Requirements

- **Python 3.12+** (or Docker)
- **ffmpeg** — required for splitting audio files over 25 MB
  - Ubuntu/Debian: `apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`
- Four API keys (all free tier):
  - [Groq](https://console.groq.com/keys) — transcription
  - [HuggingFace](https://huggingface.co/settings/tokens) — embeddings
  - [Spotify Developer](https://developer.spotify.com/dashboard) — episode metadata
  - *(No Mistral key is needed for this part of the project)*

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
# Podcast source
RSS_FEED=https://your-podcast-rss-feed.com/feed.xml
COLLECTION_NAME=my_podcast_collection
SPOTIFY_PODCAST_NAME=Exact Podcast Name As Shown On Spotify

# API keys
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret

# Pipeline behavior
MAX_DOWNLOADED_EPISODES_PER_RUN=5   # Limit per run to stay within API quotas
```

### `SPOTIFY_PODCAST_NAME`

The exact name of your podcast as it appears on Spotify (e.g. `Éducation Positive`). Used both as the search filter and to verify that the returned episode belongs to the right show. This prevents false matches when another podcast shares the same episode title or when your podcast name appears inside another show's name. If not set, falls back to the RSS feed title — but setting it explicitly is recommended.

### `MAX_DOWNLOADED_EPISODES_PER_RUN`

This controls how many new episodes are processed in a single run. The default is `5`. Lower it if you are near your Groq daily/monthly audio quota. Raise it if you are doing a large initial backfill and have quota to spare. Episodes are always processed oldest-first, so you can run the pipeline multiple times to gradually backfill a large catalog.

---

## Running the pipeline

### With Docker (recommended)

```bash
# Copy and fill in your .env file
cp .env.example .env

# Run the pipeline
docker-compose up --build
```

The ChromaDB database is mounted as a volume at `./podcast_db_local_fr` and persists between runs. On the next run, the pipeline reads the latest episode date from the database and only downloads episodes published after that date.

### Locally (without Docker)

```bash
# Install dependencies (requires uv)
uv sync

# Run the pipeline
cd src
uv run pipeline_update_db.py
```

> **Note:** Make sure `ffmpeg` is installed on your system before running locally.

---

## How incremental updates work

The pipeline is designed to run repeatedly without duplicating work:

1. On first run, it downloads all episodes since the year 2000 (effectively all of them), up to `MAX_DOWNLOADED_EPISODES_PER_RUN`.
2. After storing each episode in ChromaDB, it records the most recent `published_date` in a special `system_metadata` collection inside the database.
3. On subsequent runs, it reads that date and only downloads episodes published after it.
4. ChromaDB uses `upsert` — re-processing an episode that already exists updates it rather than creating duplicates.

---

## Swapping components

The pipeline is designed so each external dependency can be replaced independently:

| Component | Current | Alternatives |
|-----------|---------|-------------|
| Transcription | Groq Whisper API | OpenAI Whisper API, AssemblyAI, local Whisper model |
| Embeddings | HuggingFace Inference API | Cohere, OpenAI, local sentence-transformers |
| Vector database | ChromaDB (local) | Pinecone, Weaviate, Qdrant |
| Metadata | Spotify API | Apple Podcasts API, or RSS metadata only |

---

## What's next

This repository is **Part 1: the Database Builder**.

Part 2 — the AI agent that queries this database to answer questions about the podcast.
