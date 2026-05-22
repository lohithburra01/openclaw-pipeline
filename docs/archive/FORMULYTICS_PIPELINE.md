# Formulytics Pipeline — Build Instructions for Claude Code

## Context

This is the Formulytics F1 content pipeline. The goal is a fully automated zero-cost system that:
1. Detects when an F1 session ends
2. Waits for FastF1 data to become available (retries every 10 min)
3. Runs the Python render script to produce a video
4. Generates a caption using a free LLM
5. Posts the video to Instagram Reels and YouTube Shorts via upload-post.com
6. All of this runs on GitHub Actions — no server, no local machine needed

## Repo

`https://github.com/lohithburra01/openclaw-pipeline`

The only existing file is `F1UnityBaker_HiFi.py` — the core telemetry engine.
CLI: `python F1UnityBaker_HiFi.py --year 2026 --gp Monaco --session R --drivers VER NOR --fps 60 --output output/video.mp4`

## What You Need to Build

### 1. `runner.py` — orchestrator script

This script is called by GitHub Actions. It must:

- Accept a `--session` argument in format `"2026 Monaco R"` (year, event, session type)
- Check if FastF1 data is available for that session:
  ```python
  import fastf1
  session = fastf1.get_session(year, event, session_type)
  session.load(laps=True, telemetry=False, weather=False, messages=False)
  return len(session.laps) > 0
  ```
- If data is NOT available: print "DATA_NOT_READY" and exit with code 0
- If data IS available:
  - Dynamically select top 3 drivers from session results using FastF1
  - Call `F1UnityBaker_HiFi.py` via subprocess with those drivers
  - Generate a caption by calling OpenRouter API (see caption section below)
  - Upload the output video to upload-post.com (see posting section below)
  - Print "DONE" and exit 0
- Check for a `.posted_{session_slug}` flag file in the repo root — if it exists, exit immediately (already posted)
- After successful post, create that flag file, git commit and push it so future runs skip

### 2. Caption generation via OpenRouter

Use OpenRouter free tier. Model: `qwen/qwen-2.5-72b-instruct:free`
API base: `https://openrouter.ai/api/v1`
Auth header: `Authorization: Bearer $OPENROUTER_API_KEY`

The caption prompt must follow this format exactly — this is the Formulytics brand voice:

```
You are writing an Instagram Reel + YouTube Shorts caption for the F1 account "Formulytics".

Session: {session_description}  (e.g. "2026 Monaco Grand Prix Race")
Top 3 result: {p1_driver} P1, {p2_driver} P2, {p3_driver} P3
Gap P1 to P2: {gap}s

Rules:
- Hook line first (no hashtags yet), punchy, rivalry or drama framing
- 2-3 lines of context, short sentences
- One engagement question at the end
- Then hashtags on a new line: #f1 #formula1 #{eventname}GP #f1reels #shorts
- Max 150 words total
- No em dashes
```

### 3. Posting via upload-post.com

API docs: https://docs.upload-post.com/api/overview/
Auth: `Authorization: Apikey $UPLOAD_POST_API_KEY`
Endpoint: `POST https://api.upload-post.com/api/upload`

The video file must be uploaded as multipart form data. Platforms: `instagram` and `youtube`.
Pass the generated caption as the `title` field and description.

```python
import requests

def post_video(video_path, caption, title):
    with open(video_path, "rb") as f:
        response = requests.post(
            "https://api.upload-post.com/api/upload",
            headers={"Authorization": f"Apikey {os.environ['UPLOAD_POST_API_KEY']}"},
            files={"video": f},
            data={
                "title": title,
                "description": caption,
                "platform[]": ["instagram", "youtube"],
                "user": os.environ["UPLOAD_POST_USERNAME"],
            }
        )
    response.raise_for_status()
    return response.json()
```

### 4. `.github/workflows/f1_pipeline.yml`

This is the GitHub Actions workflow. It must:

- Trigger on `workflow_dispatch` (manual) AND on a cron schedule
- The cron fires every 10 minutes during a configurable time window (set per race weekend)
- Use `ubuntu-latest`
- Install Python 3.11
- Install dependencies from `requirements.txt`
- Run `python runner.py --session "${{ github.event.inputs.session || env.DEFAULT_SESSION }}"`
- If the script prints "DATA_NOT_READY", the job exits cleanly (not a failure)
- Has a `timeout-minutes: 25` to avoid runaway jobs
- Secrets used:
  - `OPENROUTER_API_KEY`
  - `UPLOAD_POST_API_KEY`
  - `UPLOAD_POST_USERNAME`
  - `GH_PAT` — a GitHub personal access token with repo write access, used to push the `.posted_*` flag file back

Example workflow structure:

```yaml
name: F1 Session Pipeline

on:
  workflow_dispatch:
    inputs:
      session:
        description: 'Session string e.g. "2026 Monaco R"'
        required: true
  schedule:
    # Edit these cron lines per race weekend (UTC times)
    - cron: '*/10 13 25 5 *'
    - cron: '*/10 14 25 5 *'

env:
  DEFAULT_SESSION: "2026 Monaco R"

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    timeout-minutes: 25
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GH_PAT }}

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run pipeline
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          UPLOAD_POST_API_KEY: ${{ secrets.UPLOAD_POST_API_KEY }}
          UPLOAD_POST_USERNAME: ${{ secrets.UPLOAD_POST_USERNAME }}
        run: python runner.py --session "${{ github.event.inputs.session || env.DEFAULT_SESSION }}"
```

### 5. `requirements.txt`

Must include at minimum:
```
fastf1
requests
```

Add anything else your existing script needs.

### 6. `sessions.json` — weekend config (optional but recommended)

A config file that maps session keys to human-readable descriptions for the caption prompt:

```json
{
  "2026 Monaco R": {
    "description": "2026 Monaco Grand Prix Race",
    "event": "Monaco",
    "cron_windows": ["'*/10 13 25 5 *'", "'*/10 14 25 5 *'"]
  }
}
```

## Secrets to Add in GitHub

Go to repo Settings → Secrets and variables → Actions → New repository secret:

| Secret name | Value |
|---|---|
| `OPENROUTER_API_KEY` | From openrouter.ai — free signup |
| `UPLOAD_POST_API_KEY` | From upload-post.com — free signup |
| `UPLOAD_POST_USERNAME` | Your upload-post.com username/account identifier |
| `GH_PAT` | GitHub PAT with `repo` scope — github.com/settings/tokens |

## Signups Needed (All Free, No Card)

1. **openrouter.ai** — get API key, use model `qwen/qwen-2.5-72b-instruct:free`
2. **upload-post.com** — connect Instagram + YouTube, get API key

## File Structure When Done

```
openclaw-pipeline/
├── F1UnityBaker_HiFi.py        (already exists)
├── runner.py                   (build this)
├── requirements.txt            (build this)
├── sessions.json               (build this)
├── .github/
│   └── workflows/
│       └── f1_pipeline.yml    (build this)
└── output/                     (gitignored, temp video storage)
```

## Important Constraints

- Zero cost — no paid APIs
- OpenRouter free model only: `qwen/qwen-2.5-72b-instruct:free`
- upload-post.com free tier: 10 uploads/month
- No em dashes anywhere in captions
- Driver selection must be dynamic from FastF1 results, never hardcoded
- The `.posted_*` flag file pattern prevents duplicate posts on retry runs
- GitHub Actions on a public repo = unlimited free minutes
