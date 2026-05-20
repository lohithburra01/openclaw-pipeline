# Race Replay Pipeline — Design

Date: 2026-05-20
Status: Approved

## Context

The Formulytics F1 content pipeline renders telemetry-driven videos. The
**race-replay** visual is a vertical 1080x1920 "position vs. laps" timelapse
chart that animates every driver's race position over the course of a session.

Today this visual exists only as the last working cell (cell 22) of
`RACE_REPLAY.ipynb` — the function `create_race_timelapse()`. It ends with a
hardcoded test call and cannot run unattended.

A parallel engine already exists for qualifying: `render.py` (config-driven
lap-comparison) plus `.github/workflows/render.yml`, which renders on GitHub
Actions and uploads results to Google Drive. The race-replay work should reach
the same level of automation.

## Goal

Produce a standalone `race_replay.py` that runs **fully automatically** on
GitHub Actions: calendar-driven, rendering each Grand Prix Race and Sprint
shortly after it finishes, with no per-race manual input and no wasted Actions
minutes.

### In scope

- Standalone `race_replay.py` derived from notebook cell 22 (notebook untouched).
- Handles both **Race** and **Sprint** sessions in one script.
- Fully automatic, calendar-driven scheduling — no manual config per race.
- Runs only in the hours right after a session ends.
- Retries data fetch across runs until FastF1 data is available.
- Uploads the finished MP4 to the existing Google Drive folder.

### Out of scope

- The qualifying script (separate engine, handled later).
- Caption generation and social-media posting.
- Any modification to `RACE_REPLAY.ipynb`.

## Key constraints (from the user)

- Zero manual input once set up.
- Must run **only right after a session ends**, not all weekend, not all day.
- Minimal GitHub Actions minutes.
- Calendar-driven — the F1 schedule decides when it runs.

## Why dynamic scheduling is required

GitHub Actions `on.schedule` cron is **static text** in the workflow file;
there is no native "run once at this future timestamp." Race start times vary
widely across the season (verified against the 2026 FastF1 calendar — e.g.
Australia 04:00, Miami 17:00, Canada 20:00, Las Vegas 04:00 UTC), so a fixed
cron cannot fire "right after the event ends" for every race.

The only way to satisfy the constraints is to have the pipeline **rewrite its
own cron** from the F1 calendar. Editing a file under `.github/workflows/`
requires a Personal Access Token with `workflow` scope; the default
`GITHUB_TOKEN` is blocked from doing so. This is the single new secret the
design introduces.

The first weekend the system will face is the **Canadian Grand Prix (Round 5)**
— a sprint weekend: Sprint Sat 2026-05-23 16:00 UTC, Race Sun 2026-05-24
20:00 UTC. The design must handle both formats from the first run.

## Architecture

A single self-scheduling script plus one workflow file.

### Files

| File | Status | Purpose |
|------|--------|---------|
| `race_replay.py` | new | Cell 22 render code + self-scheduling orchestrator |
| `.github/workflows/race_replay.yml` | new | Self-managed crons + weekly safety tick |
| `RACE_REPLAY.ipynb` | untouched | Source of the render code; not modified |

Python dependencies are installed inline in the workflow (consistent with the
existing `render.yml`); no `requirements.txt` is added.

### `race_replay.py` — two phases per invocation

Every time the script runs it performs both phases in order.

**Phase 1 — Reschedule (always runs):**

1. Load the FastF1 event schedule for the current season.
2. Collect every **Race** and **Sprint** session.
3. Compute each session's end time: `session_start_utc + end_buffer`
   (`RACE_END_BUFFER_MIN` or `SPRINT_END_BUFFER_MIN`).
4. Keep a session as "schedulable" if its last retry slot is still in the
   future (`end_time + max(RETRY_OFFSETS_MIN) > now`) and it starts within
   `LOOKAHEAD_DAYS`. This naturally includes the rest of the current weekend
   and the next weekend, and drops sessions whose retry window has passed.
5. For each schedulable session, generate retry-slot timestamps
   (`end_time + offset` for each offset in `RETRY_OFFSETS_MIN`) and convert
   each to a GitHub cron line (`M H D Mon *`).
6. Always include `WEEKLY_SAFETY_CRON`.
7. Rewrite the `on.schedule` cron block in `race_replay.yml`. If the file
   content changed, `git commit` and `git push` using `WORKFLOW_PAT`.

**Phase 2 — Render if due:**

1. Find Race/Sprint sessions whose end time is in the past and within
   `RENDER_LOOKBACK_HOURS`.
2. For each, build the output filename slug
   `{event_slug}_{year}_{R|S}.mp4`.
3. Query the Google Drive folder for a file of that name. If it exists, the
   session is already done — skip (`ALREADY_RENDERED`).
4. Otherwise call `session.load(laps=True, telemetry=False)`. If
   `len(session.laps) == 0`, the data is not ready — print `DATA_NOT_READY`
   and move on; a later cron slot will retry.
5. When data is ready, render with `create_race_timelapse(...)` and upload the
   MP4 to Google Drive.

A weekend with both a Sprint and a Race produces two videos (two Drive files),
handled independently by the Phase 2 loop.

### Render code changes vs. notebook cell 22

The render code is copied verbatim, with three minimal changes:

- **Dynamic session label.** Cell 22 hardcodes the word `SPRINT` in the
  on-screen title. Derive a label from the session type (`R` -> `RACE`,
  `S`/`Sprint` -> `SPRINT`) and use it in the title string.
- **Cache enabled.** Add `fastf1.Cache.enable_cache(...)` pointing at a cache
  directory (the workflow caches this path between runs).
- **Entry point.** Replace the hardcoded `if __name__ == "__main__"` test call
  with the orchestrator's `main()`.

### Tunable constants (top of `race_replay.py`)

| Constant | Default | Meaning |
|----------|---------|---------|
| `SEASON` | current UTC year | F1 season to schedule/render |
| `RACE_END_BUFFER_MIN` | 135 | Minutes after race start to treat as "ended" |
| `SPRINT_END_BUFFER_MIN` | 75 | Minutes after sprint start to treat as "ended" |
| `RETRY_OFFSETS_MIN` | `[15, 60, 105, 150, 195]` | Retry slots after a session ends |
| `LOOKAHEAD_DAYS` | 9 | How far ahead to schedule sessions |
| `RENDER_LOOKBACK_HOURS` | 6 | Window after end time in which a session may render |
| `WEEKLY_SAFETY_CRON` | `0 12 * * 3` | Wednesday self-heal tick |
| `DURATION_SECONDS` | 30 | Output video length |
| `FPS` | 60 | Output frame rate |
| `PORTRAIT` | `True` | 1080x1920 vertical output |

### Workflow — `.github/workflows/race_replay.yml`

- Triggers: `schedule` (the self-managed cron block; seeded with only
  `WEEKLY_SAFETY_CRON`) and `workflow_dispatch` (manual bootstrap / testing).
- `permissions: contents: write`.
- `concurrency` group so overlapping runs cannot double-render.
- `timeout-minutes: 30`.
- Steps: checkout (with `token: WORKFLOW_PAT` so the script can push workflow
  edits) -> setup Python 3.11 -> cache pip and FastF1 -> install deps
  (`fastf1`, `opencv-python-headless`, `pillow`, `numpy`, `pandas`, Google API
  libraries) -> run `python race_replay.py` -> upload the MP4 as an artifact
  backup.
- Env: the four `GDRIVE_*` secrets plus `WORKFLOW_PAT`.

## Data flow

```
GitHub Actions cron / manual dispatch
        |
        v
race_replay.py
   |                      |
   v                      v
Phase 1: Reschedule    Phase 2: Render if due
   |                      |
read FastF1 calendar   find just-ended Race/Sprint
compute end times      check Drive for output file
build cron slots         |  exists -> skip
rewrite race_replay.yml   |  missing -> session.load()
commit + push (PAT)       |     no laps -> DATA_NOT_READY (exit 0)
                          |     laps ok -> create_race_timelapse()
                          v                -> upload MP4 to Google Drive
```

## Error handling and exit codes

Exit **0** for every normal outcome so scheduled runs are not marked as
failures:

- `NO_SESSION` — nothing to render this run.
- `DATA_NOT_READY` — FastF1 has no laps yet; a later slot retries.
- `ALREADY_RENDERED` — output already in Google Drive.
- Success — rendered and uploaded.

Exit **non-zero** only on genuine failures, so they show red in the Actions
tab:

- FastF1 calendar load failure (cannot schedule or render).
- Render exception.
- Google Drive auth or upload failure (the workflow still keeps the artifact
  backup).
- `git push` of the cron edit failure.

**Self-healing:** the weekly safety tick re-derives the entire schedule from
the calendar, so even a fully missed weekend is recovered within a week. The
weekly commit also keeps the repository active, preventing GitHub from
auto-disabling the scheduled workflow after 60 days of inactivity.

## Edge cases

- **Sprint weekend** — Sprint and Race handled as independent sessions; two
  videos, two Drive files.
- **Both sessions un-rendered** when a run fires — Phase 2 loops over each.
- **Empty Drive** (first weekend) — no marker files exist, so rendering
  proceeds normally.
- **Delayed scheduled run** (GitHub load at the top of the hour) — the spread
  of retry slots absorbs the delay.
- **Stale cron lines** — cron lines carry an explicit day and month, so they
  fire once; the weekly rewrite replaces them before they could recur a year
  later.
- **Re-render prevention across retry slots** — the first slot to succeed
  uploads to Drive; subsequent slots see the file and exit in seconds.

## Testing strategy

- **Local render smoke test** — run with `--force-session "2026 Miami R"`
  (a completed past session) and a short duration to a temp file; confirm an
  MP4 is produced. Skips Drive upload.
- **Scheduling dry run** — `--dry-run` prints the cron block the script would
  write, without committing or pushing.
- **Bootstrap run** — after merge, trigger `race_replay.yml` once via
  `workflow_dispatch`; confirm it writes the Canada GP cron slots and exits
  without rendering (Canada has not happened yet).

## Secrets

| Secret | Status | Used for |
|--------|--------|----------|
| `WORKFLOW_PAT` | **new** | Classic PAT, `repo` + `workflow` scopes — lets the script commit cron edits to the workflow file |
| `GDRIVE_REFRESH_TOKEN` | existing | Google Drive auth (shared with `render.yml`) |
| `GDRIVE_CLIENT_ID` | existing | Google Drive auth |
| `GDRIVE_CLIENT_SECRET` | existing | Google Drive auth |
| `GDRIVE_FOLDER_ID` | existing | Target Drive folder |

## Bootstrap procedure

1. Create the `WORKFLOW_PAT` secret (github.com/settings/tokens — classic
   token, `repo` + `workflow` scopes).
2. Merge `race_replay.py` and `race_replay.yml` to the default branch.
3. Trigger `race_replay.yml` once via `workflow_dispatch`. It writes the first
   real cron slots (Canadian GP) and exits. The pipeline is autonomous from
   that point.
