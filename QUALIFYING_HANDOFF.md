# Qualifying Video — Build Handoff

**Read this if you are picking up work to add the QUALIFYING video format.**
The Race/Sprint pipeline is finished and working. Qualifying is the remaining
format. This document is everything you need to start.

---

## The goal

Add a **qualifying** video format to the Formulytics F1 pipeline, fitting it into
the same automated workflow that already produces race videos. The finished
qualifying videos must land in the same Google Drive folder, named
`{location}_{year}_Q.mp4` (the downstream posting pipeline is already told to
expect the `_Q` suffix).

---

## What already exists and works (the race pipeline)

| File | Purpose |
|---|---|
| `race_replay.py` | Self-contained: fetches data from OpenF1, renders the "position vs. laps" race-replay video, uploads to Google Drive, **and rewrites its own GitHub Actions cron** to self-schedule. |
| `.github/workflows/race_replay.yml` | The workflow. Runs `race_replay.py`. Its `schedule:` cron block is auto-managed by the script (between marker comments). |
| `get_gdrive_token.py` | One-time helper to (re)generate the Google Drive OAuth credentials. |
| `tests/test_race_replay.py` | Unit tests for the pure logic (24 tests). |
| `docs/superpowers/specs/2026-05-20-race-replay-pipeline-design.md` | Design spec. **Caveat: written before the OpenF1 migration — it still says FastF1. The code uses OpenF1.** |
| `docs/superpowers/plans/2026-05-20-race-replay-pipeline.md` | Implementation plan. Same caveat. |
| `render.py` + `config.json` + `.github/workflows/render.yml` | An **older, separate** engine: a cross-driver fastest-lap comparison with a track map and a delta graph ("HiFi Delta Engine"). It is config-driven, **FastF1-based**, and not integrated into the self-scheduling workflow. This is the closest existing thing to a qualifying visual — treat it as a starting reference, not as finished work. |

---

## CRITICAL learnings — do not rediscover these the hard way

1. **FastF1's data feed is BLOCKED on GitHub Actions runners.** `fastf1`'s
   `session.load()` fails on CI with "Failed to load timing data" /
   `DataNotLoadedError` — F1's feed rejects datacenter IPs. It works locally
   (residential IP) but never on GitHub. The race pipeline was migrated from
   FastF1 to **OpenF1** (`api.openf1.org`) specifically because of this.
   **The qualifying script must use OpenF1 for any data it fetches on CI.**
   Do not use FastF1 in the pipeline.

2. **Google Drive uploads** use OAuth user credentials in four GitHub secrets:
   `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`, `GDRIVE_REFRESH_TOKEN`,
   `GDRIVE_FOLDER_ID`. The OAuth consent screen must be in **Production** mode
   or the refresh token expires every 7 days. `get_gdrive_token.py` regenerates
   the credentials.

3. **`WORKFLOW_PAT` secret** (a PAT with `repo` + `workflow` scopes) is required
   because the script edits its own workflow file's cron, and the default
   `GITHUB_TOKEN` is not allowed to modify files under `.github/workflows/`.

---

## OpenF1 data available for qualifying

OpenF1 (`https://api.openf1.org/v1/`, no auth) covers qualifying:

- `sessions?year=YYYY` — includes `session_name` values `"Qualifying"` and
  `"Sprint Qualifying"`. Each has a `session_key`, `date_start`,
  `location`, `circuit_short_name`.
- `laps?session_key=K` — per-driver lap timings and sector times.
- `car_data?session_key=K` — telemetry: `speed`, `throttle`, `brake`,
  `n_gear`, `rpm`, `drs`, timestamped per driver.
- `location?session_key=K` — track position: `x`, `y`, `z` coordinates,
  timestamped per driver (for a track-map visual).
- `drivers?session_key=K` — `name_acronym` (3-letter code), `team_name`,
  `team_colour`.

So OpenF1 can support a telemetry / fastest-lap / track-map qualifying video —
which is what `render.py` does, but `render.py` would need its FastF1 data
layer reworked onto OpenF1.

---

## How the qualifying piece should fit

- **Reuse the race pipeline's patterns.** In `race_replay.py` these are directly
  reusable: `openf1_get()` (HTTP GET with 429/5xx retry), `get_drive_service()`
  / `drive_has_file()` / `drive_upload_file()`, `commit_and_push()`, and the
  self-scheduling machinery — `list_sessions()`, `schedulable_sessions()`,
  `due_sessions()`, `build_cron_lines()`, `rewrite_cron_block()` and the
  `CRON_BEGIN`/`CRON_END` marker convention.
- **Scheduling:** `race_replay.py` currently filters sessions to `Race`/`Sprint`
  only. Qualifying happens Saturday (or Friday on sprint weekends). The
  scheduling filter would need `Qualifying` added.
- **Architecture decision to make:** either extend the existing workflow/script
  to also handle qualifying, or add a parallel `quali_replay.py` + its own
  self-scheduling workflow. Decide this during design.
- **Output filename:** must be `{location}_{year}_Q.mp4` so the posting pipeline
  consumes it correctly.
- **Render engine:** `render.py` is the closest existing qualifying visual
  (fastest-lap comparison + track map). Decide whether to adapt it or design a
  fresh qualifying visual — then rework its data layer onto OpenF1 and wire in
  the Drive upload + self-scheduling.
- **Test hook:** `race_replay.yml` has a `workflow_dispatch` input
  `test_session` that renders a chosen past session end-to-end (used to verify
  on a real runner). Build the same kind of test trigger for qualifying.

---

## Suggested process

Follow the same flow that produced the race pipeline: **brainstorm the design**
(what should the qualifying video actually be?), **write a plan**, then build it
with review checkpoints. Verify on a real GitHub runner before declaring done —
the FastF1-block was only caught by testing on an actual runner.

## Repo state

Branch `main`, everything pushed. There is pre-existing uncommitted noise in the
working tree (`RACE_REPLAY.ipynb` modified, `output.mp4` deleted, a few untracked
files) — that is not part of this work; leave it alone.
