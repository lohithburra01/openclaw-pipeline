# Race-Replay Pipeline — Handoff for the Posting Pipeline

**Read this if you are the Claude setup that takes videos from Google Drive and
posts them with AI captions.** This document is the contract: it tells you
exactly what the race-replay pipeline produces and where, so you can consume it.

---

## What the race-replay pipeline is

It is the **video-production half** of the Formulytics F1 content system. It runs
fully automatically on GitHub Actions, with no human input:

1. It self-schedules from the F1 calendar (via the free OpenF1 API).
2. Shortly after each F1 **Race** or **Sprint** session ends, it renders an
   animated "position vs. laps" replay video.
3. It uploads that video to a Google Drive folder.

Your pipeline (captioning + posting) is the **next stage**. This pipeline never
captions or posts — it only produces videos and drops them in Drive.

Repo: `github.com/lohithburra01/openclaw-pipeline`
(race side = `race_replay.py` + `.github/workflows/race_replay.yml`)

---

## What you receive — the contract

**Location.** Every finished video is uploaded to the single Google Drive folder
identified by the `GDRIVE_FOLDER_ID` secret. Point your pipeline at that same
folder.

**Filename format:** `{location}_{year}_{session}.mp4`

- `location` — lowercase circuit location, non-alphanumerics collapsed to `_`
  (e.g. `montreal`, `miami`, `silverstone`, `las_vegas`).
- `year` — 4 digits (e.g. `2026`).
- `session` — `R` for a Grand Prix race, `S` for a sprint.
- Examples: `montreal_2026_R.mp4`, `montreal_2026_S.mp4`, `miami_2026_R.mp4`.

**Video format.** MP4, **vertical 1080×1920**, 30 seconds, 60 fps, roughly
60–70 MB. Sized for Instagram Reels and YouTube Shorts — post as-is.

**Content.** An animated "race position vs. lap number" chart: every driver's
position over the session, in team colours, with tyre compounds and gaps to the
leader. The on-screen title reads `{GP} {YEAR} - RACE/SPRINT - LAP n/total`.
Watermark: `@formulytics`.

**Cadence.** One video per session. A normal Grand Prix weekend produces **one**
video (the race). A sprint weekend produces **two** (sprint + race). The 2026
season has 24 races + 6 sprints ≈ 30 videos.

**Timing.** A video appears **0–3 hours after the session ends** — the pipeline
retries until the timing data is published, then renders and uploads.

**Exactly-once.** Each video is uploaded once. The pipeline checks the folder
before rendering and skips anything already there. You will not get duplicates.

---

## Tips for the posting side

- **Detecting new videos:** watch the Drive folder for filenames you have not
  seen before. Do **not** hard-code only `_R`/`_S` — a qualifying format (`_Q`)
  is being added next (see below).
- **Caption context:** the filename gives you GP location, year, and session
  type. For real race facts (winner, podium, gaps, fastest lap) for the AI
  caption, pull from the **same free OpenF1 API** the producer uses —
  `https://api.openf1.org/v1/` — no auth required. Look up the session via
  `sessions?year={year}` (match the location), then use `session_result`,
  `position`, `drivers`, and `laps` for that `session_key`.
- **Tailor by session type:** `_R` = race, `_S` = sprint — adjust caption tone
  accordingly.

---

## Coming soon

A **qualifying** video format is being built and will land in the same Drive
folder with the suffix **`_Q`** (e.g. `montreal_2026_Q.mp4`). Make sure your
new-file detection and caption logic handle `_Q` as well as `_R` and `_S`.

The pipeline is now coordinated by a Google Sheet "season manifest" — the
authoritative design is `docs/superpowers/specs/2026-05-22-pipeline-manifest-design.md`,
and `RUNBOOK.md` is the plain-language operating guide. Older posting sketches
have been moved to `docs/archive/`.
