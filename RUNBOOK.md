# Formulytics Producer Pipeline — Runbook

This is the plain-language operating guide for the F1 video producer that
runs in this repo. Two pipelines, both fully automatic:

- **`race_replay.py`** — position-vs-laps timelapse, fired after every Race
  and Sprint session.
- **`quali_replay.py`** — top-3 Q3 lap comparison on a track map, fired
  after every Qualifying and Sprint Qualifying session.

Both upload an MP4 to the same Google Drive folder. The downstream
captioning / posting side (not in this repo) picks files up from there.

## What you receive

Each finished video lands in the Drive folder identified by the
`GDRIVE_FOLDER_ID` GitHub secret, named:

```
{location}_{year}_{R|S|Q|SQ}.mp4
```

- `R` = Race, `S` = Sprint, `Q` = Qualifying, `SQ` = Sprint Qualifying.
- Example: `monte_carlo_2026_Q.mp4`, `monza_2026_R.mp4`.
- Format: MP4, vertical 1080×1920, ~30–40 seconds (Race 30s @ 60fps,
  Quali 40s @ 30fps), ~30–70 MB.
- Cadence: one MP4 per session. A normal weekend gives you 2 (Q + R). A
  sprint weekend gives you 4 (SQ + S + Q + R).

The location string for circuits with accented characters is whatever F1's
own feed publishes (e.g. `montr_al` because the source string is
`Montréal` and the slug regex turns the `é` into `_`). The downstream
poster should not hard-code circuit names — read the file name.

## How it runs automatically — short version

Each pipeline has a `.yml` workflow file with a list of cron times
(GitHub Actions fires them at exact UTC minutes). On every fire, the
script does three things in order:

1. **Rescans the F1 calendar** at `https://livetiming.formula1.com/static/{year}/Index.json`
   and rewrites its own cron block with retry slots for upcoming sessions.
2. **Checks Google Drive** to see if the expected file is already there. If yes, exit.
3. **Renders + uploads** the video for any just-ended session whose file
   isn't in Drive yet. Five retry slots are scheduled per session (at +15
   min, +1 h, +1 h 45, +2 h 30, +3 h 15 after the session ends), so even
   if F1's data isn't published yet on the first try, a later slot will
   succeed.

A **weekly safety cron at Wednesday 12:00 UTC** fires both pipelines no
matter what. This makes sure new race weekends always get scheduled even
if the previous batch of cron slots is in the past.

You do **not** need to touch anything per race, per weekend, or per season.

## Data source

Both pipelines fetch from F1's official broadcast feed:

```
https://livetiming.formula1.com/static/<year>/Index.json
https://livetiming.formula1.com/static/<session-folder>/Index.json
https://livetiming.formula1.com/static/<session-folder>/DriverList.json
https://livetiming.formula1.com/static/<session-folder>/TimingData.jsonStream
https://livetiming.formula1.com/static/<session-folder>/TimingAppData.jsonStream
https://livetiming.formula1.com/static/<session-folder>/CarData.z.jsonStream
https://livetiming.formula1.com/static/<session-folder>/Position.z.jsonStream
```

No auth, no API key, no signup. It's the same feed that powers F1's own
live-timing webpage and mobile app. The parser is `f1_livetiming.py`
(stdlib only — `urllib`, `json`, `zlib`, `base64`).

We do **not** use FastF1 or OpenF1. Both wrappers have been in production
and both broke at some point (FastF1 blocked on cloud IPs, OpenF1 added
a live-session 401 gate). The raw endpoint has been stable for years.

## Yearly / one-time maintenance

| What | When | What to do |
|---|---|---|
| **Google Drive token** | One-time setup so it never expires | At `console.cloud.google.com/apis/credentials/consent`, set the consent screen to **Production** mode. Then run `python get_gdrive_token.py` once, copy the new refresh token into the GitHub secret `GDRIVE_REFRESH_TOKEN`. After that, the token does not expire. If you ever leave the consent screen in Testing mode, the token dies every 7 days. |
| **GitHub `WORKFLOW_PAT`** | One-time when creating it | Generate at `github.com/settings/tokens` (classic), scopes `repo` + `workflow`, expiry **No expiration**. Save as repo secret `WORKFLOW_PAT`. If you forget and let it expire, the cron self-scheduling silently freezes — symptom: no new commits to the cron block in `.github/workflows/*.yml`. |
| **Team colors** | When F1 teams change (roughly once a year) | Edit `get_constructor_color()` in `race_replay.py` and `quali_replay.py`. New teams render in grey until updated. |
| **Season change** | Never | The script uses `datetime.utcnow().year`. On Jan 1 it automatically starts looking at the new season. |

## GitHub secrets used

```
GDRIVE_CLIENT_ID         OAuth client id (Google Cloud)
GDRIVE_CLIENT_SECRET     OAuth client secret
GDRIVE_REFRESH_TOKEN     OAuth refresh token (per the table above)
GDRIVE_FOLDER_ID         Google Drive folder id where videos land
WORKFLOW_PAT             GitHub PAT with repo+workflow scope, no expiry
SMTP_USERNAME            Gmail address (for failure emails)
SMTP_PASSWORD            Gmail app password
```

## What to do if something fails

Both pipelines email `info@formula-neon.com` on failure with a direct
link to the failed Actions run. The most common failure modes:

1. **F1's livetiming endpoint returns 5xx / times out.** Transient. The
   parser retries 429/5xx with backoff inside one run; if the whole run
   fails, the next cron slot retries. Usually self-heals.

2. **F1's data not yet published for the just-ended session.** Logged
   as `DATA_NOT_READY`. Not an error — the workflow exits 0 and the next
   retry slot picks it up. Sometimes data is up to 2–3 hours late.

3. **F1 changes policy / blocks the endpoint.** This is the one
   irrecoverable case. If you start seeing 401/403 from
   `livetiming.formula1.com`, that endpoint is closed for us and we'd
   need to switch source. Has not happened in years.

4. **Google Drive token expired.** Symptom: failure email, log shows
   401 from Google. Fix: see the maintenance table above.

5. **Render error / corrupt MP4.** Look at the workflow's artifact (the
   workflow uploads `output/*.mp4` regardless of success, retained for
   5 days). Compare against a known-good past render to spot the
   regression.

To **manually trigger a render** (e.g. you missed a session):
Actions tab → "F1 Race Replay" (or Quali) → Run workflow → leave inputs
blank for a normal run, or type something like `2024 Monaco R` (or
`2024 Monaco Q`) in `test_session` to render a specific past session and
upload it to Drive prefixed with `TEST_`.

To **render locally** (when CI is broken and you need a video now):

```
python race_replay.py --force-session "2026 Monaco R"
# or
python quali_replay.py --force-session "2026 Monaco Q"
```

Requires `opencv-python-headless pillow numpy scipy`. Output goes to
`output/` next to the script. Drive upload is skipped on local runs.

## Files in this repo

| File | What it does |
|---|---|
| `race_replay.py` | The race/sprint pipeline (data + render + schedule + upload). |
| `quali_replay.py` | The quali/sprint-quali pipeline. |
| `f1_livetiming.py` | Raw fetcher + parser for `livetiming.formula1.com`. Used by both. |
| `.github/workflows/race_replay.yml` | Race workflow + auto-managed cron block. |
| `.github/workflows/quali_replay.yml` | Quali workflow + auto-managed cron block. |
| `.github/workflows/raw-pipeline-smoke.yml` | CI smoke test — fires on changes to the scripts above; renders one short quali + one short race on a clean runner. Don't delete. |
| `Formula1-*.ttf.ttf` | Font files used by both renderers for titles/labels. |
| `get_gdrive_token.py` | One-off helper to (re)issue the Google OAuth refresh token. |
| `tests/` | Unit tests for the scheduling/orchestration logic. |
