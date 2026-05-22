# Formulytics Pipeline — Season Manifest Design

Date: 2026-05-22
Status: Approved (pending user review of this spec)

## Context

The Formulytics F1 content pipeline has two halves:

- **Producer** — `race_replay.py` on GitHub Actions. Calendar-driven and
  self-scheduling. Renders a "position vs. laps" video after each Race or Sprint
  and uploads it to a Google Drive folder. Already built and working.
- **Poster** — an n8n workflow. Takes a video, writes an AI caption, posts it to
  Instagram and YouTube via Zernio. In design (V1/V2/V3 build docs).

Today the poster discovers work by scanning the Drive folder. That has three
problems the owner has identified:

1. **No deliberate starting state.** 7 races of the 2026 season have already
   finished and there are no videos for them in Drive. A folder-scanning poster
   has no concept of "the season" — only "what happens to be in the folder."
2. **State is opaque.** Whether a video has been posted lives in Drive file
   metadata the owner cannot easily see.
3. **Too many breaking points.** The folder-scanning poster (V3) relied on file
   triggers, custom properties, an hourly sweep, and a timing window — several
   independent things that can each fail.

The owner is non-technical and needs a system that runs unattended for years,
is documented well enough that future edits are easy, and has as few fragile
moving parts as possible.

## Goal

Introduce a **Google Sheet "season manifest"** as the single source of truth for
the whole pipeline. Every session is a row; the producer and poster both read
and write it. The Drive folder is demoted to a file store — it holds the MP4s
but is no longer how the pipeline decides what to do.

### In scope

- A Google Sheet manifest with a defined schema.
- A self-seeding step that fills the manifest from the F1 calendar.
- Producer changes: consult and update the manifest instead of scanning Drive.
- Poster changes: drive the n8n workflow off the manifest instead of the folder.
- Robustness work for multi-year unattended operation (see Error Handling).
- One authoritative spec (this document) plus a maintenance runbook.

### Out of scope

- **Backfilling the 7 past races.** Decided: they are not posted. They are
  recorded in the manifest as `skipped`.
- **The qualifying (`_Q`) renderer.** Separate future work. The manifest schema
  accommodates `_Q` so adding it later is a non-event.
- **Changing the render visual** or the caption brand voice.

## Decisions

These were settled during brainstorming and are fixed for this design:

1. The 7 already-finished 2026 races are **not** backfilled — seeded as `skipped`.
2. The manifest covers the **whole pipeline** — producer and poster both use it.
3. The manifest is a **Google Sheet**.
4. The poster **remains an n8n workflow** (not collapsed into GitHub Actions).
5. n8n runs on **n8n Cloud** (managed hosting).

## Architecture

```
                    ┌──────────────────────────┐
                    │   Google Sheet manifest   │  <- single source of truth
                    │   (one row per session)   │
                    └─────────▲────────▲────────┘
              writes render columns │        │ writes post columns
                                    │        │
        ┌───────────────────────────┘        └────────────────────────┐
        │                                                             │
┌───────┴────────────┐                                   ┌────────────┴───────┐
│  Producer          │   uploads MP4    ┌─────────────┐   │  Poster            │
│  race_replay.py    │ ───────────────► │ Drive folder│ ◄─│  n8n Cloud         │
│  (GitHub Actions)  │                  │ (file store)│   │  downloads MP4     │
└────────────────────┘                  └─────────────┘   └────────────────────┘
```

- The **manifest** is the brain. Nothing scans the Drive folder anymore.
- The **producer** keeps its self-scheduling cron (that decides *when* it runs,
  to save GitHub minutes). What changes is *how it knows what to do*: the
  manifest row, not a Drive lookup.
- The **Drive folder** is purely file transport. The manifest row carries the
  `drive_file_id`, so the poster fetches the exact file directly — no scanning.
- The **poster** polls the manifest on a schedule and posts rows that are
  rendered but not yet posted.

## The manifest sheet

One Google Sheet, one tab named `sessions`, one row per session. Rows accumulate
across seasons (the `session_id` carries the year, so years never collide).

### Schema

Columns are referenced **by header name** in all code, never by position — so
reordering or adding columns in the sheet cannot break the pipeline.

| Header | Written by | Meaning |
|---|---|---|
| `session_id` | seed | Unique key: `{location}_{year}_{session}`, e.g. `monaco_2026_R`. Matches the Drive filename stem. |
| `gp_name` | seed | Full Grand Prix name from OpenF1, e.g. `Monaco Grand Prix`. |
| `session` | seed | `Race` or `Sprint` (`Qualifying` reserved for future `_Q` work). |
| `session_date` | seed | UTC date the session ran, `YYYY-MM-DD`. |
| `openf1_session_key` | seed | OpenF1 numeric session key. Lets the poster skip a lookup. |
| `render_status` | producer | `pending` / `rendered` / `skipped` / `error`. |
| `rendered_at` | producer | UTC timestamp the MP4 was uploaded. |
| `drive_file_id` | producer | Drive file id of the uploaded MP4. |
| `render_notes` | producer | Latest render message or error reason. |
| `post_status` | poster | `pending` / `posting` / `posted` / `skipped` / `error`. |
| `post_started_at` | poster | UTC timestamp a poster run claimed the row. |
| `posted_at` | poster | UTC timestamp the post succeeded. |
| `post_link` | poster | URL(s) of the published post. |
| `attempts` | poster | Count of failed post attempts (caps auto-retry). |
| `post_notes` | poster | Latest post message or error reason. |

### Status meanings

**`render_status`:**

- `pending` — not yet rendered; the producer renders it when the session is due.
- `rendered` — MP4 produced and uploaded; `drive_file_id` is set.
- `skipped` — deliberately not rendered (the past races); producer ignores it.
- `error` — last render attempt failed; `render_notes` has the reason; producer
  retries it while the session is still within its retry window.

**`post_status`:**

- `pending` — not yet posted; becomes eligible once `render_status` is `rendered`.
- `posting` — a poster run claimed this row at `post_started_at`. Other runs skip
  it. If `post_started_at` is more than 30 minutes old, that run is presumed
  crashed and the row becomes eligible again.
- `posted` — successfully posted; `posted_at` and `post_link` are set.
- `skipped` — deliberately not posted (past races; or the owner sets it manually
  to stop a row that keeps failing).
- `error` — last post attempt failed; retried on later polls until `attempts`
  reaches 3, then left for the owner (still visible, and an alert was emailed).

### Hard rules

- The producer **only ever writes** the render columns (`render_status`,
  `rendered_at`, `drive_file_id`, `render_notes`).
- The poster **only ever writes** the post columns (`post_status`,
  `post_started_at`, `posted_at`, `post_link`, `attempts`, `post_notes`).
- These two column sets are disjoint, and each side updates **only its own
  cells**, never a whole row — so the two writers can never clobber each other,
  even on the same row at the same time.
- Seeding is **append-only**: it adds rows that do not exist and never modifies
  the status of an existing row.

### Seeding (the "start-up" step)

A `ensure_manifest` step runs at the start of **every** producer invocation
(idempotent — usually it finds nothing to add):

1. Read every row currently in the sheet.
2. Fetch the season's Race and Sprint sessions from OpenF1.
3. For each session that has **no row yet**, append one:
   - Fill `session_id`, `gp_name`, `session`, `session_date`,
     `openf1_session_key`.
   - If the session's end time is already in the past **at the moment the row is
     first created**, set `render_status` and `post_status` to `skipped`.
     Otherwise set both to `pending`.
4. For existing `pending` rows, refresh `session_date` and `openf1_session_key`
   if the OpenF1 calendar changed (F1 reschedules races mid-season). Never touch
   a row whose status is not `pending`.

On the very first run (2026-05-22), this writes one row per 2026 Race and Sprint.
The 7 finished races and any finished sprints land as `skipped`; the Canadian GP
(2026-05-23/24) and everything after land as `pending`. Each subsequent season,
the new calendar's sessions are appended automatically as `pending` — the sheet
maintains itself across years.

## Producer changes (`race_replay.py`)

The render code and the self-scheduling cron logic are unchanged. The additions:

### New: manifest module

A set of functions (in `race_replay.py` or a small `manifest.py` imported by it):

- `get_sheets_service()` — authenticated Google Sheets API client, reusing the
  existing `GDRIVE_*` OAuth credentials (the OAuth token must now also carry the
  `spreadsheets` scope — see Setup).
- `read_manifest(service)` — returns all rows as a list of dicts keyed by header.
- `ensure_manifest(service, sessions, now)` — the seeding step above.
- `update_render_row(service, session_id, **fields)` — updates only the named
  render cells of one row, located by matching `session_id`.

New constant `MANIFEST_SHEET_ID`, read from an environment variable of the same
name (a new GitHub secret).

### Changed: orchestration

`main()` gains a Phase 0 and a changed Phase 2:

- **Phase 0 — `ensure_manifest`.** Runs every invocation.
- **Phase 1 — reschedule.** Unchanged.
- **Phase 2 — render due sessions.** For each due session:
  - Look up its manifest row. If `render_status` is `rendered` or `skipped`,
    skip. If the row is missing, skip and log (Phase 0 will create it next run).
  - Defensive adoption: if the row is `pending`/`error` but a Drive file with the
    expected name already exists, adopt it — set `render_status = rendered` with
    that file id, do not re-render.
  - Otherwise render. If the OpenF1 data is not ready, leave the row `pending`
    and move on (a later cron slot retries).
  - **Validate the output** before uploading: the file must exist and be larger
    than 1 MB (a real 30-second 1080×1920 video is tens of MB; anything smaller
    is a broken render). If invalid, set `render_status = error` with a note and
    do not upload.
  - On a valid render: upload to Drive, then
    `update_render_row(session_id, render_status="rendered", rendered_at=now,
    drive_file_id=id, render_notes="")`.
  - On any exception: `render_status = error`, `render_notes = <reason>`.

The manifest row replaces `drive_has_file` as the authoritative "already done"
marker. `drive_has_file` is kept only for the defensive-adoption check above.

### New: off-season heartbeat

GitHub disables a scheduled workflow after 60 days with no repository commits.
During the season the producer commits cron updates often, but the winter break
is longer than 60 days. Fix: the weekly safety-tick run always updates and
commits a `.pipeline_heartbeat` file (a UTC timestamp). The weekly tick cron is
always present, so this guarantees at least one commit per week year-round.

### New: failure alert

The workflow gains a final step that runs `if: failure()` and emails
`info@formula-neon.com` so a red producer run is never invisible.

## Poster changes (n8n workflow — supersedes V3)

The Drive trigger, the hourly Drive sweep, the custom-property tracking, and the
30-minute age window are all **removed**. They are replaced by manifest polling.
The result is one trigger and one mechanism.

### Nodes

1. **Schedule Trigger** — every 15 minutes.
2. **Read Manifest** — Google Sheets node, read all rows of the `sessions` tab.
3. **Pick Row To Post** (Code) — choose one eligible row, or end the run.
   Eligible = `render_status` is `rendered` **and** one of:
   - `post_status` is `pending`; or
   - `post_status` is `error` and `attempts` < 3; or
   - `post_status` is `posting` and `post_started_at` is more than 30 minutes ago
     (a crashed earlier run).
   If several are eligible, pick the oldest by `session_date`. One row per run
   keeps each run simple and naturally rate-limits Zernio.
4. **Claim Row** — Google Sheets update: set `post_status = posting`,
   `post_started_at = now`. Claiming *before* doing any work is the in-progress
   marker that prevents double-posting.
5. **Download Video** — Google Drive download by `drive_file_id`.
6. **OpenF1 enrichment** — using `openf1_session_key` from the row: fetch
   `meetings`, `session_result`, and `drivers`. (The row already carries the
   session key, so the "find the session" step from V3 is gone.)
7. **Build Race Summary** (Code) — combine into a confirmed-results block.
8. **Generate Caption** — Gemini, JSON output.
9. **Parse Caption** (Code) — JSON parse; strip em dashes.
10. **Post to Zernio** — HTTP, multipart (video + caption). *See Open Item.*
11. **Mark Posted** — Google Sheets update: `post_status = posted`,
    `posted_at = now`, `post_link`, `post_notes = ""`.

On failure at any step, the workflow's error path updates the row to
`post_status = error`, increments `attempts`, writes the reason to `post_notes`,
and the error workflow emails `info@formula-neon.com`.

Every HTTP and Sheets node has **Retry On Fail** enabled (3 attempts) to ride out
transient blips. The workflow is set to **not run concurrently** in n8n.

The exported workflow JSON is committed to the repo (`n8n/` directory) as a
version-controlled backup, with the n8n version it was built on recorded.

## Data flow, end to end

```
1. (once) First producer run seeds the sheet — ~30 rows (24 races + 6 sprints).
          Finished sessions (7 races + any finished sprints) = skipped.
          Rest = pending.

2. A race ends. The producer's self-scheduled cron fires, sees the row is
   "pending", confirms OpenF1 data is ready, renders the video, validates it,
   uploads it to Drive, and sets the row to "rendered" + drive_file_id.

3. Within 15 minutes the poster's poll finds a "rendered, not posted" row,
   claims it ("posting"), downloads the video by its file id, builds the
   caption from OpenF1 + Gemini, posts via Zernio, and sets the row to
   "posted" + posted_at + post_link.

4. The owner opens the sheet any time and sees the whole season's status.
```

## Error handling and robustness

Every row in this table is either handled by the design or is a documented
human-maintenance item. This is the system's complete known risk surface.

| Risk | Handling |
|---|---|
| A render fails or the job times out | Row stays `pending`; the next cron slot retries. One bad run loses nothing. |
| Render produces a broken/tiny file | Producer validates file size > 1 MB before uploading; a bad file becomes `error`, never posted. |
| OpenF1 data not ready when the producer runs | Row stays `pending`; later cron slots retry until data appears. |
| OpenF1 rate-limits or has a brief outage | `openf1_get` retries with backoff on 429/5xx; a longer outage just delays the row, never loses it. |
| OpenF1 changes its schema | All OpenF1 calls live in one documented place; a future fix is one spot. Defensive field access. |
| A whole weekend is missed | The weekly safety-tick re-derives the schedule and recovers it. |
| Poster crashes mid-post | The `posting` claim + 30-minute staleness check means the row is neither double-posted nor stuck. |
| Transient API blip (Gemini, Zernio, Sheets, Drive) | Retry On Fail (3×) on every external node. |
| Persistent post failure | Row goes `error`, `attempts` increments, retried up to 3×, then left visible; an alert is emailed. |
| Owner reorders/adds columns in the sheet | Code addresses columns by header name, never position. |
| Owner accidentally damages the sheet | Google Sheets keeps automatic version history; the header row and status columns are protected ranges. |
| Producer and poster write the same row at once | They write disjoint column sets, cell-scoped updates — no clobbering. |
| Sheet is wiped entirely | Re-seed restores rows (past sessions as `skipped`); for full recovery, restore via Sheets version history. |
| GitHub disables the workflow in the off-season | Weekly heartbeat commit keeps the repository active. |
| Producer or poster fails silently | Failure emails on both halves to `info@formula-neon.com`. |
| Drive folder filling up over years | Documented maintenance item; ~2 GB/year. Add a cleanup step later or buy storage. |
| n8n offline (n8n Cloud outage) | The manifest preserves state; the poster resumes and catches up. Nothing is lost. |
| All state | Lives in the sheet — rebuilding n8n, or touching Drive, never loses pipeline state. |

### Yearly maintenance runbook (human items)

These cannot be designed away — they go in the runbook:

1. **Google login token** — if Drive/Sheets access fails, regenerate the OAuth
   refresh token with `get_gdrive_token.py`. The consent screen must stay in
   **Production** mode or the token expires every 7 days.
2. **Team colours** — `get_constructor_color` in `race_replay.py` has a 2026
   team table. When F1 teams change, update it, or new teams render grey.
3. **Gemini model name** — AI models are retired every year or two. When captions
   fail with a model error, update the one model-name constant.
4. **Zernio social connections** — Instagram/YouTube drop their connections every
   few months. Reconnect them in Zernio when posts start failing.
5. **GitHub token (`WORKFLOW_PAT`)** — must be created with **no expiry**, or
   scheduling silently freezes the day it expires.

## Setup / bootstrap (one-time, by the owner)

1. Create the Google Sheet with a tab named `sessions` and the header row from
   the schema above. Note its sheet ID.
2. Add the `spreadsheets` scope to the producer's Google OAuth and regenerate the
   refresh token with `get_gdrive_token.py`. Confirm the consent screen is in
   Production mode.
3. Add GitHub secret `MANIFEST_SHEET_ID`. Confirm `WORKFLOW_PAT` has no expiry.
4. In n8n Cloud, create credentials: Google Sheets, Google Drive, Gemini
   (header auth), Zernio (header auth).
5. Confirm the n8n Drive folder ID equals the producer's `GDRIVE_FOLDER_ID`.
6. Trigger one producer run manually; confirm the sheet is seeded (~30 rows, the
   finished sessions marked `skipped`).
7. Import the poster workflow into n8n Cloud and activate it.
8. Verify the Zernio API details and finalize the post node (see Open Item).

## Secrets

| Secret | Where | Purpose |
|---|---|---|
| `GDRIVE_REFRESH_TOKEN` | GitHub + n8n | Google auth (Drive + Sheets), Production consent screen |
| `GDRIVE_CLIENT_ID` | GitHub | Google OAuth client |
| `GDRIVE_CLIENT_SECRET` | GitHub | Google OAuth client |
| `GDRIVE_FOLDER_ID` | GitHub + n8n | Target Drive folder for the MP4s |
| `MANIFEST_SHEET_ID` | GitHub + n8n | The manifest Google Sheet |
| `WORKFLOW_PAT` | GitHub | Lets the producer push its own cron edits — **no expiry** |
| Gemini API key | n8n credential | Caption generation |
| Zernio API key | n8n credential | Posting to Instagram + YouTube |

## Testing strategy

- **Producer unit tests** — extend the existing suite for the new pure logic:
  the seed decision (is a session past at creation time?), manifest-row parsing,
  and `session_id` construction.
- **Seeding dry run** — a producer flag that prints the rows it would add without
  writing to the sheet.
- **End-to-end producer** — `--test-session` renders a past session; confirm a
  manifest row is updated correctly (against a test sheet, not the live one).
- **Poster** — run against a test sheet with a hand-made `rendered` row; confirm
  it posts and writes back `posted`.
- **Real-runner check** — verify the producer on an actual GitHub runner before
  declaring done (the FastF1 block was only ever caught on a real runner).

## Open item

**Zernio API is unverified.** The endpoint, authentication, and request body for
the post step are assumptions carried from the V3 build doc. The poster's post
node is built **last**, against Zernio's real API documentation. Zernio's monthly
post allowance must also be confirmed to cover a full season (~30 sessions, more
once qualifying is added).

## Documentation cleanup

On implementation, the scattered and superseded docs are consolidated:

- This spec becomes the **authoritative design**.
- A plain-language **runbook** is written for the owner (the maintenance items,
  the setup steps, "what to do if X").
- `N8N_WORKFLOW_BUILD.md`, `N8N_WORKFLOW_V2.md`, `N8N_WORKFLOW_V3.md`, and
  `FORMULYTICS_PIPELINE.md` are archived or deleted — they are superseded and
  their continued presence is itself a source of confusion. `RACE_PIPELINE.md`
  is updated to reference the manifest.
