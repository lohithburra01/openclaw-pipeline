# Producer + Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the F1 race-replay producer seed and maintain a Google Sheet "season manifest", and drive its render decisions from that manifest instead of scanning Google Drive.

**Architecture:** Extend the existing single-file `race_replay.py` (the codebase deliberately uses one self-contained file — follow that pattern). New pure functions (manifest diffing, seeding, validation, heartbeat) are unit-tested with pytest; new Google Sheets I/O is integration-verified. The producer gains a Phase 0 that seeds/maintains the manifest, the render loop reads and writes manifest rows, and a weekly heartbeat commit keeps the scheduled workflow alive through the off-season.

**Tech Stack:** Python 3.11, Google Sheets API v4 (`google-api-python-client`, already installed), OpenF1 API, GitHub Actions, pytest.

**Scope:** This is Plan 1 of 2. It covers the producer and the manifest — working, testable software on its own (the producer seeds and maintains the sheet). Plan 2 (the n8n poster) is written separately once Zernio's API is verified.

**Spec:** `docs/superpowers/specs/2026-05-22-pipeline-manifest-design.md`

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `race_replay.py` | modify | All producer + manifest logic. New functions go in a new `MANIFEST & HEALTH` section; `output_slug`, `get_drive_service`, `drive_has_file`, `list_sessions`, `render_due`, `main` are modified in place. |
| `tests/test_race_replay.py` | modify | 19 new unit tests for the new pure functions, appended to the existing 24. |
| `.github/workflows/race_replay.yml` | modify | Pass `MANIFEST_SHEET_ID`; add a failure-email step. |
| `get_gdrive_token.py` | modify | Add the Google Sheets scope so the regenerated token can write the sheet. |
| `.pipeline_heartbeat` | create (by code) | A committed timestamp file; its weekly commit keeps the repo active. |
| `RUNBOOK.md` | create | Plain-language operating guide for the owner. |
| `docs/archive/` | create | Destination for superseded docs. |
| `RACE_PIPELINE.md` | modify | Add a pointer to the manifest spec. |

The Google Sheet manifest itself is not a repo file — it is created by the owner (see Manual Steps) and populated by the code.

### Manifest column schema (used throughout)

The sheet's tab is named `sessions`. Row 1 is the header. Columns, in this exact order:

| # | Header | Written by |
|---|--------|-----------|
| A | `session_id` | seed |
| B | `gp_name` | seed |
| C | `session` | seed |
| D | `session_date` | seed |
| E | `openf1_session_key` | seed |
| F | `render_status` | producer |
| G | `rendered_at` | producer |
| H | `drive_file_id` | producer |
| I | `render_notes` | producer |
| J | `post_status` | poster (Plan 2) |
| K | `post_started_at` | poster (Plan 2) |
| L | `posted_at` | poster (Plan 2) |
| M | `post_link` | poster (Plan 2) |
| N | `attempts` | poster (Plan 2) |
| O | `post_notes` | poster (Plan 2) |

The producer writes only the contiguous block **F:I**. This plan never writes J:O.

---

## Task 1: Pure ID and formatting helpers

**Files:**
- Modify: `race_replay.py` (replace `output_slug`, add helpers next to it)
- Test: `tests/test_race_replay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_race_replay.py`:
```python
def test_session_id_for():
    assert rr.session_id_for("Monaco", 2026, "R") == "monaco_2026_R"
    assert rr.session_id_for("Las Vegas", 2026, "S") == "las_vegas_2026_S"


def test_session_name_for():
    assert rr.session_name_for("R") == "Race"
    assert rr.session_name_for("S") == "Sprint"


def test_fmt_dt():
    assert rr.fmt_dt(datetime(2026, 5, 24, 22, 15, 30)) == "2026-05-24 22:15:30"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_race_replay.py::test_session_id_for -q`
Expected: FAIL — `AttributeError: module 'race_replay' has no attribute 'session_id_for'`.

- [ ] **Step 3: Implement the helpers**

In `race_replay.py`, find the existing `output_slug` function:
```python
def output_slug(event_name, year, kind):
    """Stable output filename, e.g. 'canadian_grand_prix_2026_R.mp4'."""
    base = re.sub(r"[^a-z0-9]+", "_", event_name.lower()).strip("_")
    return f"{base}_{year}_{kind}.mp4"
```
Replace that whole function with:
```python
def session_id_for(event_name, year, kind):
    """Stable session id, e.g. 'monaco_2026_R'. Matches the Drive filename
    stem and is the manifest's primary key."""
    base = re.sub(r"[^a-z0-9]+", "_", str(event_name).lower()).strip("_")
    return f"{base}_{year}_{kind}"


def output_slug(event_name, year, kind):
    """Stable output filename, e.g. 'monaco_2026_R.mp4'."""
    return session_id_for(event_name, year, kind) + ".mp4"


def session_name_for(kind):
    """Session kind 'R'/'S' -> human-readable manifest 'session' value."""
    return "Sprint" if kind == "S" else "Race"


def fmt_dt(dt):
    """A datetime -> manifest timestamp string 'YYYY-MM-DD HH:MM:SS'."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def utcnow_str():
    """Current UTC time as a manifest timestamp string."""
    return fmt_dt(datetime.utcnow())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_race_replay.py -q`
Expected: PASS — 27 passed (the 24 existing tests, including `test_output_slug`, still pass plus the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add race_replay.py tests/test_race_replay.py
git commit -m "feat: add session_id and timestamp helpers"
```

---

## Task 2: Google credentials refactor, Sheets client, and drive_find_file

**Files:**
- Modify: `race_replay.py` (`get_drive_service`, `drive_has_file`)
- Modify: `get_gdrive_token.py`

These functions perform network I/O and are verified by a clean import plus the live integration check in Task 8 — not unit tests.

- [ ] **Step 1: Refactor the Google service builders**

In `race_replay.py`, find `get_drive_service`:
```python
def get_drive_service():
    """Authenticated Google Drive v3 client, using the GDRIVE_* env vars."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = Credentials(
        token=None,
        refresh_token=os.environ["GDRIVE_REFRESH_TOKEN"],
        client_id=os.environ["GDRIVE_CLIENT_ID"],
        client_secret=os.environ["GDRIVE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds)
```
Replace that whole function with:
```python
def get_google_credentials():
    """Refreshed Google OAuth credentials from the GDRIVE_* env vars. The
    refresh token must carry both the Drive and Spreadsheets scopes."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    creds = Credentials(
        token=None,
        refresh_token=os.environ["GDRIVE_REFRESH_TOKEN"],
        client_id=os.environ["GDRIVE_CLIENT_ID"],
        client_secret=os.environ["GDRIVE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(Request())
    return creds


def get_drive_service():
    """Authenticated Google Drive v3 client."""
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=get_google_credentials())


def get_sheets_service():
    """Authenticated Google Sheets v4 client."""
    from googleapiclient.discovery import build
    return build("sheets", "v4", credentials=get_google_credentials())
```

- [ ] **Step 2: Refactor `drive_has_file` into `drive_find_file`**

In `race_replay.py`, find `drive_has_file`:
```python
def drive_has_file(service, folder_id, filename):
    """True if a non-trashed file named `filename` exists in the folder."""
    safe = filename.replace("\\", "\\\\").replace("'", "\\'")
    query = f"name = '{safe}' and '{folder_id}' in parents and trashed = false"
    resp = service.files().list(q=query, fields="files(id,name)").execute()
    return len(resp.get("files", [])) > 0
```
Replace that whole function with:
```python
def drive_find_file(service, folder_id, filename):
    """Return the id of a non-trashed file named `filename` in the folder,
    or None if there is no such file."""
    safe = filename.replace("\\", "\\\\").replace("'", "\\'")
    query = f"name = '{safe}' and '{folder_id}' in parents and trashed = false"
    resp = service.files().list(q=query, fields="files(id,name)").execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def drive_has_file(service, folder_id, filename):
    """True if a non-trashed file named `filename` exists in the folder."""
    return drive_find_file(service, folder_id, filename) is not None
```

- [ ] **Step 3: Add the Sheets scope to the token helper**

In `get_gdrive_token.py`, find:
```python
SCOPES = ["https://www.googleapis.com/auth/drive"]
```
Replace with:
```python
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]
```

- [ ] **Step 4: Verify the module still imports**

Run: `python -c "import race_replay; print('OK', hasattr(race_replay,'get_sheets_service'), hasattr(race_replay,'drive_find_file'))"`
Expected: `OK True True` (no traceback).

- [ ] **Step 5: Run the test suite (nothing should have broken)**

Run: `python -m pytest tests/test_race_replay.py -q`
Expected: PASS — 27 passed.

- [ ] **Step 6: Commit**

```bash
git add race_replay.py get_gdrive_token.py
git commit -m "feat: add Google Sheets client and Drive file-id lookup"
```

---

## Task 3: Manifest schema constants and pure logic

**Files:**
- Modify: `race_replay.py` (add the `MANIFEST & HEALTH` section; modify `list_sessions`)
- Test: `tests/test_race_replay.py`

- [ ] **Step 1: Add the manifest constants**

In `race_replay.py`, find the line `OPENF1_BASE = "https://api.openf1.org/v1"` and add immediately after it:
```python

# --- MANIFEST (Google Sheet) ---
MANIFEST_TAB = "sessions"
MANIFEST_HEADERS = [
    "session_id", "gp_name", "session", "session_date", "openf1_session_key",
    "render_status", "rendered_at", "drive_file_id", "render_notes",
    "post_status", "post_started_at", "posted_at", "post_link", "attempts",
    "post_notes",
]
```

- [ ] **Step 2: Add `meeting_key` to `list_sessions` output**

In `race_replay.py`, find this block inside `list_sessions`:
```python
        sessions.append({
            "event": s.get("location") or s.get("circuit_short_name"),
            "kind": kind,
            "start": start,
            "end": session_end_time(start, kind),
            "session_key": int(s["session_key"]),
        })
```
Replace it with:
```python
        sessions.append({
            "event": s.get("location") or s.get("circuit_short_name"),
            "kind": kind,
            "start": start,
            "end": session_end_time(start, kind),
            "session_key": int(s["session_key"]),
            "meeting_key": s.get("meeting_key"),
        })
```

- [ ] **Step 3: Write the failing tests**

Append to `tests/test_race_replay.py`:
```python
def test_seed_status_for():
    now = datetime(2026, 5, 22, 12, 0)
    assert rr.seed_status_for(datetime(2026, 5, 10, 0, 0), now) == "skipped"
    assert rr.seed_status_for(datetime(2026, 6, 1, 0, 0), now) == "pending"


def _manifest_session(event, kind, start, end, key=99, meeting_key=7,
                       gp_name="Test Grand Prix"):
    return {"event": event, "kind": kind, "start": start, "end": end,
            "session_key": key, "meeting_key": meeting_key, "gp_name": gp_name}


def test_build_manifest_row_future_session():
    now = datetime(2026, 5, 22, 12, 0)
    s = _manifest_session("Montreal", "R", datetime(2026, 5, 24, 18, 0),
                          datetime(2026, 5, 24, 20, 15), gp_name="Canadian Grand Prix")
    row = rr.build_manifest_row(s, now)
    assert len(row) == len(rr.MANIFEST_HEADERS)
    assert row[0] == "montreal_2026_R"      # session_id
    assert row[1] == "Canadian Grand Prix"  # gp_name
    assert row[2] == "Race"                 # session
    assert row[3] == "2026-05-24"           # session_date
    assert row[5] == "pending"              # render_status
    assert row[9] == "pending"              # post_status
    assert row[13] == "0"                   # attempts


def test_build_manifest_row_past_session_is_skipped():
    now = datetime(2026, 5, 22, 12, 0)
    s = _manifest_session("Miami", "R", datetime(2026, 5, 3, 19, 0),
                          datetime(2026, 5, 3, 21, 15))
    row = rr.build_manifest_row(s, now)
    assert row[5] == "skipped"
    assert row[9] == "skipped"


def test_attach_gp_names():
    sessions = [{"event": "Montreal", "meeting_key": 1}]
    meetings = [{"meeting_key": 1, "meeting_name": "Canadian Grand Prix"}]
    out = rr.attach_gp_names(sessions, meetings)
    assert out[0]["gp_name"] == "Canadian Grand Prix"


def test_attach_gp_names_fallback_to_event():
    sessions = [{"event": "Montreal", "meeting_key": 999}]
    out = rr.attach_gp_names(sessions, [])
    assert out[0]["gp_name"] == "Montreal"


def test_manifest_plan_appends_missing_session():
    now = datetime(2026, 5, 22, 12, 0)
    sessions = [_manifest_session("Montreal", "R", datetime(2026, 5, 24, 18, 0),
                                  datetime(2026, 5, 24, 20, 15))]
    appends, refreshes = rr.manifest_plan([], sessions, now)
    assert len(appends) == 1
    assert appends[0][0] == "montreal_2026_R"
    assert refreshes == []


def test_manifest_plan_skips_session_with_existing_row():
    now = datetime(2026, 5, 22, 12, 0)
    sessions = [_manifest_session("Montreal", "R", datetime(2026, 5, 24, 18, 0),
                                  datetime(2026, 5, 24, 20, 15))]
    existing = [{"session_id": "montreal_2026_R", "render_status": "rendered",
                 "session_date": "2026-05-24", "openf1_session_key": "99", "_row": 2}]
    appends, refreshes = rr.manifest_plan(existing, sessions, now)
    assert appends == []
    assert refreshes == []


def test_manifest_plan_refreshes_pending_row_when_date_moved():
    now = datetime(2026, 5, 22, 12, 0)
    sessions = [_manifest_session("Montreal", "R", datetime(2026, 5, 25, 18, 0),
                                  datetime(2026, 5, 25, 20, 15))]
    existing = [{"session_id": "montreal_2026_R", "render_status": "pending",
                 "session_date": "2026-05-24", "openf1_session_key": "99", "_row": 2}]
    appends, refreshes = rr.manifest_plan(existing, sessions, now)
    assert appends == []
    assert refreshes == [(2, "2026-05-25", "99")]
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python -m pytest tests/test_race_replay.py::test_seed_status_for -q`
Expected: FAIL — `AttributeError: ... has no attribute 'seed_status_for'`.

- [ ] **Step 5: Implement the pure manifest logic**

In `race_replay.py`, find the line `# SCHEDULING & ORCHESTRATION` section header and, immediately **above** `def main():`, add a new section:
```python

# ============================================================
# MANIFEST & HEALTH
# ============================================================
def seed_status_for(session_end, now):
    """A session whose end time is already past when its row is first created
    is 'skipped'; everything still upcoming is 'pending'."""
    return "skipped" if session_end <= now else "pending"


def build_manifest_row(session, now):
    """Build a full manifest row (list of values in MANIFEST_HEADERS order)
    for a freshly-seeded session."""
    status = seed_status_for(session["end"], now)
    return [
        session_id_for(session["event"], SEASON, session["kind"]),  # session_id
        session.get("gp_name", session["event"]),                   # gp_name
        session_name_for(session["kind"]),                          # session
        session["start"].strftime("%Y-%m-%d"),                      # session_date
        str(session["session_key"]),                                # openf1_session_key
        status,                                                     # render_status
        "",                                                         # rendered_at
        "",                                                         # drive_file_id
        "",                                                         # render_notes
        status,                                                     # post_status
        "",                                                         # post_started_at
        "",                                                         # posted_at
        "",                                                         # post_link
        "0",                                                        # attempts
        "",                                                         # post_notes
    ]


def attach_gp_names(sessions, meetings):
    """Add a 'gp_name' key to each session by joining on meeting_key. Falls
    back to the session's location when no meeting matches."""
    by_key = {m.get("meeting_key"): m.get("meeting_name") for m in meetings}
    for s in sessions:
        s["gp_name"] = by_key.get(s.get("meeting_key")) or s["event"]
    return sessions


def manifest_plan(existing_rows, sessions, now):
    """Pure diff between the manifest and the calendar.

    Returns (appends, refreshes):
      - appends: full rows (MANIFEST_HEADERS order) for sessions with no row.
      - refreshes: (row_number, session_date, openf1_session_key) tuples for
        existing 'pending' rows whose schedule fields have changed.
    Never modifies the status of an existing row.
    """
    by_id = {r["session_id"]: r for r in existing_rows}
    appends, refreshes = [], []
    for s in sessions:
        sid = session_id_for(s["event"], SEASON, s["kind"])
        row = by_id.get(sid)
        if row is None:
            appends.append(build_manifest_row(s, now))
            continue
        if row.get("render_status") == "pending":
            new_date = s["start"].strftime("%Y-%m-%d")
            new_key = str(s["session_key"])
            if (row.get("session_date") != new_date
                    or row.get("openf1_session_key") != new_key):
                refreshes.append((row["_row"], new_date, new_key))
    return appends, refreshes
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_race_replay.py -q`
Expected: PASS — 35 passed.

- [ ] **Step 7: Commit**

```bash
git add race_replay.py tests/test_race_replay.py
git commit -m "feat: add manifest schema and pure seeding logic"
```

---

## Task 4: Manifest Google Sheets I/O

**Files:**
- Modify: `race_replay.py` (add to the `MANIFEST & HEALTH` section)
- Test: `tests/test_race_replay.py`

The value-parsing logic is pure and unit-tested; the network functions are verified by import here and by the live check in Task 8.

- [ ] **Step 1: Write the failing tests for the pure parser**

Append to `tests/test_race_replay.py`:
```python
def test_parse_manifest_values_empty():
    assert rr.parse_manifest_values([]) == []


def test_parse_manifest_values_keys_and_row_numbers():
    values = [
        ["session_id", "render_status"],
        ["monaco_2026_R", "pending"],
        ["miami_2026_R", "rendered"],
    ]
    rows = rr.parse_manifest_values(values)
    assert rows[0] == {"session_id": "monaco_2026_R",
                       "render_status": "pending", "_row": 2}
    assert rows[1]["_row"] == 3


def test_parse_manifest_values_pads_short_rows():
    values = [["session_id", "render_status", "drive_file_id"],
              ["monaco_2026_R", "pending"]]
    rows = rr.parse_manifest_values(values)
    assert rows[0]["drive_file_id"] == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_race_replay.py::test_parse_manifest_values_empty -q`
Expected: FAIL — `AttributeError: ... has no attribute 'parse_manifest_values'`.

- [ ] **Step 3: Implement the manifest I/O functions**

Append to the `MANIFEST & HEALTH` section of `race_replay.py`:
```python
def parse_manifest_values(values):
    """Raw sheet values (first row = headers) -> list of row dicts keyed by
    header, each with a 1-based '_row' sheet row number. Short rows are
    padded with empty strings."""
    if not values:
        return []
    headers = values[0]
    rows = []
    for i, raw in enumerate(values[1:], start=2):
        record = {h: (raw[j] if j < len(raw) else "") for j, h in enumerate(headers)}
        record["_row"] = i
        rows.append(record)
    return rows


def load_meetings(season):
    """Return the OpenF1 meetings list for a season."""
    return openf1_get(f"meetings?year={season}")


def read_manifest(service, sheet_id):
    """Read the manifest tab into a list of row dicts (see parse_manifest_values)."""
    resp = service.spreadsheets().values().get(
        spreadsheetId=sheet_id, range=MANIFEST_TAB
    ).execute()
    return parse_manifest_values(resp.get("values", []))


def update_render_row(service, sheet_id, row_number,
                      render_status, rendered_at, drive_file_id, render_notes):
    """Write the four producer-owned columns (F:I) of one manifest row."""
    service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=f"{MANIFEST_TAB}!F{row_number}:I{row_number}",
        valueInputOption="RAW",
        body={"values": [[render_status, rendered_at, drive_file_id, render_notes]]},
    ).execute()


def ensure_manifest(service, sheet_id, sessions, now):
    """Seed/maintain the manifest. Writes the header row if the sheet is
    empty, appends rows for new sessions, and refreshes the schedule fields
    of pending rows. Append-only with respect to status."""
    raw = service.spreadsheets().values().get(
        spreadsheetId=sheet_id, range=MANIFEST_TAB
    ).execute().get("values", [])

    if not raw:
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id, range=f"{MANIFEST_TAB}!A1",
            valueInputOption="RAW", body={"values": [MANIFEST_HEADERS]},
        ).execute()
        existing = []
    else:
        existing = parse_manifest_values(raw)

    appends, refreshes = manifest_plan(existing, sessions, now)

    if appends:
        service.spreadsheets().values().append(
            spreadsheetId=sheet_id, range=MANIFEST_TAB,
            valueInputOption="RAW", insertDataOption="INSERT_ROWS",
            body={"values": appends},
        ).execute()

    if refreshes:
        data = [
            {"range": f"{MANIFEST_TAB}!D{row}:E{row}", "values": [[date, key]]}
            for row, date, key in refreshes
        ]
        service.spreadsheets().values().batchUpdate(
            spreadsheetId=sheet_id,
            body={"valueInputOption": "RAW", "data": data},
        ).execute()

    print(f"  manifest: {len(appends)} rows added, {len(refreshes)} refreshed")
    return appends, refreshes
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_race_replay.py -q`
Expected: PASS — 38 passed.

- [ ] **Step 5: Verify the module imports cleanly**

Run: `python -c "import race_replay as r; print('OK', all(hasattr(r,n) for n in ['read_manifest','update_render_row','ensure_manifest','load_meetings']))"`
Expected: `OK True`.

- [ ] **Step 6: Commit**

```bash
git add race_replay.py tests/test_race_replay.py
git commit -m "feat: add manifest sheet read/seed/update I/O"
```

---

## Task 5: Render validation and the off-season heartbeat

**Files:**
- Modify: `race_replay.py` (add to the `MANIFEST & HEALTH` section)
- Test: `tests/test_race_replay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_race_replay.py`:
```python
def test_is_valid_render_rejects_missing(tmp_path):
    assert rr.is_valid_render(str(tmp_path / "nope.mp4")) is False


def test_is_valid_render_rejects_tiny(tmp_path):
    p = tmp_path / "tiny.mp4"
    p.write_bytes(b"x" * 100)
    assert rr.is_valid_render(str(p)) is False


def test_is_valid_render_accepts_large(tmp_path):
    p = tmp_path / "big.mp4"
    p.write_bytes(b"x" * (rr.MIN_RENDER_BYTES + 1))
    assert rr.is_valid_render(str(p)) is True


def test_heartbeat_due_when_no_prior():
    assert rr.heartbeat_due("", datetime(2026, 5, 22, 12, 0)) is True


def test_heartbeat_due_when_stale():
    assert rr.heartbeat_due("2026-05-10 12:00:00", datetime(2026, 5, 22, 12, 0)) is True


def test_heartbeat_not_due_when_fresh():
    assert rr.heartbeat_due("2026-05-21 12:00:00", datetime(2026, 5, 22, 12, 0)) is False


def test_heartbeat_due_when_unparseable():
    assert rr.heartbeat_due("garbage", datetime(2026, 5, 22, 12, 0)) is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_race_replay.py::test_is_valid_render_rejects_missing -q`
Expected: FAIL — `AttributeError: ... has no attribute 'is_valid_render'`.

- [ ] **Step 3: Implement validation and heartbeat**

First, add two constants. In `race_replay.py`, find the `MANIFEST_HEADERS` list (added in Task 3) and add immediately after its closing `]`:
```python
HEARTBEAT_FILE = ".pipeline_heartbeat"
HEARTBEAT_INTERVAL_DAYS = 5
MIN_RENDER_BYTES = 1_000_000
```

Then append to the `MANIFEST & HEALTH` section:
```python
def is_valid_render(path):
    """True if `path` is a plausibly-complete video file: it exists and is
    larger than MIN_RENDER_BYTES. Guards against a broken or empty render."""
    return os.path.exists(path) and os.path.getsize(path) > MIN_RENDER_BYTES


def heartbeat_due(last_str, now):
    """True if the heartbeat should be refreshed: there is no prior heartbeat,
    it cannot be parsed, or it is older than HEARTBEAT_INTERVAL_DAYS."""
    if not last_str:
        return True
    try:
        last = datetime.strptime(last_str.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return True
    return (now - last) >= timedelta(days=HEARTBEAT_INTERVAL_DAYS)


def read_heartbeat():
    """Return the timestamp string in HEARTBEAT_FILE, or '' if it is absent."""
    if not os.path.exists(HEARTBEAT_FILE):
        return ""
    with open(HEARTBEAT_FILE, encoding="utf-8") as f:
        return f.read().strip()


def heartbeat(now):
    """If the heartbeat is stale, rewrite and commit it. The commit keeps the
    repository active so GitHub does not disable the scheduled workflow during
    the off-season."""
    if not heartbeat_due(read_heartbeat(), now):
        print("Heartbeat fresh; no commit needed.")
        return
    with open(HEARTBEAT_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(fmt_dt(now) + "\n")
    commit_and_push(HEARTBEAT_FILE, "chore: pipeline heartbeat")
    print("Heartbeat updated.")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_race_replay.py -q`
Expected: PASS — 45 passed.

- [ ] **Step 5: Commit**

```bash
git add race_replay.py tests/test_race_replay.py
git commit -m "feat: add render validation and off-season heartbeat"
```

---

## Task 6: Wire the manifest into the producer

**Files:**
- Modify: `race_replay.py` (`render_due`, `main`)

This task changes orchestration; it is verified by the full suite plus the `--dry-run` and live checks in Task 8.

- [ ] **Step 1: Replace `render_due`**

In `race_replay.py`, find the entire `render_due` function (it begins `def render_due(now):` and ends just before `def main():`). Replace the whole function with:
```python
def render_due(now):
    """Phase 2: render any just-ended session that the manifest says is not
    yet done, then record the result back to the manifest."""
    sessions = list_sessions(load_event_schedule(SEASON))
    candidates = due_sessions(sessions, now)
    if not candidates:
        print("NO_SESSION")
        return

    drive = get_drive_service()
    sheets = get_sheets_service()
    folder = os.environ["GDRIVE_FOLDER_ID"]
    sheet_id = os.environ["MANIFEST_SHEET_ID"]
    rows_by_id = {r["session_id"]: r for r in read_manifest(sheets, sheet_id)}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for s in candidates:
        sid = session_id_for(s["event"], SEASON, s["kind"])
        row = rows_by_id.get(sid)
        if row is None:
            print(f"  NO_MANIFEST_ROW ({sid}); it will be seeded next run")
            continue
        if row["render_status"] in ("rendered", "skipped"):
            print(f"  SKIP ({sid}, render_status={row['render_status']})")
            continue

        slug = sid + ".mp4"

        # Defensive adoption: a previous run may have uploaded the file but
        # crashed before recording it. Adopt the existing file instead of
        # re-rendering.
        existing_id = drive_find_file(drive, folder, slug)
        if existing_id:
            print(f"  ADOPT ({slug} already in Drive)")
            update_render_row(sheets, sheet_id, row["_row"],
                              "rendered", utcnow_str(), existing_id, "")
            continue

        if not session_has_data(s["session_key"]):
            print(f"  DATA_NOT_READY ({sid})")
            continue

        out_path = os.path.join(OUTPUT_DIR, slug)
        try:
            create_race_timelapse(
                session_key=s["session_key"], gp=s["event"], year=SEASON,
                save_path=out_path, duration=DURATION_SECONDS, fps=FPS,
                portrait=PORTRAIT, session_label=session_label_for(s["kind"]),
            )
        except Exception as exc:
            update_render_row(sheets, sheet_id, row["_row"],
                              "error", "", "", f"render failed: {exc}"[:200])
            print(f"  RENDER_ERROR ({sid}): {exc}")
            continue

        if not is_valid_render(out_path):
            update_render_row(sheets, sheet_id, row["_row"],
                              "error", "", "", "render produced no valid video file")
            print(f"  INVALID_RENDER ({sid})")
            continue

        file_id = drive_upload_file(drive, folder, out_path)
        update_render_row(sheets, sheet_id, row["_row"],
                          "rendered", utcnow_str(), file_id, "")
        print(f"  UPLOADED ({slug}, drive id {file_id})")
```

- [ ] **Step 2: Add Phase 0 and the heartbeat to `main`**

In `race_replay.py`, find the tail of `main` (the normal-run block):
```python
    now = datetime.utcnow()
    print(f"=== race_replay.py @ {now} UTC (season {SEASON}) ===")
    reschedule(now, dry_run=args.dry_run)
    if not args.dry_run:
        render_due(now)
```
Replace it with:
```python
    now = datetime.utcnow()
    print(f"=== race_replay.py @ {now} UTC (season {SEASON}) ===")

    # Phase 0 - seed/maintain the manifest (skipped on a dry run, which writes
    # nothing anywhere).
    if not args.dry_run:
        sheets = get_sheets_service()
        sched = list_sessions(load_event_schedule(SEASON))
        sched = attach_gp_names(sched, load_meetings(SEASON))
        ensure_manifest(sheets, os.environ["MANIFEST_SHEET_ID"], sched, now)
    else:
        print("(dry-run) skipping manifest update.")

    # Phase 1 - rewrite the workflow cron from the calendar.
    reschedule(now, dry_run=args.dry_run)

    # Phase 2 - render due sessions, then refresh the off-season heartbeat.
    if not args.dry_run:
        render_due(now)
        heartbeat(now)
```

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest tests/test_race_replay.py -q`
Expected: PASS — 45 passed (the orchestration change touches no pure function, so every test still passes).

- [ ] **Step 4: Verify the module imports and the CLI still parses**

Run: `python race_replay.py --help`
Expected: usage text listing `--dry-run`, `--force-session`, `--duration`, `--test-session`. No traceback.

- [ ] **Step 5: Commit**

```bash
git add race_replay.py
git commit -m "feat: drive the producer's render decisions from the manifest"
```

---

## Task 7: GitHub Actions workflow updates

**Files:**
- Modify: `.github/workflows/race_replay.yml`

- [ ] **Step 1: Pass the manifest sheet id to the run step**

In `.github/workflows/race_replay.yml`, find:
```yaml
      - name: Run race replay pipeline
        env:
          GDRIVE_REFRESH_TOKEN: ${{ secrets.GDRIVE_REFRESH_TOKEN }}
          GDRIVE_CLIENT_ID: ${{ secrets.GDRIVE_CLIENT_ID }}
          GDRIVE_CLIENT_SECRET: ${{ secrets.GDRIVE_CLIENT_SECRET }}
          GDRIVE_FOLDER_ID: ${{ secrets.GDRIVE_FOLDER_ID }}
        run: python race_replay.py --test-session "${{ github.event.inputs.test_session }}"
```
Replace with:
```yaml
      - name: Run race replay pipeline
        env:
          GDRIVE_REFRESH_TOKEN: ${{ secrets.GDRIVE_REFRESH_TOKEN }}
          GDRIVE_CLIENT_ID: ${{ secrets.GDRIVE_CLIENT_ID }}
          GDRIVE_CLIENT_SECRET: ${{ secrets.GDRIVE_CLIENT_SECRET }}
          GDRIVE_FOLDER_ID: ${{ secrets.GDRIVE_FOLDER_ID }}
          MANIFEST_SHEET_ID: ${{ secrets.MANIFEST_SHEET_ID }}
        run: python race_replay.py --test-session "${{ github.event.inputs.test_session }}"
```

- [ ] **Step 2: Add a failure-email step**

In `.github/workflows/race_replay.yml`, find the final step:
```yaml
      - name: Upload artifact backup
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: race-replay-video
          path: output/*.mp4
          retention-days: 5
          if-no-files-found: ignore
```
Add immediately after it (same indentation, as the last step in the job):
```yaml

      - name: Email on failure
        if: failure()
        uses: dawidd6/action-send-mail@v3
        with:
          server_address: smtp.gmail.com
          server_port: 465
          secure: true
          username: ${{ secrets.SMTP_USERNAME }}
          password: ${{ secrets.SMTP_PASSWORD }}
          subject: "Formulytics producer run failed"
          to: info@formula-neon.com
          from: Formulytics Pipeline <${{ secrets.SMTP_USERNAME }}>
          body: |
            The race-replay producer workflow failed.
            Open the run log:
            ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
```

- [ ] **Step 3: Validate the YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/race_replay.yml',encoding='utf-8')); print('YAML OK')"`
Expected: `YAML OK` (if `pyyaml` is not installed, run `pip install -q pyyaml` first).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/race_replay.yml
git commit -m "feat: pass manifest sheet id and alert on producer failure"
```

---

## Task 8: Integration verification

**Files:** none changed — this task only runs and checks.

- [ ] **Step 1: Run the full unit-test suite**

Run: `python -m pytest tests/test_race_replay.py -q`
Expected: PASS — 45 passed.

- [ ] **Step 2: Verify the scheduling dry run does not touch the sheet**

Run: `python race_replay.py --dry-run`
Expected: prints `(dry-run) skipping manifest update.`, then the cron block, then `(dry-run) workflow file not modified.` Exit code 0. No `MANIFEST_SHEET_ID` is required for a dry run. Confirm `git status` shows no modified files.

- [ ] **Step 3: Live manifest seed against a TEST sheet**

This step needs the Google credentials and a throwaway test sheet — do not use the real manifest sheet. Create a blank Google Sheet, rename its tab to `sessions`, share nothing extra (the owner's account already has access), and copy its id from the URL.

Run (PowerShell, with the four `GDRIVE_*` values and the test sheet id):
```powershell
$env:GDRIVE_REFRESH_TOKEN="..."; $env:GDRIVE_CLIENT_ID="..."
$env:GDRIVE_CLIENT_SECRET="..."; $env:GDRIVE_FOLDER_ID="..."
$env:MANIFEST_SHEET_ID="<test-sheet-id>"
python -c "import race_replay as r; from datetime import datetime; s=r.get_sheets_service(); sess=r.attach_gp_names(r.list_sessions(r.load_event_schedule(2026)), r.load_meetings(2026)); r.ensure_manifest(s, '<test-sheet-id>', sess, datetime.utcnow())"
```
Expected: prints `manifest: NN rows added, 0 refreshed` where NN is about 30. Open the test sheet: row 1 is the 15 headers; every 2026 Race and Sprint has a row; sessions before today show `render_status` = `skipped`, upcoming ones show `pending`.

- [ ] **Step 4: Verify seeding is idempotent**

Run the same command from Step 3 again.
Expected: `manifest: 0 rows added, 0 refreshed` — no duplicate rows are created.

- [ ] **Step 5: Delete the test sheet**

Delete the throwaway sheet from Google Drive. No commit — this task changed no files.

---

## Task 9: Documentation — runbook and archive

**Files:**
- Create: `RUNBOOK.md`
- Create: `docs/archive/` (move three files into it)
- Modify: `RACE_PIPELINE.md`

- [ ] **Step 1: Create `RUNBOOK.md`**

Create `RUNBOOK.md` with exactly this content:
```markdown
# Formulytics Pipeline — Runbook

A plain-language operating guide for the F1 content pipeline. For the full
design, see `docs/superpowers/specs/2026-05-22-pipeline-manifest-design.md`.

## What the pipeline is

Two halves, coordinated by one Google Sheet (the "manifest"):

- **Producer** — renders a video after each F1 race/sprint and uploads it to a
  Google Drive folder. Runs automatically on GitHub Actions.
- **Poster** — captions the video and posts it to Instagram and YouTube. Runs
  on n8n Cloud.

The **manifest sheet** has one row per session. Open it any time to see what has
been rendered (`render_status`) and posted (`post_status`).

## Statuses

- `render_status`: `pending`, `rendered`, `skipped`, `error`.
- `post_status`: `pending`, `posting`, `posted`, `skipped`, `error`.

`skipped` means the pipeline deliberately ignores the row. You can set a row's
`post_status` to `skipped` yourself to stop the poster retrying something.

## Yearly maintenance checklist

1. **Google login token.** If Drive or Sheets access fails, regenerate it: run
   `python get_gdrive_token.py`, then update the `GDRIVE_REFRESH_TOKEN` GitHub
   secret. The Google Cloud consent screen must stay in **Production** mode, or
   the token expires every 7 days.
2. **Team colours.** `get_constructor_color` in `race_replay.py` lists 2026
   teams. When F1 teams change, update it, or new teams render grey.
3. **Gemini model name.** AI models are retired every year or two. When captions
   fail with a model error, update the model name.
4. **Zernio connections.** Instagram and YouTube drop their links to Zernio
   every few months. Reconnect them in Zernio when posts start failing.
5. **GitHub token (`WORKFLOW_PAT`).** Must have **no expiry**, or the producer's
   scheduling silently freezes the day it expires.

## What to do if...

- **A race did not post.** Open the manifest, find the row.
  - `render_status` is `error`: open the producer's GitHub Actions logs. To
    retry, set `render_status` back to `pending`.
  - `render_status` is `rendered` but `post_status` is `error`: the poster
    failed. To retry, set `post_status` back to `pending`.
- **You got a "producer run failed" email.** Open the linked GitHub Actions run;
  the log names the failed step. Most often it is OpenF1 data not ready (this
  auto-recovers on the next run) or an expired Google token (regenerate it).
- **Drive or Sheets access fails.** Regenerate the Google token (item 1 above).
- **Posting stopped entirely.** Check Zernio's Instagram/YouTube connections and
  reconnect them.
- **The manifest sheet looks damaged.** Google Sheets keeps automatic version
  history: File -> Version history -> restore an earlier version.

## Storage

Videos accumulate in Google Drive at roughly 2 GB per year. When the account
nears its storage limit, delete old posted videos or add storage.
```

- [ ] **Step 2: Archive the superseded docs**

Run:
```bash
mkdir -p docs/archive
git mv N8N_WORKFLOW_BUILD.md docs/archive/N8N_WORKFLOW_BUILD.md
git mv N8N_WORKFLOW_V2.md docs/archive/N8N_WORKFLOW_V2.md
git mv FORMULYTICS_PIPELINE.md docs/archive/FORMULYTICS_PIPELINE.md
```
Note: `N8N_WORKFLOW_V3.md` is intentionally kept in place — it is the working
reference for Plan 2 (the n8n poster) until that plan supersedes it.

- [ ] **Step 3: Point `RACE_PIPELINE.md` at the manifest spec**

In `RACE_PIPELINE.md`, find the line near the end:
```markdown
(An older, superseded sketch of the posting design exists in
`FORMULYTICS_PIPELINE.md` — ignore it; this document is authoritative for what
the race pipeline actually produces.)
```
Replace it with:
```markdown
The pipeline is now coordinated by a Google Sheet "season manifest" — the
authoritative design is `docs/superpowers/specs/2026-05-22-pipeline-manifest-design.md`,
and `RUNBOOK.md` is the plain-language operating guide. Older posting sketches
have been moved to `docs/archive/`.
```

- [ ] **Step 4: Commit**

```bash
git add RUNBOOK.md RACE_PIPELINE.md docs/archive/
git commit -m "docs: add runbook, archive superseded pipeline docs"
```

---

## Manual steps for the user (not code)

These require the repo owner and cannot be done by the engineer:

1. **Create the manifest sheet.** Make a new Google Sheet with the account that
   owns the Drive folder. Rename its tab to `sessions`. Leave it otherwise
   empty — the first producer run writes the header row. Copy the sheet id from
   the URL (`docs.google.com/spreadsheets/d/<THIS>/edit`).
2. **Regenerate the Google token with the Sheets scope.** Run
   `python get_gdrive_token.py` (now requesting Drive + Sheets), approve in the
   browser, and update the `GDRIVE_REFRESH_TOKEN` GitHub secret. Confirm the
   Google Cloud consent screen is in **Production** mode.
3. **Add GitHub secrets** (repo Settings -> Secrets and variables -> Actions):
   `MANIFEST_SHEET_ID`, `SMTP_USERNAME` (a Gmail address), `SMTP_PASSWORD` (a
   Gmail app password). Confirm `WORKFLOW_PAT` is set with **no expiry**.
4. **Bootstrap run.** From the Actions tab, run `F1 Race Replay` once via
   `workflow_dispatch` with a blank `test_session`. Confirm the manifest sheet
   is seeded (about 30 rows; finished sessions marked `skipped`).
5. **Protect the sheet (recommended).** In the sheet, protect row 1 and columns
   F:O so day-to-day viewing cannot accidentally damage the pipeline's data.

---

## Self-Review

**Spec coverage:**
- Manifest sheet + 15-column schema — File Structure schema table; Task 3 (`MANIFEST_HEADERS`).
- Self-seeding, past sessions `skipped` — Task 3 (`seed_status_for`, `build_manifest_row`, `manifest_plan`), Task 4 (`ensure_manifest`).
- Append-only seeding; refresh pending rows — Task 3 (`manifest_plan`), Task 4 (`ensure_manifest`).
- Producer reads/writes only F:I, cell-scoped — Task 4 (`update_render_row`).
- Producer consults the manifest, defensive adoption, output validation — Task 6 (`render_due`), Task 5 (`is_valid_render`).
- Phase 0 ensure_manifest in `main` — Task 6.
- Off-season heartbeat commit — Task 5 (`heartbeat`), wired in Task 6.
- Producer failure alert — Task 7.
- `MANIFEST_SHEET_ID` plumbing — Task 6, Task 7, Manual Steps.
- Sheets scope on the OAuth token — Task 2, Manual Steps.
- Runbook + doc consolidation — Task 9.
- Testing strategy (unit tests, dry run, live seed) — Tasks 1-6, Task 8.
- Out of scope (no backfill, no `_Q` renderer, n8n poster) — stated in the header; the poster is Plan 2.

**Placeholder scan:** No TBD/TODO. Every code step shows complete code; every command shows expected output.

**Type consistency:** `session_id_for(event, year, kind)` is defined in Task 1 and called with that signature in Tasks 3 and 6. `update_render_row(service, sheet_id, row_number, render_status, rendered_at, drive_file_id, render_notes)` is defined in Task 4 and called with exactly those 7 arguments in Task 6. `manifest_plan` returns `(appends, refreshes)` in Task 3 and is unpacked that way in Task 4. Row dicts carry `_row` (set in `parse_manifest_values`, Task 4) and it is read in `manifest_plan` (Task 3) and `render_due` (Task 6). `MANIFEST_HEADERS` order (Task 3) matches the F:I and D:E column ranges used in Task 4. Session dicts gain `meeting_key` (Task 3) which `attach_gp_names` consumes (Task 3).
