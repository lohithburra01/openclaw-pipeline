# Race Replay Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `race_replay.py` that renders F1 race/sprint position timelapses fully automatically on GitHub Actions, calendar-driven, with no manual input.

**Architecture:** A single self-scheduling Python script. Every run it (1) reads the FastF1 calendar and rewrites its own workflow's cron block so it fires shortly after each Race/Sprint, and (2) renders any just-ended session whose video is not already in Google Drive. One workflow file holds the self-managed crons plus a weekly safety tick.

**Tech Stack:** Python 3.11, FastF1, OpenCV (`opencv-python-headless`), Pillow, NumPy, pandas, Google Drive API, GitHub Actions, pytest (dev only).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `race_replay.py` | The whole pipeline: render code (from notebook cell 22) + scheduling/orchestration |
| `.github/workflows/race_replay.yml` | Self-managed cron schedule + weekly safety tick; runs the script |
| `tests/test_race_replay.py` | Unit tests for the pure scheduling/parsing functions |
| `.gitignore` | Keep render output and caches out of git |

The render code is **copied verbatim** from `RACE_REPLAY.ipynb` cell 22 (index 22). The notebook is never modified. Within `race_replay.py` the pure logic functions (testable, no I/O) are separated from the I/O functions (FastF1, Drive, git) so the bulk of the code is unit-tested.

**Function inventory (final names — use these exactly):**

- Pure: `session_label_for`, `session_end_time`, `retry_slots`, `cron_for_datetime`, `output_slug`, `parse_force_session`, `list_sessions`, `schedulable_sessions`, `due_sessions`, `build_cron_lines`, `rewrite_cron_block`
- I/O: `load_event_schedule`, `session_has_data`, `get_drive_service`, `drive_has_file`, `drive_upload_file`, `commit_and_push`
- Orchestration: `reschedule`, `render_due`, `main`
- Render (copied from cell 22): `get_constructor_color`, `get_tire_info`, `hex_to_rgb`, `get_ordinal`, `load_fonts`, `create_race_timelapse`

A "session dict" used throughout has exactly these keys: `event` (str), `round` (int), `kind` (`"R"` or `"S"`), `start` (datetime), `end` (datetime).

---

## Task 1: Scaffold race_replay.py from notebook cell 22

**Files:**
- Create: `race_replay.py`
- Create: `.gitignore`

- [ ] **Step 1: Extract cell 22 into race_replay.py**

Run:
```bash
python -c "import json; nb=json.load(open('RACE_REPLAY.ipynb',encoding='utf-8')); open('race_replay.py','w',encoding='utf-8',newline='\n').write(''.join(nb['cells'][22]['source']))"
```
Expected: `race_replay.py` created, ~297 lines, starts with `import fastf1` and ends with an `if __name__ == "__main__":` block calling `create_race_timelapse(... gp='MIAMI' ...)`.

- [ ] **Step 2: Insert config block after the imports**

Read `race_replay.py`. Find the import line `import os` (the last import in cell 22's header, line ~6). Insert the following block immediately after it:

```python

# ============================================================
# RACE REPLAY PIPELINE - configuration
# ============================================================
import argparse
import re
import subprocess
import sys
from datetime import datetime, timedelta

SEASON = int(os.environ.get("RACE_SEASON", datetime.utcnow().year))
RACE_END_BUFFER_MIN = 135       # race counted as "ended" 2h15m after start
SPRINT_END_BUFFER_MIN = 75      # sprint counted as "ended" 1h15m after start
RETRY_OFFSETS_MIN = [15, 60, 105, 150, 195]  # retry slots (minutes after end)
LOOKAHEAD_DAYS = 9              # how far ahead to write cron slots
RENDER_LOOKBACK_HOURS = 6       # window after end-time in which a session renders
WEEKLY_SAFETY_CRON = "0 12 * * 3"            # Wednesday self-heal tick
DURATION_SECONDS = 30
FPS = 60
PORTRAIT = True
WORKFLOW_PATH = os.path.join(".github", "workflows", "race_replay.yml")
CRON_BEGIN = "# === BEGIN AUTO-MANAGED CRON (edited by race_replay.py) ==="
CRON_END = "# === END AUTO-MANAGED CRON ==="
CACHE_DIR = ".fastf1_cache"
OUTPUT_DIR = "output"

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
```

- [ ] **Step 3: Make the on-screen session label dynamic**

The render function hardcodes the word `SPRINT` in the title. First, add a parameter to the `create_race_timelapse` signature. Read the file to get the exact two-line signature (it is `def create_race_timelapse(year=2026, gp='Bahrain', session_type='R', ` then `                          save_path='output.mp4', duration=30, fps=60, portrait=True):`). Replace it with:

```python
def create_race_timelapse(year=2026, gp='Bahrain', session_type='R',
                          save_path='output.mp4', duration=30, fps=60, portrait=True,
                          session_label='RACE'):
```

Then find the title line (inside the render loop):
```python
        title_str = f"{gp.upper()} {year} - SPRINT - LAP {int(leader_lap)}/{total_laps}"
```
Replace `SPRINT` with `{session_label}`:
```python
        title_str = f"{gp.upper()} {year} - {session_label} - LAP {int(leader_lap)}/{total_laps}"
```

- [ ] **Step 4: Remove the hardcoded `__main__` test block and add the orchestration section header**

Read the file. Delete the entire trailing block (the last ~3 lines):
```python
if __name__ == "__main__":
    create_race_timelapse(year=2026, gp='MIAMI', session_type='R',
                          save_path='output.mp4', duration=15, fps=60, portrait=True)
```
Replace it with this section header (later tasks append functions below it):
```python

# ============================================================
# SCHEDULING & ORCHESTRATION
# ============================================================
```

- [ ] **Step 5: Create .gitignore**

Create `.gitignore` with this content (append these lines if the file already exists):
```
output/
.fastf1_cache/
__pycache__/
.pytest_cache/
```

- [ ] **Step 6: Verify the module imports cleanly**

Run:
```bash
python -c "import race_replay; print('OK', race_replay.SEASON, hasattr(race_replay, 'create_race_timelapse'))"
```
Expected: `OK 2026 True` (no traceback).

- [ ] **Step 7: Commit**

```bash
git add race_replay.py .gitignore
git commit -m "feat: scaffold race_replay.py from notebook cell 22"
```

---

## Task 2: Pure helper functions

**Files:**
- Modify: `race_replay.py` (append to the SCHEDULING & ORCHESTRATION section)
- Create: `tests/test_race_replay.py`

- [ ] **Step 1: Install pytest and create the test file**

Run:
```bash
pip install -q pytest
```
Create `tests/test_race_replay.py`:
```python
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import race_replay as rr


def test_session_label_for():
    assert rr.session_label_for("R") == "RACE"
    assert rr.session_label_for("S") == "SPRINT"


def test_session_end_time_race():
    assert rr.session_end_time(datetime(2026, 5, 24, 20, 0), "R") == datetime(2026, 5, 24, 22, 15)


def test_session_end_time_sprint():
    assert rr.session_end_time(datetime(2026, 5, 23, 16, 0), "S") == datetime(2026, 5, 23, 17, 15)


def test_retry_slots():
    slots = rr.retry_slots(datetime(2026, 5, 24, 22, 15))
    assert len(slots) == 5
    assert slots[0] == datetime(2026, 5, 24, 22, 30)
    assert slots[-1] == datetime(2026, 5, 25, 1, 30)


def test_cron_for_datetime():
    assert rr.cron_for_datetime(datetime(2026, 5, 24, 22, 30)) == "30 22 24 5 *"


def test_output_slug():
    assert rr.output_slug("Canadian Grand Prix", 2026, "R") == "canadian_grand_prix_2026_R.mp4"
    assert rr.output_slug("Miami Grand Prix", 2026, "S") == "miami_grand_prix_2026_S.mp4"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python -m pytest tests/test_race_replay.py -q
```
Expected: FAIL — `AttributeError: module 'race_replay' has no attribute 'session_label_for'`.

- [ ] **Step 3: Implement the helpers**

Append to `race_replay.py` (under the SCHEDULING & ORCHESTRATION header):
```python
def session_label_for(kind):
    """kind is 'R' or 'S'; returns the on-screen title label."""
    return "SPRINT" if kind == "S" else "RACE"


def session_end_time(start, kind):
    """Estimated wall-clock end of a session, given its start datetime."""
    buffer = SPRINT_END_BUFFER_MIN if kind == "S" else RACE_END_BUFFER_MIN
    return start + timedelta(minutes=buffer)


def retry_slots(end_time):
    """Datetimes at which the workflow should retry rendering this session."""
    return [end_time + timedelta(minutes=off) for off in RETRY_OFFSETS_MIN]


def cron_for_datetime(dt):
    """One UTC datetime -> a GitHub Actions 5-field cron string."""
    return f"{dt.minute} {dt.hour} {dt.day} {dt.month} *"


def output_slug(event_name, year, kind):
    """Stable output filename, e.g. 'canadian_grand_prix_2026_R.mp4'."""
    base = re.sub(r"[^a-z0-9]+", "_", event_name.lower()).strip("_")
    return f"{base}_{year}_{kind}.mp4"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_race_replay.py -q
```
Expected: PASS — 6 passed.

- [ ] **Step 5: Commit**

```bash
git add race_replay.py tests/test_race_replay.py
git commit -m "feat: add pure scheduling helper functions"
```

---

## Task 3: Parse the FastF1 calendar into session dicts

**Files:**
- Modify: `race_replay.py`
- Test: `tests/test_race_replay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_race_replay.py`:
```python
def _fake_schedule():
    return pd.DataFrame([{
        "EventName": "Sample Grand Prix", "RoundNumber": 9,
        "Session1": "Practice 1", "Session1DateUtc": pd.Timestamp("2026-07-03 10:00"),
        "Session2": "Sprint Qualifying", "Session2DateUtc": pd.Timestamp("2026-07-03 14:00"),
        "Session3": "Sprint", "Session3DateUtc": pd.Timestamp("2026-07-04 11:00"),
        "Session4": "Qualifying", "Session4DateUtc": pd.Timestamp("2026-07-04 15:00"),
        "Session5": "Race", "Session5DateUtc": pd.Timestamp("2026-07-05 14:00"),
    }])


def test_list_sessions_picks_only_race_and_sprint():
    sessions = rr.list_sessions(_fake_schedule())
    assert len(sessions) == 2
    assert sorted(s["kind"] for s in sessions) == ["R", "S"]


def test_list_sessions_computes_fields():
    sessions = rr.list_sessions(_fake_schedule())
    race = next(s for s in sessions if s["kind"] == "R")
    assert race["event"] == "Sample Grand Prix"
    assert race["round"] == 9
    assert race["start"] == datetime(2026, 7, 5, 14, 0)
    assert race["end"] == datetime(2026, 7, 5, 16, 15)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python -m pytest tests/test_race_replay.py::test_list_sessions_picks_only_race_and_sprint -q
```
Expected: FAIL — `AttributeError: ... has no attribute 'list_sessions'`.

- [ ] **Step 3: Implement `list_sessions`**

Append to `race_replay.py`:
```python
def list_sessions(schedule_df):
    """FastF1 event-schedule DataFrame -> list of race/sprint session dicts.

    Matches the session names 'Race' and 'Sprint' exactly, so 'Sprint
    Qualifying' / 'Sprint Shootout' are ignored.
    """
    sessions = []
    for _, ev in schedule_df.iterrows():
        for i in range(1, 6):
            name = ev.get(f"Session{i}")
            date = ev.get(f"Session{i}DateUtc")
            if name not in ("Race", "Sprint"):
                continue
            if date is None or pd.isna(date):
                continue
            kind = "S" if name == "Sprint" else "R"
            start = pd.Timestamp(date).to_pydatetime()
            sessions.append({
                "event": ev["EventName"],
                "round": int(ev["RoundNumber"]),
                "kind": kind,
                "start": start,
                "end": session_end_time(start, kind),
            })
    return sessions
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_race_replay.py -q
```
Expected: PASS — 8 passed.

- [ ] **Step 5: Commit**

```bash
git add race_replay.py tests/test_race_replay.py
git commit -m "feat: parse FastF1 calendar into session dicts"
```

---

## Task 4: Schedulable and due-session filters

**Files:**
- Modify: `race_replay.py`
- Test: `tests/test_race_replay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_race_replay.py`:
```python
def _session(start, end):
    return {"event": "X", "round": 1, "kind": "R", "start": start, "end": end}


def test_schedulable_includes_upcoming_within_lookahead():
    now = datetime(2026, 5, 20, 12, 0)
    s = _session(datetime(2026, 5, 24, 20, 0), datetime(2026, 5, 24, 22, 15))
    assert rr.schedulable_sessions([s], now) == [s]


def test_schedulable_excludes_beyond_lookahead():
    now = datetime(2026, 5, 20, 12, 0)
    s = _session(datetime(2026, 7, 1, 13, 0), datetime(2026, 7, 1, 15, 15))
    assert rr.schedulable_sessions([s], now) == []


def test_schedulable_excludes_passed_retry_window():
    now = datetime(2026, 5, 20, 12, 0)
    s = _session(datetime(2026, 5, 3, 17, 0), datetime(2026, 5, 3, 19, 15))
    assert rr.schedulable_sessions([s], now) == []


def test_schedulable_keeps_just_ended_session_with_slots_left():
    # session ended 30 min ago; later retry slots are still in the future
    now = datetime(2026, 5, 23, 17, 45)
    s = _session(datetime(2026, 5, 23, 16, 0), datetime(2026, 5, 23, 17, 15))
    assert rr.schedulable_sessions([s], now) == [s]


def test_due_sessions_within_lookback():
    s = _session(datetime(2026, 5, 24, 20, 0), datetime(2026, 5, 24, 22, 15))
    assert rr.due_sessions([s], datetime(2026, 5, 24, 23, 0)) == [s]


def test_due_sessions_excludes_not_yet_ended():
    s = _session(datetime(2026, 5, 24, 20, 0), datetime(2026, 5, 24, 22, 15))
    assert rr.due_sessions([s], datetime(2026, 5, 24, 21, 0)) == []


def test_due_sessions_excludes_stale():
    s = _session(datetime(2026, 5, 24, 20, 0), datetime(2026, 5, 24, 22, 15))
    assert rr.due_sessions([s], datetime(2026, 5, 25, 6, 0)) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python -m pytest tests/test_race_replay.py::test_schedulable_includes_upcoming_within_lookahead -q
```
Expected: FAIL — `AttributeError: ... has no attribute 'schedulable_sessions'`.

- [ ] **Step 3: Implement the filters**

Append to `race_replay.py`:
```python
def schedulable_sessions(sessions, now):
    """Sessions worth writing cron slots for: their last retry slot is still
    in the future, and they start within the lookahead horizon."""
    last_offset = max(RETRY_OFFSETS_MIN)
    horizon = now + timedelta(days=LOOKAHEAD_DAYS)
    out = []
    for s in sessions:
        last_slot = s["end"] + timedelta(minutes=last_offset)
        if last_slot > now and s["start"] < horizon:
            out.append(s)
    return out


def due_sessions(sessions, now):
    """Sessions that have ended and are still inside the render window."""
    out = []
    for s in sessions:
        if s["end"] <= now <= s["end"] + timedelta(hours=RENDER_LOOKBACK_HOURS):
            out.append(s)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_race_replay.py -q
```
Expected: PASS — 15 passed.

- [ ] **Step 5: Commit**

```bash
git add race_replay.py tests/test_race_replay.py
git commit -m "feat: add schedulable and due-session filters"
```

---

## Task 5: Build the cron block and rewrite the workflow file

**Files:**
- Modify: `race_replay.py`
- Test: `tests/test_race_replay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_race_replay.py`:
```python
def test_build_cron_lines_always_includes_safety():
    lines = rr.build_cron_lines([], datetime(2026, 5, 20, 12, 0))
    assert lines == [rr.WEEKLY_SAFETY_CRON]


def test_build_cron_lines_adds_session_slots():
    now = datetime(2026, 5, 20, 12, 0)
    s = _session(datetime(2026, 5, 24, 20, 0), datetime(2026, 5, 24, 22, 15))
    lines = rr.build_cron_lines([s], now)
    assert "30 22 24 5 *" in lines        # end + 15 min
    assert rr.WEEKLY_SAFETY_CRON in lines
    assert len(lines) == 6                # safety + 5 retry slots


def test_rewrite_cron_block_replaces_only_managed_region():
    workflow = (
        "name: F1 Race Replay\n"
        "on:\n"
        "  schedule:\n"
        "    " + rr.CRON_BEGIN + "\n"
        "    - cron: \"0 12 * * 3\"\n"
        "    " + rr.CRON_END + "\n"
        "jobs:\n"
        "  render:\n"
    )
    out = rr.rewrite_cron_block(workflow, ["1 2 3 4 *", "5 6 7 8 *"])
    assert "- cron: \"1 2 3 4 *\"" in out
    assert "- cron: \"5 6 7 8 *\"" in out
    assert "0 12 * * 3" not in out
    assert out.count(rr.CRON_BEGIN) == 1
    assert out.count(rr.CRON_END) == 1
    assert out.startswith("name: F1 Race Replay\n")
    assert out.endswith("jobs:\n  render:\n")


def test_rewrite_cron_block_missing_markers_raises():
    import pytest
    with pytest.raises(RuntimeError):
        rr.rewrite_cron_block("no markers here", ["1 2 3 4 *"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python -m pytest tests/test_race_replay.py::test_build_cron_lines_always_includes_safety -q
```
Expected: FAIL — `AttributeError: ... has no attribute 'build_cron_lines'`.

- [ ] **Step 3: Implement the cron builders**

Append to `race_replay.py`:
```python
def build_cron_lines(sessions, now):
    """Sorted, de-duplicated cron strings: the weekly safety tick plus a retry
    slot for every schedulable session."""
    lines = {WEEKLY_SAFETY_CRON}
    for s in schedulable_sessions(sessions, now):
        for slot in retry_slots(s["end"]):
            lines.add(cron_for_datetime(slot))
    return sorted(lines)


def rewrite_cron_block(workflow_text, cron_lines):
    """Return workflow_text with the region between CRON_BEGIN and CRON_END
    replaced by `cron_lines`. The marker comment lines themselves are kept."""
    if CRON_BEGIN not in workflow_text or CRON_END not in workflow_text:
        raise RuntimeError(
            f"Workflow file is missing the cron markers {CRON_BEGIN!r}/{CRON_END!r}"
        )
    begin = workflow_text.index(CRON_BEGIN)
    end = workflow_text.index(CRON_END)
    head = workflow_text[:begin + len(CRON_BEGIN)]
    tail = workflow_text[end:]
    block = "".join(f'\n    - cron: "{c}"' for c in cron_lines)
    return f"{head}{block}\n    {tail}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_race_replay.py -q
```
Expected: PASS — 19 passed.

- [ ] **Step 5: Commit**

```bash
git add race_replay.py tests/test_race_replay.py
git commit -m "feat: build cron block and rewrite workflow file"
```

---

## Task 6: FastF1 I/O functions

**Files:**
- Modify: `race_replay.py`

These functions hit the network, so they are verified with a live integration call rather than unit tests.

- [ ] **Step 1: Implement the FastF1 helpers**

Append to `race_replay.py`:
```python
def load_event_schedule(season):
    """Return the FastF1 event-schedule DataFrame for a season."""
    return fastf1.get_event_schedule(season, include_testing=False)


def session_has_data(year, event_name, kind):
    """True once FastF1 has lap data for the session, False if not ready yet."""
    try:
        session = fastf1.get_session(year, event_name, kind)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        return len(session.laps) > 0
    except Exception as exc:
        print(f"  data-readiness check failed: {exc}")
        return False
```

- [ ] **Step 2: Verify against the live FastF1 API**

Run:
```bash
python -c "import race_replay as r; df=r.load_event_schedule(2026); s=r.list_sessions(df); print('sessions:', len(s)); print('Miami R data ready:', r.session_has_data(2026,'Miami','R'))"
```
Expected: `sessions:` followed by a number around 28, then `Miami R data ready: True` (the Miami GP is in the past, so its data exists).

- [ ] **Step 3: Commit**

```bash
git add race_replay.py
git commit -m "feat: add FastF1 calendar and session-data helpers"
```

---

## Task 7: Google Drive helpers

**Files:**
- Modify: `race_replay.py`

The Google API libraries are imported lazily inside these functions so that `import race_replay` (and the unit tests) work without those libraries installed. Real Drive verification happens at bootstrap; here we only check the module still imports.

- [ ] **Step 1: Implement the Drive helpers**

Append to `race_replay.py`:
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


def drive_has_file(service, folder_id, filename):
    """True if a non-trashed file named `filename` exists in the folder."""
    safe = filename.replace("\\", "\\\\").replace("'", "\\'")
    query = f"name = '{safe}' and '{folder_id}' in parents and trashed = false"
    resp = service.files().list(q=query, fields="files(id,name)").execute()
    return len(resp.get("files", [])) > 0


def drive_upload_file(service, folder_id, path):
    """Upload a local file into the Drive folder; return the new file id."""
    from googleapiclient.http import MediaFileUpload

    metadata = {"name": os.path.basename(path), "parents": [folder_id]}
    media = MediaFileUpload(path, mimetype="video/mp4", resumable=True)
    result = service.files().create(
        body=metadata, media_body=media, fields="id,name"
    ).execute()
    return result["id"]
```

- [ ] **Step 2: Verify the module still imports without Google libraries**

Run:
```bash
python -c "import race_replay; print('import OK; drive helpers present:', all(hasattr(race_replay,n) for n in ['get_drive_service','drive_has_file','drive_upload_file']))"
```
Expected: `import OK; drive helpers present: True`.

- [ ] **Step 3: Commit**

```bash
git add race_replay.py
git commit -m "feat: add Google Drive marker-check and upload helpers"
```

---

## Task 8: Git commit-and-push helper

**Files:**
- Modify: `race_replay.py`

- [ ] **Step 1: Implement `commit_and_push`**

Append to `race_replay.py`:
```python
def commit_and_push(path, message):
    """Commit a single file and push. Only acts inside GitHub Actions; on a
    local machine it prints what it would do and returns."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        print(f"  (local run) would commit and push {path}")
        return
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(
        ["git", "config", "user.email",
         "41898282+github-actions[bot]@users.noreply.github.com"],
        check=True,
    )
    subprocess.run(["git", "add", path], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push"], check=True)
    print(f"  committed and pushed {path}")
```

- [ ] **Step 2: Verify the local-run guard**

Run:
```bash
python -c "import race_replay as r; r.commit_and_push('x.txt','msg')"
```
Expected: `  (local run) would commit and push x.txt` (no git command runs, no error).

- [ ] **Step 3: Commit**

```bash
git add race_replay.py
git commit -m "feat: add git commit-and-push helper for cron updates"
```

---

## Task 9: Orchestration and CLI entry point

**Files:**
- Modify: `race_replay.py`
- Test: `tests/test_race_replay.py`

- [ ] **Step 1: Write the failing test for `parse_force_session`**

Append to `tests/test_race_replay.py`:
```python
def test_parse_force_session():
    assert rr.parse_force_session("2026 Miami R") == (2026, "Miami", "R")
    assert rr.parse_force_session("2026 Canadian Grand Prix S") == (2026, "Canadian Grand Prix", "S")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_race_replay.py::test_parse_force_session -q
```
Expected: FAIL — `AttributeError: ... has no attribute 'parse_force_session'`.

- [ ] **Step 3: Implement orchestration, `parse_force_session`, and `main`**

Append to `race_replay.py`:
```python
def parse_force_session(text):
    """'2026 Miami R' -> (2026, 'Miami', 'R'). Event name may contain spaces."""
    parts = text.split()
    return int(parts[0]), " ".join(parts[1:-1]), parts[-1].upper()


def reschedule(now, dry_run=False):
    """Phase 1: rewrite the workflow's cron block from the F1 calendar."""
    sessions = list_sessions(load_event_schedule(SEASON))
    cron_lines = build_cron_lines(sessions, now)
    print("Cron block to apply:")
    for c in cron_lines:
        print(f'  - cron: "{c}"')

    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        current = f.read()
    updated = rewrite_cron_block(current, cron_lines)

    if updated == current:
        print("Schedule unchanged.")
        return
    if dry_run:
        print("(dry-run) workflow file not modified.")
        return
    with open(WORKFLOW_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(updated)
    commit_and_push(WORKFLOW_PATH, "chore: update race-replay cron schedule")
    print("Schedule updated.")


def render_due(now):
    """Phase 2: render any just-ended session not already in Google Drive."""
    sessions = list_sessions(load_event_schedule(SEASON))
    candidates = due_sessions(sessions, now)
    if not candidates:
        print("NO_SESSION")
        return

    service = get_drive_service()
    folder = os.environ["GDRIVE_FOLDER_ID"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for s in candidates:
        slug = output_slug(s["event"], SEASON, s["kind"])
        print(f"Candidate: {s['event']} ({s['kind']}) -> {slug}")
        if drive_has_file(service, folder, slug):
            print(f"  ALREADY_RENDERED ({slug})")
            continue
        if not session_has_data(SEASON, s["event"], s["kind"]):
            print(f"  DATA_NOT_READY ({s['event']} {s['kind']})")
            continue
        out_path = os.path.join(OUTPUT_DIR, slug)
        create_race_timelapse(
            year=SEASON, gp=s["event"], session_type=s["kind"],
            save_path=out_path, duration=DURATION_SECONDS, fps=FPS,
            portrait=PORTRAIT, session_label=session_label_for(s["kind"]),
        )
        file_id = drive_upload_file(service, folder, out_path)
        print(f"  UPLOADED ({slug}, drive id {file_id})")


def main():
    parser = argparse.ArgumentParser(description="F1 race-replay pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cron block without writing or committing")
    parser.add_argument("--force-session", metavar="SPEC",
                        help='render one session locally, e.g. "2026 Miami R"')
    parser.add_argument("--duration", type=int, default=DURATION_SECONDS,
                        help="override video length in seconds (for testing)")
    args = parser.parse_args()

    if args.force_session:
        year, event, kind = parse_force_session(args.force_session)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, output_slug(event, year, kind))
        create_race_timelapse(
            year=year, gp=event, session_type=kind, save_path=out_path,
            duration=args.duration, fps=FPS, portrait=PORTRAIT,
            session_label=session_label_for(kind),
        )
        print(f"Rendered {out_path}")
        return

    now = datetime.utcnow()
    print(f"=== race_replay.py @ {now} UTC (season {SEASON}) ===")
    reschedule(now, dry_run=args.dry_run)
    if not args.dry_run:
        render_due(now)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_race_replay.py -q
```
Expected: PASS — 22 passed.

- [ ] **Step 5: Verify the CLI parses**

Run:
```bash
python race_replay.py --help
```
Expected: usage text listing `--dry-run`, `--force-session`, and `--duration`.

- [ ] **Step 6: Commit**

```bash
git add race_replay.py tests/test_race_replay.py
git commit -m "feat: add pipeline orchestration and CLI entry point"
```

---

## Task 10: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/race_replay.yml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/race_replay.yml` with exactly this content (the marker comment lines must match `CRON_BEGIN`/`CRON_END` in `race_replay.py`):
```yaml
name: F1 Race Replay

on:
  workflow_dispatch:
  schedule:
    # === BEGIN AUTO-MANAGED CRON (edited by race_replay.py) ===
    - cron: "0 12 * * 3"
    # === END AUTO-MANAGED CRON ===

permissions:
  contents: write

concurrency:
  group: race-replay
  cancel-in-progress: false

jobs:
  render:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.WORKFLOW_PAT }}

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-racereplay

      - name: Cache FastF1
        uses: actions/cache@v4
        with:
          path: .fastf1_cache
          key: ${{ runner.os }}-fastf1-racereplay

      - name: Install dependencies
        run: |
          pip install -q fastf1 opencv-python-headless pillow numpy pandas \
            google-auth google-auth-httplib2 google-api-python-client

      - name: Run race replay pipeline
        env:
          GDRIVE_REFRESH_TOKEN: ${{ secrets.GDRIVE_REFRESH_TOKEN }}
          GDRIVE_CLIENT_ID: ${{ secrets.GDRIVE_CLIENT_ID }}
          GDRIVE_CLIENT_SECRET: ${{ secrets.GDRIVE_CLIENT_SECRET }}
          GDRIVE_FOLDER_ID: ${{ secrets.GDRIVE_FOLDER_ID }}
        run: python race_replay.py

      - name: Upload artifact backup
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: race-replay-video
          path: output/*.mp4
          retention-days: 5
          if-no-files-found: ignore
```

- [ ] **Step 2: Verify the markers match the script**

Run:
```bash
python -c "import race_replay as r; t=open(r.WORKFLOW_PATH,encoding='utf-8').read(); print('markers OK:', r.CRON_BEGIN in t and r.CRON_END in t)"
```
Expected: `markers OK: True`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/race_replay.yml
git commit -m "feat: add self-scheduling race-replay workflow"
```

---

## Task 11: Integration verification

**Files:** none changed — this task only runs and checks the pipeline.

- [ ] **Step 1: Verify the scheduling dry run against the live calendar**

Run:
```bash
python race_replay.py --dry-run
```
Expected: prints `Cron block to apply:` followed by `- cron:` lines that include `0 12 * * 3` and several dated slots for the Canadian Grand Prix (round 5, May 23-25), then `(dry-run) workflow file not modified.` Exit code 0. Confirm `.github/workflows/race_replay.yml` is unchanged (`git status` shows it clean).

- [ ] **Step 2: Verify a real render via force-session**

Run (short duration so it finishes fast):
```bash
python race_replay.py --force-session "2026 Miami R" --duration 3
```
Expected: FastF1 loads the Miami race, frames render, ends with `Rendered output\miami_grand_prix_2026_R.mp4`. Confirm the file exists and is larger than 100 KB:
```bash
python -c "import os; p='output/miami_grand_prix_2026_R.mp4'; print(p, os.path.getsize(p), 'bytes')"
```
Expected: a size well over 100000 bytes.

- [ ] **Step 3: Confirm the rendered title label is dynamic**

The Miami render above was a Race, so its title shows `MIAMI GRAND PRIX 2026 - RACE - LAP ...`. Spot-check by opening `output/miami_grand_prix_2026_R.mp4`, or trust the unit test `test_session_label_for` plus the verified `session_label` wiring. Note for the reviewer: a Sprint session would render `- SPRINT -` instead.

- [ ] **Step 4: Run the full unit-test suite once more**

Run:
```bash
python -m pytest tests/test_race_replay.py -q
```
Expected: PASS — 22 passed.

- [ ] **Step 5: Clean up the local test artifact and commit nothing**

Run:
```bash
python -c "import os; os.path.exists('output/miami_grand_prix_2026_R.mp4') and os.remove('output/miami_grand_prix_2026_R.mp4'); print('cleaned')"
```
Expected: `cleaned`. `git status` should show no changes (the `output/` directory is gitignored). No commit needed.

---

## Post-Implementation: Manual steps for the user (not code)

These cannot be done by the engineer — they require the repo owner:

1. **Create the `WORKFLOW_PAT` secret.** At github.com/settings/tokens generate a classic token with the `repo` and `workflow` scopes. Add it in repo Settings -> Secrets and variables -> Actions as `WORKFLOW_PAT`.
2. **Confirm the four `GDRIVE_*` secrets exist** (they are already used by `render.yml`).
3. **Merge this branch** to the default branch.
4. **Bootstrap:** trigger `F1 Race Replay` once from the Actions tab (`workflow_dispatch`). It writes the first real cron slots (Canadian GP) and exits. The pipeline is autonomous from then on.

---

## Self-Review

**Spec coverage:**
- Standalone `race_replay.py` from cell 22, notebook untouched — Task 1.
- Handles Race and Sprint — `list_sessions` matches both; `session_label_for` labels both; Tasks 2-3.
- Fully automatic, calendar-driven scheduling — Tasks 5, 9 (`reschedule`).
- Runs only right after a session ends — `retry_slots` + `build_cron_lines` + workflow cron, Tasks 2, 5, 10.
- Retry data fetch across runs — `session_has_data` returns False -> `DATA_NOT_READY`, next cron slot retries; Tasks 6, 9.
- Upload to existing Google Drive folder — Tasks 7, 9.
- Drive existence acts as the "already rendered" marker — `drive_has_file`, Tasks 7, 9.
- Weekly self-heal tick — `WEEKLY_SAFETY_CRON`, Tasks 1, 5, 10.
- One new secret `WORKFLOW_PAT`; reuse `GDRIVE_*` — Task 10, Post-Implementation.
- SPRINT title-label bug fixed — Task 1 Step 3.
- Testing strategy (`--dry-run`, `--force-session`, bootstrap) — Tasks 9, 11, Post-Implementation.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has expected output.

**Type consistency:** Session dicts use the keys `event/round/kind/start/end` everywhere (Tasks 3, 4, 9 and the test helper `_session`). Function names match the inventory. `create_race_timelapse` is always called with the `session_label` keyword added in Task 1. `CRON_BEGIN`/`CRON_END` constants (Task 1) match the workflow markers (Task 10), verified in Task 10 Step 2.
