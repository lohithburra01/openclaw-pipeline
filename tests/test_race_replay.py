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
