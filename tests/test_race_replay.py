import os
import sys
from datetime import datetime

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


def _fake_openf1_sessions():
    return [
        {"session_key": 1, "session_name": "Practice 1", "date_start": "2026-07-03T10:00:00+00:00", "location": "Silverstone"},
        {"session_key": 2, "session_name": "Sprint", "date_start": "2026-07-04T11:00:00+00:00", "location": "Silverstone"},
        {"session_key": 3, "session_name": "Race", "date_start": "2026-07-05T14:00:00+00:00", "location": "Silverstone"},
    ]


def test_list_sessions_picks_only_race_and_sprint():
    sessions = rr.list_sessions(_fake_openf1_sessions())
    assert len(sessions) == 2
    assert sorted(s["kind"] for s in sessions) == ["R", "S"]


def test_list_sessions_computes_fields():
    sessions = rr.list_sessions(_fake_openf1_sessions())
    race = next(s for s in sessions if s["kind"] == "R")
    assert race["event"] == "Silverstone"
    assert race["session_key"] == 3
    assert race["start"] == datetime(2026, 7, 5, 14, 0)
    assert race["end"] == datetime(2026, 7, 5, 16, 15)


def test_position_at():
    events = [(10, 5), (20, 3), (30, 1)]
    assert rr.position_at(events, 25) == 3
    assert rr.position_at(events, 5) == 5
    assert rr.position_at(events, 100) == 1


def test_compound_for_lap():
    stints = [
        {"lap_start": 1, "lap_end": 10, "compound": "MEDIUM"},
        {"lap_start": 11, "lap_end": 30, "compound": "HARD"},
    ]
    assert rr.compound_for_lap(stints, 5) == "MEDIUM"
    assert rr.compound_for_lap(stints, 20) == "HARD"
    assert rr.compound_for_lap(stints, 99) == "UNKNOWN"


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


def test_rewrite_cron_block_reversed_markers_raises():
    import pytest
    workflow = (
        "on:\n  schedule:\n"
        "    " + rr.CRON_END + "\n"
        "    " + rr.CRON_BEGIN + "\n"
    )
    with pytest.raises(RuntimeError):
        rr.rewrite_cron_block(workflow, ["1 2 3 4 *"])


def test_parse_force_session():
    assert rr.parse_force_session("2026 Miami R") == (2026, "Miami", "R")
    assert rr.parse_force_session("2026 Canadian Grand Prix S") == (2026, "Canadian Grand Prix", "S")


def test_parse_force_session_rejects_bad_input():
    import pytest
    with pytest.raises(SystemExit):
        rr.parse_force_session("2026 R")
    with pytest.raises(SystemExit):
        rr.parse_force_session("2026 Miami X")
