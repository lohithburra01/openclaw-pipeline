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
