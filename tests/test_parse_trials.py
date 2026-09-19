"""Offline regression tests for the barrier-trial parser, against a checked-in fixture."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from hkjc.data.parse.trials import parse_barrier_trials

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hkjc"
HTML = (FIXTURES / "btresult_2026-06-05.html").read_text(encoding="utf-8")


def test_parse_barrier_trials() -> None:
    runs = parse_barrier_trials(HTML, date(2026, 6, 5))
    assert len(runs) >= 10  # multiple batches, several runners each

    first = runs[0]
    assert first.trial_date == date(2026, 6, 5)
    assert first.batch == 1
    assert first.venue == "ST"
    assert first.surface == "ALL WEATHER TRACK"
    assert first.distance_m == 1200
    assert first.going == "GOOD"
    assert first.batch_time_s is not None and abs(first.batch_time_s - 70.80) < 0.01

    assert first.horse_id == "HK_2024_K400"
    assert first.horse_name == "AEROVOLANIC"
    assert first.jockey == "Z Purton"  # trials carry names, not linked ids
    assert first.trainer == "P C Ng"
    assert first.draw == 7
    assert first.gear == "B"
    assert first.result == "Passed"
    assert first.comment is not None and first.comment.startswith("Led all the way")
    assert first.time_s is not None and abs(first.time_s - 70.80) < 0.01


def test_trials_cover_multiple_batches() -> None:
    runs = parse_barrier_trials(HTML, date(2026, 6, 5))
    assert {r.batch for r in runs} >= {1, 2, 3}


def test_trial_backfill_dates_fill_the_gap_before_the_listed_span() -> None:
    from hkjc.data.pipeline import trial_backfill_dates

    listed = [date(2025, 6, 19), date(2025, 6, 24)]
    gap = trial_backfill_dates(date(2025, 6, 10), listed, today=date(2026, 9, 19))
    assert gap[0] == date(2025, 6, 10)
    assert gap[-1] == date(2025, 6, 18)  # stops the day before the earliest listed date
    assert len(gap) == 9
    # since inside the listed span -> nothing to backfill
    assert trial_backfill_dates(date(2025, 6, 20), listed, today=date(2026, 9, 19)) == []
    # no listing at all -> probe through today
    assert trial_backfill_dates(date(2026, 9, 17), [], today=date(2026, 9, 19))[-1] == date(
        2026, 9, 19
    )
