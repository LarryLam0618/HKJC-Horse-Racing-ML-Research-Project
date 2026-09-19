"""As-of correctness tests for the feature builder (PLAN.md §1H): the leakage-critical
guarantee that horse-history features use *strictly prior* runs only."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from hkjc.features import build
from hkjc.features.build import (
    _add_class_drop,
    _add_weight_dynamics,
    _class_ord,
    _horse_history,
    lbw_to_lengths,
)


def test_lbw_to_lengths() -> None:
    assert lbw_to_lengths("-") == 0.0
    assert lbw_to_lengths("1-1/2") == 1.5
    assert lbw_to_lengths("3/4") == 0.75
    assert lbw_to_lengths("2-3/4") == 2.75
    assert lbw_to_lengths("10") == 10.0
    assert lbw_to_lengths("NK") == 0.3
    assert lbw_to_lengths("SH") == 0.05
    assert lbw_to_lengths("DIST") == 30.0
    assert lbw_to_lengths(None) is None
    assert lbw_to_lengths("garbage") is None


def _horse_frame() -> pl.DataFrame:
    # One horse, three dated runs; won race 1 and race 3.
    return pl.DataFrame(
        {
            "horse_id": ["H1", "H1", "H1"],
            "race_date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            "race_no": [1, 1, 1],
            "season": ["2019-20", "2019-20", "2019-20"],
            "won": [1, 0, 1],
            "placed": [1, 0, 1],
            "finish_pos": [1, 5, 1],
            "lbw_len": [0.0, 3.0, 0.0],
            "speed": [16.0, 15.0, 16.5],
            "dist_band": [1200, 1200, 1200],
            "going": ["G", "G", "G"],
        }
    )


def test_career_features_are_strictly_prior() -> None:
    out = _horse_history(_horse_frame()).sort("race_date")
    # career_run_number counts only *previous* runs (0 on debut).
    assert out["career_run_number"].to_list() == [0, 1, 2]
    # win_rate_prior must exclude the current race's own result.
    win_rate = out["win_rate_prior"].to_list()
    assert win_rate[0] is None  # debut: no prior runs
    assert win_rate[1] == 1.0  # after 1 prior run (a win)
    assert win_rate[2] == 0.5  # after 2 prior runs (1 win of 2)
    # days since last run is the gap to the previous run, not 0 for the current.
    assert out["days_since_last_run"].to_list() == [None, 31, 29]


def test_recent_form_excludes_current_run() -> None:
    out = _horse_history(_horse_frame()).sort("race_date")
    # avg_finish_last3 on race 3 averages prior finishes {1, 5} = 3.0, not including the 1.
    assert out["avg_finish_last3"].to_list()[2] == 3.0


def _weight_frame() -> pl.DataFrame:
    # One horse, four runs; body weight 1000 -> 1010 -> 1002, then +20 after a long break.
    return pl.DataFrame(
        {
            "horse_id": ["H1"] * 4,
            "race_date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1), date(2020, 6, 1)],
            "race_no": [1, 1, 1, 1],
            "declared_weight": [1000, 1010, 1002, 1022],
            "days_since_last_run": [None, 31, 29, 92],
        }
    )


def test_weight_dynamics_use_prior_runs_only() -> None:
    out = _add_weight_dynamics(_weight_frame()).sort("race_date")
    # Change vs the previous run; undefined on debut.
    assert out["bw_chg"].to_list() == [None, 10.0, -8.0, 20.0]
    # Baseline is the mean of *prior* declared weights, excluding this run's own weight.
    assert out["bw_vs_avg"].to_list()[0] is None
    assert out["bw_vs_avg"].to_list()[1] == 10.0  # 1010 - mean(1000)
    assert out["bw_vs_avg"].to_list()[2] == 1002 - (1000 + 1010) / 2
    # Only pounds *lost* count; a gain is 0.
    assert out["bw_drop"].to_list() == [None, 0.0, 8.0, 0.0]
    # Heavier (>8lb) *and* off a >45-day break -> only the last run qualifies.
    assert out["bw_up_fresh"].to_list() == [0.0, 0.0, 0.0, 1.0]


def _class_frame() -> pl.DataFrame:
    # Two horses of the same trainer: H1 drops from Class 2 to Class 4 and wins; H2 drops later.
    return pl.DataFrame(
        {
            "_row": [0, 1, 2, 3],
            "horse_id": ["H1", "H1", "H2", "H2"],
            "race_date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 1, 1), date(2020, 3, 1)],
            "race_no": [1, 2, 3, 4],
            "race_class": ["Class 2", "Class 4", "Class 3", "Class 5"],
            "trainer_name": ["T1"] * 4,
            "trainer_code": [None, None, None, None],
            "won": [0, 1, 0, 0],
        }
    )


def test_class_drop_and_trainer_rate_are_strictly_prior() -> None:
    out = _add_class_drop(_class_frame()).sort("_row")
    # A drop is measured against the horse's own previous race (none on debut).
    assert out["is_drop"].to_list() == [0, 1, 0, 1]
    # H1's drop is the trainer's first ever class-drop runner -> no prior rate yet.
    assert out["drop_x_trsr"].to_list()[1] == 0.0
    # H2's later drop sees exactly one prior drop runner, which won -> rate 1.0.
    assert out["drop_x_trsr"].to_list()[3] == 1.0
    # Non-drop runs carry no trainer-deployment term at all.
    assert out["drop_x_trsr"].to_list()[0] == 0.0
    assert out["drop_x_trsr"].to_list()[2] == 0.0


def test_class_ordinal_ranks_company() -> None:
    classes = pl.DataFrame(
        {"race_class": ["Class 1", "Class 4 (Restricted)", "Group One", "Griffin Race", None]}
    )
    got = classes.select(_class_ord(pl.col("race_class")))["race_class"].to_list()
    assert got == [1.0, 4.0, 0.5, 4.5, 3.0]


def _pace_runs() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            "venue": ["ST"] * 3,
            "race_no": [1, 1, 1],
            "saddle": [1, 1, 1],
            "horse_id": ["H1"] * 3,
        }
    )


def test_pace_aggregates_are_lagged(monkeypatch: pytest.MonkeyPatch) -> None:
    # A sectional describes the run it belongs to, so the aggregate a race sees must summarise
    # the horse's *previous* runs only (same discipline as the NLP group).
    metrics = pl.DataFrame(
        {
            "race_date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            "venue": ["ST"] * 3,
            "race_no": [1, 1, 1],
            "saddle": [1, 1, 1],
            "late_rel": [0.2, 0.6, 1.0],
            "pace_close": [0.4, 0.8, 1.2],
            "led_held_hp": [1.0, 0.0, 1.0],
            "hidden_hp": [0.0, 1.0, 1.0],
        }
    )
    monkeypatch.setattr(build, "_pace_metrics", lambda _cfg: metrics)
    out = build._add_pace(_pace_runs(), cfg=None).sort("race_date")  # type: ignore[arg-type]
    assert out["late_rel3"].to_list() == [None, 0.2, pytest.approx(0.4)]
    assert out["pace_close3"].to_list() == [None, 0.4, pytest.approx(0.6)]
    assert out["led_held3"].to_list() == [None, 1.0, 1.0]
    assert out["hidden_hp3"].to_list() == [None, 0.0, 1.0]
