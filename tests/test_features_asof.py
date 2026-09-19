"""As-of correctness tests for the feature builder (PLAN.md §1H): the leakage-critical
guarantee that horse-history features use *strictly prior* runs only."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from hkjc.features import build
from hkjc.features.build import (
    _add_class_drop,
    _add_trial_signal,
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
        {
            "race_class": [
                "Class 1",
                "Class 4 (Restricted)",
                "Group One",
                "GROUP-3",
                "Griffin Race",
                "4 Year Olds",  # age/condition race: company unknown, not "Class 3"
                None,  # the M7 card carries no race class
            ]
        }
    )
    got = classes.select(_class_ord(pl.col("race_class")))["race_class"].to_list()
    assert got == [1.0, 4.0, 0.5, 0.5, 4.5, None, None]


def test_class_drop_is_unknown_when_card_has_no_class() -> None:
    # Race-day rows carry race_class=None. A null class must not be read as a middle class:
    # an ex-Class-2 horse is *not* a drop and an ex-Class-5 horse is *not* a rise.
    runs = pl.DataFrame(
        {
            "_row": [0, 1, 2, 3],
            "horse_id": ["A", "A", "B", "B"],
            "race_date": [date(2024, 1, 1), date(2024, 2, 1)] * 2,
            "race_no": [1, 1, 2, 2],
            "race_class": ["Class 2", None, "Class 5", None],
            "trainer_name": ["T"] * 4,
            "trainer_code": [None] * 4,
            "won": [0, 0, 0, 0],
        }
    )
    out = _add_class_drop(runs).sort("_row")
    assert out["is_drop"].to_list() == [0, 0, 0, 0]
    assert out["drop_x_trsr"].to_list() == [0.0, 0.0, 0.0, 0.0]


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


def _sectionals(n_races: int) -> pl.DataFrame:
    # n_races races at the same distance, one runner each, two sections; the leader's first
    # section gets progressively faster so the expanding z-score is non-trivial.
    rows = []
    for i in range(n_races):
        for sec, t in ((1, 24.0 - i), (2, 23.0)):
            rows.append(
                {
                    "race_date": date(2020, 1, 1 + i),
                    "venue": "ST",
                    "race_no": 1,
                    "saddle": 1,
                    "finishing_order": 1,
                    "section_index": sec,
                    "running_position": 1,
                    "section_time_s": t,
                }
            )
    return pl.DataFrame(rows)


def _fake_read_raw(sectionals: pl.DataFrame) -> object:
    races = (
        sectionals.select("race_date", "venue", "race_no")
        .unique()
        .with_columns(distance_m=pl.lit(1200))
    )
    tables = {"sectionals": sectionals, "races": races}

    def _read(_cfg: object, table: str, columns: list[str]) -> pl.DataFrame:
        return tables[table].select([c for c in columns if c in tables[table].columns])

    return _read


def test_pace_pressure_is_as_of(monkeypatch: pytest.MonkeyPatch) -> None:
    # The z-score for a race must only depend on races run up to and including it: adding
    # later races must not change an earlier race's pace_press (repo as-of rule).
    monkeypatch.setattr(build, "_read_raw", _fake_read_raw(_sectionals(3)))
    early = build._pace_metrics(cfg=None).sort("race_date")  # type: ignore[arg-type]
    monkeypatch.setattr(build, "_read_raw", _fake_read_raw(_sectionals(6)))
    late = build._pace_metrics(cfg=None).sort("race_date")  # type: ignore[arg-type]
    # pace_close = late_rel * (1 + clip(pace_press)); late_rel is 0 for a lone runner, so
    # compare the flags' driver via led_held_hp, which needs pace_press > 0.4, and the row count.
    assert early.height == 3 and late.height == 6
    for col in ("late_rel", "pace_close", "led_held_hp", "hidden_hp"):
        assert early[col].to_list() == late[col].to_list()[:3], col


def test_pace_pressure_neutral_when_undefined(monkeypatch: pytest.MonkeyPatch) -> None:
    # With a single comparable race there is no spread to standardise against: the pressure
    # is the neutral 0, so pace_close == late_rel and the flags cannot fire (not null).
    monkeypatch.setattr(build, "_read_raw", _fake_read_raw(_sectionals(1)))
    out = build._pace_metrics(cfg=None)  # type: ignore[arg-type]
    assert out.height == 1
    assert out["pace_close"].to_list() == out["late_rel"].to_list()
    assert out["led_held_hp"].to_list() == [0.0]
    assert out["hidden_hp"].to_list() == [0.0]


def test_trial_signal_uses_prior_trials_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Horse H: race 1 on 2024-01-10, race 2 on 2024-04-20 (100-day break). Trials: one before
    # race 1 (won easily), two between the races (one Failed), one ON race-day 2 (must be
    # ignored: not strictly prior), one after (must never appear).
    trials = pl.DataFrame(
        {
            "horse_id": ["H"] * 5 + ["X"] * 2,
            "trial_date": [
                date(2024, 1, 2),
                date(2024, 3, 1),
                date(2024, 4, 5),
                date(2024, 4, 20),
                date(2024, 5, 1),
                date(2024, 1, 2),
                date(2024, 4, 5),
            ],
            "location": ["SHA TIN ALL WEATHER TRACK"] * 7,
            "batch": [1, 1, 1, 1, 1, 1, 1],
            "time_s": [70.0, 71.0, 70.5, 70.0, 70.0, 71.0, 70.0],
            "result": [None, "Failed", "Passed", None, None, None, None],
            "comment": ["Led throughout; won easily.", "Green.", "Kept on.", "", "", "", ""],
        }
    )
    monkeypatch.setattr(build, "_read_raw", lambda _cfg, _t, columns: trials.select(columns))
    runs = pl.DataFrame(
        {
            "_row": [0, 1],
            "horse_id": ["H", "H"],
            "race_date": [date(2024, 1, 10), date(2024, 4, 20)],
            "race_no": [1, 1],
            "days_since_last_run": [None, 100],
        }
    )
    out = _add_trial_signal(runs, cfg=None).sort("_row")  # type: ignore[arg-type]
    # race 1: latest trial 2024-01-02 (8 days) -- H beat X in its batch, "won easily".
    assert out["bt_easy_win"].to_list()[0] == 1.0
    assert out["bt_margin_last"].to_list()[0] == 0.0
    assert out["bt_rank_last"].to_list()[0] == 0.0
    assert out["bt_n_between"].to_list()[0] == 1.0  # no previous race -> everything prior counts
    assert out["bt_failed_between"].to_list()[0] == 0.0
    assert out["bt_first_up_trial"].to_list()[0] == 1.0  # debut counts as a break with a trial
    # race 2: the race-day trial is NOT the latest (gap 0 is not prior); latest = 2024-04-05,
    # where H (70.5) lost to X (70.0) -> margin 0.5, rank 1/1 = 1.0, not an easy win.
    assert out["bt_margin_last"].to_list()[1] == pytest.approx(0.5)
    assert out["bt_rank_last"].to_list()[1] == 1.0
    assert out["bt_easy_win"].to_list()[1] == 0.0
    assert out["bt_n_between"].to_list()[1] == 2.0  # 03-01 and 04-05; not 04-20, not 05-01
    assert out["bt_failed_between"].to_list()[1] == 1.0
    assert out["bt_first_up_trial"].to_list()[1] == 1.0


def test_trial_signal_without_archive_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build, "_read_raw", lambda _cfg, _t, columns: pl.DataFrame())
    runs = pl.DataFrame(
        {
            "_row": [0],
            "horse_id": ["H"],
            "race_date": [date(2024, 1, 10)],
            "race_no": [1],
            "days_since_last_run": [30],
        }
    )
    out = _add_trial_signal(runs, cfg=None)  # type: ignore[arg-type]
    assert out["bt_n_between"].to_list() == [None]
