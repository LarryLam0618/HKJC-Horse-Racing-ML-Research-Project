"""As-of feature builder (PLAN.md §2, §1H).

Produces one row per runner per race with leakage-safe features: every value is computable
from information available at ``event_time <= race_off_time``. The horse's own *prior* runs
are the backbone (career-stage + form), connections use rolling career-to-date strike rates
(NOT the current ``people`` snapshot, which would leak the future), the closing-line SP is
kept behind the market wall, and a deterministic noise ``canary`` rides along to prove the
pipeline is clean.

The heavy lifting is vectorised in Polars, reading the raw Parquet directly (with column
projection) -- much faster than a DuckDB ``union_by_name`` glob over the thousands of
per-horse files. As-of correctness rests on two disciplines:
* cumulative aggregates use ``cum_* - current`` (strictly prior), never the current row;
* per-partition rolling/cumulative columns are computed on a frame pre-sorted by date within
  the partition (horse, or jockey/trainer via a join-back on a stable row id).
"""

from __future__ import annotations

import re
from datetime import datetime
from fractions import Fraction

import polars as pl

from hkjc.common.config import AppConfig, get_config
from hkjc.data.store.writer import season_label
from hkjc.features import store
from hkjc.features.base import FEATURE_SPECS, NLP_FEATURES, PACE_FEATURES

# Lengths-behind-winner word tokens -> approximate lengths.
_LBW_WORDS: dict[str, float] = {
    "": 0.0,
    "-": 0.0,
    "---": 0.0,
    "NOSE": 0.02,
    "NSE": 0.02,
    "SHD": 0.05,
    "SH": 0.05,
    "HD": 0.1,
    "HEAD": 0.1,
    "NK": 0.3,
    "NECK": 0.3,
    "N": 0.3,
    "DH": 0.0,  # dead-heat with the horse ahead -> ~level
    "DIST": 30.0,
}
_FRAC_MAP = {"¼": "1/4", "½": "1/2", "¾": "3/4"}
_NUM_TOKEN_RE = re.compile(r"^[0-9 /]+$")


def lbw_to_lengths(raw: str | None) -> float | None:
    """Convert a beaten-lengths token (``1-1/2``, ``NK``, ``SH``, ``-``) to float lengths."""
    if raw is None:
        return None
    token = raw.strip().upper().replace("\xa0", " ")
    for uni, ascii_frac in _FRAC_MAP.items():
        token = token.replace(uni, " " + ascii_frac)
    token = token.strip()
    if token in _LBW_WORDS:
        return _LBW_WORDS[token]
    body = token.replace("-", " ").strip()
    if not _NUM_TOKEN_RE.match(body):
        return None
    try:
        return float(sum(Fraction(part) for part in body.split()))
    except (ValueError, ZeroDivisionError):
        return None


def _read_raw(cfg: AppConfig, table: str, columns: list[str]) -> pl.DataFrame:
    """Read a raw table's Parquet (recursively), projecting ``columns``.

    ``missing_columns="insert"`` tolerates schema drift across files (e.g. older ``results``
    predate the ``jockey_name``/``trainer_name`` columns) by filling absent columns with null
    -- the polars equivalent of DuckDB's ``union_by_name``.
    """
    base = cfg.paths.raw_dir / table
    if not base.exists():
        return pl.DataFrame()
    files = [str(p) for p in base.rglob("*.parquet")]
    if not files:
        return pl.DataFrame()
    return pl.read_parquet(files, columns=columns, missing_columns="insert")


def _places_expr(field: pl.Expr, cfg: AppConfig) -> pl.Expr:
    """Paid-place count as a Polars expression, per config field-size rules."""
    rules = sorted(cfg.backtest.place_rules.places_by_field_size, key=lambda r: r.min_runners)
    expr = pl.lit(0)
    for rule in rules:  # ascending so larger thresholds overwrite smaller ones
        expr = pl.when(field >= rule.min_runners).then(pl.lit(rule.places)).otherwise(expr)
    return expr.cast(pl.Int64)


_RESULTS_COLS = [
    "race_date",
    "venue",
    "race_no",
    "finish_pos",
    "saddle",
    "horse_id",
    "jockey_code",
    "jockey_name",
    "trainer_code",
    "trainer_name",
    "actual_weight",
    "declared_weight",
    "draw",
    "lbw_raw",
    "finish_time_s",
    "win_odds",
]


def _load_runs(cfg: AppConfig) -> pl.DataFrame:
    """Load the runner spine = results joined to race meta (one row per runner)."""
    results = _read_raw(cfg, "results", columns=_RESULTS_COLS)
    races = _read_raw(
        cfg,
        "races",
        columns=[
            "race_date",
            "venue",
            "race_no",
            "race_index",
            "distance_m",
            "going",
            "surface",
            "rail",
            "race_class",
        ],
    )
    if results.is_empty() or races.is_empty():
        return pl.DataFrame()
    runs = results.join(races, on=["race_date", "venue", "race_no"], how="left")
    return runs.filter(pl.col("horse_id").is_not_null())


def _rail_offset(rail: pl.Expr) -> pl.Expr:
    """Rail token (``A``, ``C``, ``A+3``, ``C+2``) -> metres of displacement (0 if none)."""
    return rail.str.extract(r"([+-]?\d+)", 1).cast(pl.Float64).fill_null(0.0)


def _race_context(runs: pl.DataFrame, cfg: AppConfig) -> pl.DataFrame:
    """Add field size, paid places, labels, per-run speed/lbw, and calendar fields."""
    runs = runs.with_columns(
        field_size=pl.len().over(["race_date", "venue", "race_no"]),
        season=pl.col("race_date").map_elements(season_label, return_dtype=pl.String),
        month=pl.col("race_date").dt.month(),
        day_of_week=pl.col("race_date").dt.weekday() - 1,  # polars: 1=Mon -> 0=Mon
        is_turf=pl.col("surface").str.to_uppercase().str.contains("TURF").cast(pl.Int64),
        rail_offset=_rail_offset(pl.col("rail")),
        lbw_len=pl.col("lbw_raw").map_elements(lbw_to_lengths, return_dtype=pl.Float64),
        speed=pl.when(pl.col("finish_time_s") > 0)
        .then(pl.col("distance_m") / pl.col("finish_time_s"))
        .otherwise(None),
    )
    runs = runs.with_columns(n_places=_places_expr(pl.col("field_size"), cfg))
    runs = runs.with_columns(
        won=(pl.col("finish_pos") == 1).fill_null(False).cast(pl.Int64),
        placed=(pl.col("finish_pos") <= pl.col("n_places")).fill_null(False).cast(pl.Int64),
        draw_rel=(pl.col("draw") / pl.col("field_size")),
        dist_band=((pl.col("distance_m") // 200) * 200),
    )
    return runs


def _horse_history(runs: pl.DataFrame) -> pl.DataFrame:
    """As-of horse features (career stage + prior form). Frame sorted by horse then date."""
    runs = runs.sort(["horse_id", "race_date", "race_no"])
    g = "horse_id"
    prior_runs = pl.col("race_date").cum_count().over(g) - 1
    is_first_of_season = (pl.col("season") != pl.col("season").shift(1).over(g)).fill_null(True)
    runs = runs.with_columns(
        career_run_number=prior_runs,
        days_since_debut=(
            pl.col("race_date") - pl.col("race_date").first().over(g)
        ).dt.total_days(),
        days_since_last_run=(
            pl.col("race_date") - pl.col("race_date").shift(1).over(g)
        ).dt.total_days(),
        _seasons_incl=is_first_of_season.cast(pl.Int64).cum_sum().over(g),
        _is_first_season=is_first_of_season.cast(pl.Int64),
        _prior_wins=(pl.col("won").cum_sum().over(g) - pl.col("won")),
        _prior_places=(pl.col("placed").cum_sum().over(g) - pl.col("placed")),
        avg_finish_last3=pl.col("finish_pos")
        .shift(1)
        .rolling_mean(window_size=3, min_samples=1)
        .over(g),
        avg_lbw_last3=pl.col("lbw_len").shift(1).rolling_mean(window_size=3, min_samples=1).over(g),
        recent_speed=pl.col("speed").shift(1).rolling_mean(window_size=3, min_samples=1).over(g),
        _band_runs=pl.col("race_date").cum_count().over([g, "dist_band"]) - 1,
        _band_places=(pl.col("placed").cum_sum().over([g, "dist_band"]) - pl.col("placed")),
        _going_runs=pl.col("race_date").cum_count().over([g, "going"]) - 1,
        _going_places=(pl.col("placed").cum_sum().over([g, "going"]) - pl.col("placed")),
    )
    runs = runs.with_columns(
        seasons_active=(pl.col("_seasons_incl") - pl.col("_is_first_season")),
        win_rate_prior=pl.when(pl.col("career_run_number") > 0)
        .then(pl.col("_prior_wins") / pl.col("career_run_number"))
        .otherwise(None),
        place_rate_prior=pl.when(pl.col("career_run_number") > 0)
        .then(pl.col("_prior_places") / pl.col("career_run_number"))
        .otherwise(None),
        dist_match_rate=pl.when(pl.col("_band_runs") > 0)
        .then(pl.col("_band_places") / pl.col("_band_runs"))
        .otherwise(None),
        going_match_rate=pl.when(pl.col("_going_runs") > 0)
        .then(pl.col("_going_places") / pl.col("_going_runs"))
        .otherwise(None),
    )
    return runs.drop(
        "_seasons_incl",
        "_is_first_season",
        "_prior_wins",
        "_prior_places",
        "_band_runs",
        "_band_places",
        "_going_runs",
        "_going_places",
    )


def _canonical_connection_key(keyed: pl.DataFrame, name_col: str, code_col: str) -> pl.DataFrame:
    """Add a stable ``_key`` per connection, merging the two id regimes.

    Old result pages carry the *name* (no link); recent pages carry the *code* (no name).
    A name->code map built from rows that have both lets us key a long-career jockey/trainer
    by a single identity instead of splitting their history at the era boundary. Fallback
    order: code -> code-mapped-from-name -> name.
    """
    pair = keyed.select(name_col, code_col).drop_nulls()
    if pair.height:
        name2code = pair.group_by(name_col).agg(pl.col(code_col).mode().first().alias("_mapped"))
        keyed = keyed.join(name2code, on=name_col, how="left")
    else:
        keyed = keyed.with_columns(pl.lit(None, dtype=pl.String).alias("_mapped"))
    return keyed.with_columns(
        _key=pl.coalesce([pl.col(code_col), pl.col("_mapped"), pl.col(name_col)])
    )


def _connection_rates(
    runs: pl.DataFrame, name_col: str, code_col: str, prefix: str
) -> pl.DataFrame:
    """Rolling career-to-date win rate for a connection (jockey/trainer), as-of and joined
    back by the stable ``_row`` id."""
    keyed = _canonical_connection_key(
        runs.select("_row", name_col, code_col, "race_date", "race_no", "won"),
        name_col,
        code_col,
    )
    sub = keyed.filter(pl.col("_key").is_not_null())
    if sub.is_empty():
        return runs.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_win_rate"),
            pl.lit(0, dtype=pl.Int64).alias(f"{prefix}_runs_prior"),
        )
    sub = sub.sort(["_key", "race_date", "race_no"]).with_columns(
        runs_prior=pl.col("race_date").cum_count().over("_key") - 1,
        prior_wins=pl.col("won").cum_sum().over("_key") - pl.col("won"),
    )
    sub = sub.with_columns(
        win_rate=pl.when(pl.col("runs_prior") > 0)
        .then(pl.col("prior_wins") / pl.col("runs_prior"))
        .otherwise(None)
    ).select(
        "_row",
        pl.col("win_rate").alias(f"{prefix}_win_rate"),
        pl.col("runs_prior").alias(f"{prefix}_runs_prior"),
    )
    return runs.join(sub, on="_row", how="left").with_columns(
        pl.col(f"{prefix}_runs_prior").fill_null(0)
    )


def _add_rating(runs: pl.DataFrame, cfg: AppConfig) -> pl.DataFrame:
    """As-of official rating (going into the race) from horse_form, joined on race_index."""
    form = _read_raw(cfg, "horse_form", columns=["horse_id", "race_index", "rating"])
    if form.is_empty():
        runs = runs.with_columns(pl.lit(None, dtype=pl.Int64).alias("as_of_rating"))
    else:
        ratings = (
            form.filter(pl.col("race_index").is_not_null())
            .unique(subset=["horse_id", "race_index"], keep="first")
            .rename({"rating": "as_of_rating"})
        )
        runs = runs.join(ratings, on=["horse_id", "race_index"], how="left")
    # Forward race-day cards (M7) have no stored race_index to join on, so they carry their
    # own as-of rating from the GraphQL card -- fold it in before the trend is computed.
    if "_card_rating" in runs.columns:
        runs = runs.with_columns(
            as_of_rating=pl.coalesce([pl.col("as_of_rating"), pl.col("_card_rating")])
        )
    runs = runs.sort(["horse_id", "race_date", "race_no"]).with_columns(
        rating_trend3=(
            pl.col("as_of_rating") - pl.col("as_of_rating").shift(3).over("horse_id")
        ).cast(pl.Float64)
    )
    return runs


def _add_bio(runs: pl.DataFrame, cfg: AppConfig) -> pl.DataFrame:
    """Bio block: age-at-race (birth_year anchor + imputed flag), pedigree/categoricals."""
    cols = [
        "horse_id",
        "birth_year",
        "import_type",
        "sex",
        "country_of_origin",
        "sire",
        "dam",
        "dams_sire",
    ]
    bio = _read_raw(cfg, "horses", columns=cols)
    if bio.is_empty():
        bio = pl.DataFrame(schema=dict.fromkeys(cols, pl.String))
    runs = runs.join(bio, on="horse_id", how="left")
    # Layered age-at-race (decided 2026-06-15): exact birth_year where the horse was active at
    # scrape; else a debut-age heuristic (HK-debut year - typical debut age by import type:
    # ~3 for griffin imports PPG/ISG, ~4 for previously-raced PP); else null. age_imputed
    # flags everything that is not the exact scraped value, so models can down-weight it.
    debut_year = pl.col("race_date").dt.year().min().over("horse_id")
    debut_age = (
        pl.when(pl.col("import_type").is_in(["PPG", "ISG"]))
        .then(3)
        .when(pl.col("import_type") == "PP")
        .then(4)
        .otherwise(4)
    )
    birth_year_est = (debut_year - debut_age).cast(pl.Int64)
    runs = runs.with_columns(
        age_imputed=pl.col("birth_year").is_null().cast(pl.Int64),
        _birth_year=pl.coalesce([pl.col("birth_year"), birth_year_est]),
    )
    runs = runs.with_columns(
        age_at_race=(pl.col("race_date").dt.year() - pl.col("_birth_year")).cast(pl.Float64)
    )
    return runs.drop("_birth_year")


def _add_trial_recency(runs: pl.DataFrame, cfg: AppConfig) -> pl.DataFrame:
    """Days since most recent prior barrier trial (as-of join by horse_id)."""
    trials = _read_raw(cfg, "barrier_trials", columns=["horse_id", "trial_date"])
    if trials.is_empty():
        return runs.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("days_since_trial"),
            pl.lit(0, dtype=pl.Int64).alias("had_recent_trial"),
        )
    t = trials.filter(pl.col("horse_id").is_not_null()).unique().sort(["trial_date"])
    joined = runs.sort(["race_date"]).join_asof(
        t, left_on="race_date", right_on="trial_date", by="horse_id", strategy="backward"
    )
    joined = joined.with_columns(
        days_since_trial=(pl.col("race_date") - pl.col("trial_date")).dt.total_days()
    )
    return joined.with_columns(
        had_recent_trial=(
            pl.col("days_since_trial").is_not_null() & (pl.col("days_since_trial") <= 60)
        ).cast(pl.Int64)
    ).drop("trial_date")


def _add_context(runs: pl.DataFrame, cfg: AppConfig) -> pl.DataFrame:
    """Weather, public holiday, trial recency, market wall, and the leakage canary."""
    weather = _read_raw(cfg, "weather", columns=["venue", "date", "mean_temp"])
    if not weather.is_empty():
        w = weather.rename({"date": "race_date"}).unique(
            subset=["venue", "race_date"], keep="first"
        )
        runs = runs.join(w, on=["venue", "race_date"], how="left")
    else:
        runs = runs.with_columns(pl.lit(None, dtype=pl.Float64).alias("mean_temp"))

    holidays = _read_raw(cfg, "public_holidays", columns=["date"])
    if not holidays.is_empty():
        hol = (
            holidays.rename({"date": "race_date"})
            .unique()
            .with_columns(is_public_holiday=pl.lit(1, dtype=pl.Int64))
        )
        runs = runs.join(hol, on="race_date", how="left").with_columns(
            pl.col("is_public_holiday").fill_null(0)
        )
    else:
        runs = runs.with_columns(pl.lit(0, dtype=pl.Int64).alias("is_public_holiday"))

    runs = _add_trial_recency(runs, cfg)

    # Market wall: SP -> overround-adjusted implied win prob (kept for the blend policy only).
    inv = pl.when(pl.col("win_odds") > 0).then(1.0 / pl.col("win_odds")).otherwise(None)
    runs = runs.with_columns(_inv_odds=inv)
    runs = runs.with_columns(
        market_prob=(
            pl.col("_inv_odds") / pl.col("_inv_odds").sum().over(["race_date", "venue", "race_no"])
        )
    ).drop("_inv_odds")

    # Deterministic noise canary: must score ~0 (PLAN.md §1H).
    key = pl.concat_str(
        [
            pl.col("race_date").cast(pl.String),
            pl.col("venue"),
            pl.col("race_no").cast(pl.String),
            pl.col("saddle").cast(pl.String),
        ],
        separator="|",
    )
    return runs.with_columns(canary_random=(key.hash(seed=1234) % 1_000_000) / 1_000_000.0)


def _add_nlp(runs: pl.DataFrame, cfg: AppConfig) -> pl.DataFrame:
    """Join the per-run comment NLP features and **lag** them (prior run only, PLAN.md §1C).

    A comment describes the run it belongs to, so each NLP signal is shifted one run forward
    per horse -- the value seen for a target race is the horse's *previous* comment.
    """
    from hkjc.features.nlp.build import build_comment_features, comment_features_path

    keys = ["race_date", "venue", "race_no", "horse_id"]
    path = comment_features_path(cfg)
    cf = pl.read_parquet(path) if path.is_file() else build_comment_features(cfg, persist=True)
    if cf.is_empty():
        return runs.with_columns(*[pl.lit(None, dtype=pl.Float64).alias(c) for c in NLP_FEATURES])
    cf = cf.select([*keys, *NLP_FEATURES]).unique(subset=keys, keep="first")
    runs = runs.join(cf, on=keys, how="left")
    return runs.sort(["horse_id", "race_date", "race_no"]).with_columns(
        *[pl.col(c).shift(1).over("horse_id").alias(c) for c in NLP_FEATURES]
    )


_SECTIONAL_COLS = [
    "race_date",
    "venue",
    "race_no",
    "saddle",
    "finishing_order",
    "section_index",
    "running_position",
    "section_time_s",
]
_PACE_WINDOW = 4  # prior runs averaged into the *3 aggregates (as in reports/w456.py)


def _pace_metrics(cfg: AppConfig) -> pl.DataFrame:
    """Per-run pace / closing metrics from the sectional archive (#7; starts 2008-04-02).

    Everything here describes the run it belongs to (post-race), so it is *never* used raw --
    :func:`_add_pace` lags it one run per horse before it reaches the model.

    * ``pace_press`` -- how fast the race's leader went early, z-scored within
      (distance, number of sections); +ve = a hot early pace.
    * ``late_rel``   -- the runner's last-section time vs the field median (+ve = finished faster).
    * ``pace_close`` -- ``late_rel`` amplified when the pace was hot (closing into a fast pace is
      worth more than closing into a crawl).
    * ``led_held_hp``/``hidden_hp`` -- flags for a front-runner that withstood a fast pace, and
      for a fast finish hidden behind a bad placing in a truly-run race.
    """
    sec = _read_raw(cfg, "sectionals", columns=_SECTIONAL_COLS)
    if sec.is_empty():
        return pl.DataFrame()
    keys = ["race_date", "venue", "race_no", "saddle"]
    race_keys = ["race_date", "venue", "race_no"]
    sec = sec.drop_nulls(["section_index", "section_time_s"])
    if sec.is_empty():
        return pl.DataFrame()
    sec = sec.with_columns(nsec=pl.col("section_index").max().over(keys))
    last = (
        sec.filter(pl.col("section_index") == pl.col("nsec"))
        .select(*keys, "nsec", "finishing_order", pl.col("section_time_s").alias("last400"))
        .unique(subset=keys, keep="first")
    )
    first = (
        sec.filter(pl.col("section_index") == 1)
        .select(
            *keys,
            pl.col("section_time_s").alias("sec1"),
            pl.col("running_position").alias("early_pos"),
        )
        .unique(subset=keys, keep="first")
    )
    races = _read_raw(cfg, "races", columns=[*race_keys, "distance_m"]).unique(
        subset=race_keys, keep="first"
    )
    run = last.join(first, on=keys, how="left")
    if not races.is_empty():
        run = run.join(races, on=race_keys, how="left")
    else:
        run = run.with_columns(pl.lit(None, dtype=pl.Int64).alias("distance_m"))

    # Early-pace pressure is a *race* property: the leader's first-section time, z-scored
    # within comparable races (same distance and same number of timed sections).
    lead = pl.col("sec1").min().over(race_keys)
    run = run.with_columns(_lead_sec1=lead)
    norm = ["distance_m", "nsec"]
    run = run.with_columns(
        pace_press=-(
            (pl.col("_lead_sec1") - pl.col("_lead_sec1").mean().over(norm))
            / pl.col("_lead_sec1").std().over(norm)
        )
    )
    run = run.with_columns(late_rel=pl.col("last400").median().over(race_keys) - pl.col("last400"))
    run = run.with_columns(
        pace_close=pl.col("late_rel") * (1.0 + pl.col("pace_press").clip(0.0, 2.5)),
        led_held_hp=(
            (pl.col("early_pos") <= 3)
            & (pl.col("finishing_order") <= 3)
            & (pl.col("pace_press") > 0.4)
        )
        .fill_null(value=False)
        .cast(pl.Float64),
        hidden_hp=(
            (pl.col("finishing_order") >= 6)
            & (pl.col("late_rel") >= 0.35)
            & (pl.col("pace_press") > 0.0)
        )
        .fill_null(value=False)
        .cast(pl.Float64),
    )
    return run.select(*keys, "late_rel", "pace_close", "led_held_hp", "hidden_hp")


def _add_pace(runs: pl.DataFrame, cfg: AppConfig) -> pl.DataFrame:
    """Join the per-run pace metrics and **lag** them (prior runs only, like the NLP group).

    A sectional describes the run it belongs to, so each aggregate is shifted one run forward
    per horse: the value seen for a target race summarises the horse's previous runs. Runs with
    no sectional row (pre-2008-04, or a missing page) simply contribute nothing to the window.
    """
    keys = ["race_date", "venue", "race_no", "saddle"]
    metrics = _pace_metrics(cfg)
    if metrics.is_empty():
        return runs.with_columns(*[pl.lit(None, dtype=pl.Float64).alias(c) for c in PACE_FEATURES])
    runs = runs.join(metrics, on=keys, how="left").sort(["horse_id", "race_date", "race_no"])
    g = "horse_id"
    w = _PACE_WINDOW
    runs = runs.with_columns(
        late_rel3=pl.col("late_rel").shift(1).rolling_mean(window_size=w, min_samples=1).over(g),
        pace_close3=pl.col("pace_close")
        .shift(1)
        .rolling_mean(window_size=w, min_samples=1)
        .over(g),
        led_held3=pl.col("led_held_hp").shift(1).rolling_sum(window_size=w, min_samples=1).over(g),
        hidden_hp3=pl.col("hidden_hp").shift(1).rolling_sum(window_size=w, min_samples=1).over(g),
    )
    return runs.drop("late_rel", "pace_close", "led_held_hp", "hidden_hp")


def _add_weight_dynamics(runs: pl.DataFrame) -> pl.DataFrame:
    """Body-weight dynamics vs the horse's own prior runs (strictly prior; declared weight).

    ``declared_weight`` is the horse's body weight in lbs, published pre-race, so the *current*
    value is legal; only the comparison baselines come from earlier runs. ``bw_up_fresh`` is the
    combination the study found to matter -- returning heavier *and* off a real break.
    """
    g = "horse_id"
    runs = runs.sort([g, "race_date", "race_no"])
    bw = pl.col("declared_weight").cast(pl.Float64)
    runs = runs.with_columns(
        _bw=bw,
        _bw_prev=bw.shift(1).over(g),
        _bw_sum_prior=bw.fill_null(0.0).shift(1).cum_sum().over(g),
        _bw_n_prior=bw.is_not_null().cast(pl.Int64).shift(1).cum_sum().over(g),
    )
    runs = runs.with_columns(bw_chg=pl.col("_bw") - pl.col("_bw_prev"))
    runs = runs.with_columns(
        bw_vs_avg=pl.when(pl.col("_bw_n_prior") > 0)
        .then(pl.col("_bw") - pl.col("_bw_sum_prior") / pl.col("_bw_n_prior"))
        .otherwise(None),
        bw_up_fresh=((pl.col("bw_chg") > 8) & (pl.col("days_since_last_run") > 45))
        .fill_null(value=False)
        .cast(pl.Float64),
        bw_drop=(-pl.col("bw_chg")).clip(lower_bound=0.0),
    )
    return runs.drop("_bw", "_bw_prev", "_bw_sum_prior", "_bw_n_prior")


def _class_ord(race_class: pl.Expr) -> pl.Expr:
    """Class token -> ordinal (lower = better company): Group 0.5, Class N -> N, Griffin 4.5."""
    numbered = race_class.str.extract(r"^Class\s+(\d)", 1).cast(pl.Float64)
    return (
        pl.when(numbered.is_not_null())
        .then(numbered)
        .when(race_class.str.contains("Group"))
        .then(pl.lit(0.5))
        .when(race_class.str.contains("Griffin"))
        .then(pl.lit(4.5))
        .otherwise(pl.lit(3.0))
    )


def _add_class_drop(runs: pl.DataFrame) -> pl.DataFrame:
    """Class-drop flag + the trainer's as-of strike rate *on class-drop runners*.

    Both sides are strictly prior: the drop compares this race's class with the horse's previous
    race, and the trainer rate is a running career-to-date figure over their earlier class-drop
    runners only (keyed by the canonical connection id, so a long career is not split at the
    name/code era boundary).
    """
    g = "horse_id"
    runs = runs.sort([g, "race_date", "race_no"]).with_columns(
        _class_ord=_class_ord(pl.col("race_class"))
    )
    runs = runs.with_columns(_prev_class=pl.col("_class_ord").shift(1).over(g))
    runs = runs.with_columns(
        is_drop=((pl.col("_class_ord") - pl.col("_prev_class")) > 0)
        .fill_null(value=False)
        .cast(pl.Int64)
    )
    keyed = _canonical_connection_key(
        runs.select(
            "_row", "trainer_name", "trainer_code", "race_date", "race_no", "won", "is_drop"
        ),
        "trainer_name",
        "trainer_code",
    )
    drops = keyed.filter((pl.col("is_drop") == 1) & pl.col("_key").is_not_null())
    if drops.is_empty():
        runs = runs.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("_tr_drop_sr"))
    else:
        drops = drops.sort(["_key", "race_date", "race_no"]).with_columns(
            _n_prior=pl.col("race_date").cum_count().over("_key") - 1,
            _w_prior=pl.col("won").cum_sum().over("_key") - pl.col("won"),
        )
        drops = drops.with_columns(
            _tr_drop_sr=pl.when(pl.col("_n_prior") > 0)
            .then(pl.col("_w_prior") / pl.col("_n_prior"))
            .otherwise(0.0)
        ).select("_row", "_tr_drop_sr")
        runs = runs.join(drops, on="_row", how="left").with_columns(
            pl.col("_tr_drop_sr").fill_null(0.0)
        )
    runs = runs.with_columns(
        drop_x_trsr=(pl.col("is_drop").cast(pl.Float64) * pl.col("_tr_drop_sr"))
    )
    return runs.drop("_class_ord", "_prev_class", "_tr_drop_sr")


def _compute_features(runs: pl.DataFrame, cfg: AppConfig) -> pl.DataFrame:
    """The as-of feature pipeline (shared by the historical build + the M7 forward card)."""
    runs = runs.with_row_index("_row")
    runs = _race_context(runs, cfg)
    runs = _horse_history(runs)
    runs = _connection_rates(runs, "jockey_name", "jockey_code", "jockey")
    runs = _connection_rates(runs, "trainer_name", "trainer_code", "trainer")
    runs = _add_rating(runs, cfg)
    runs = _add_bio(runs, cfg)
    runs = _add_context(runs, cfg)
    runs = _add_nlp(runs, cfg)
    runs = _add_pace(runs, cfg)
    runs = _add_weight_dynamics(runs)
    runs = _add_class_drop(runs)
    return runs


def build_features(cfg: AppConfig | None = None, *, persist: bool = True) -> pl.DataFrame:
    """Build the as-of ``features_runner`` table and (optionally) persist it.

    Returns the feature frame with exactly the columns declared in
    :data:`hkjc.features.base.FEATURE_SPECS`, plus ``feature_version`` and ``computed_at``.
    """
    cfg = cfg or get_config()
    runs = _load_runs(cfg)
    if runs.is_empty():
        msg = "No results/races stored; run the M1 scraper first."
        raise RuntimeError(msg)
    runs = _compute_features(runs, cfg)

    wanted = [spec.name for spec in FEATURE_SPECS]
    for name in wanted:
        if name not in runs.columns:
            runs = runs.with_columns(pl.lit(None).alias(name))
    out = (
        runs.select(wanted)
        .with_columns(
            feature_version=pl.lit(cfg.features.feature_version),
            computed_at=pl.lit(datetime.now().isoformat(timespec="seconds")),
        )
        .sort(["race_date", "venue", "race_no", "saddle"])
    )
    if persist:
        store.write_features(out, cfg)
    return out


def build_forward_features(
    forward: pl.DataFrame, cfg: AppConfig | None = None, *, history: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Compute as-of features for an upcoming card's runners (M7 race-day).

    ``forward`` is one row per upcoming runner with the runner-spine columns (race_date, venue,
    race_no, saddle, horse_id, jockey/trainer name+code, draw, distance_m, going, surface, rail,
    race_class, plus ``_card_rating`` from the GraphQL card); outcome columns are simply absent.
    The card's runners are appended to the historical runner spine so each horse's **prior**
    aggregates pick up its real history, the shared pipeline runs, and the forward rows are
    returned with the same columns as :func:`build_features`. In production the upcoming meeting
    is not yet in ``results`` (no overlap); ``history`` lets a test truncate the spine.
    """
    cfg = cfg or get_config()
    hist = history if history is not None else _load_runs(cfg)
    if hist.is_empty():
        msg = "No historical results stored; run the M1 scraper first."
        raise RuntimeError(msg)
    combined = pl.concat(
        [hist.with_columns(_forward=pl.lit(False)), forward.with_columns(_forward=pl.lit(True))],
        how="diagonal_relaxed",
    )
    out = _compute_features(combined, cfg).filter(pl.col("_forward"))
    wanted = [spec.name for spec in FEATURE_SPECS]
    for name in wanted:
        if name not in out.columns:
            out = out.with_columns(pl.lit(None).alias(name))
    return out.select(wanted).with_columns(feature_version=pl.lit(cfg.features.feature_version))
