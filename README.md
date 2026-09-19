# HKJC Horse-Racing ML Research Platform

A **local, single-user research platform** that predicts HKJC **WIN + PLACE** probabilities
for Sha Tin (`ST`) and Happy Valley (`HV`) races, detects value against the live
pari-mutuel odds, sizes stakes with Kelly variants, and **backtests honestly**.

> A methodology / research sandbox for **market-efficiency and probabilistic-modelling**
> questions on pari-mutuel data. It **recommends only; it never places bets** (there is no
> bet/submit path anywhere in the codebase), and it is not financial advice.

## About this fork

This is a fork of **[stevw-repo/HKJC-Horse-Racing-ML-Research-Project](https://github.com/stevw-repo/HKJC-Horse-Racing-ML-Research-Project)**
by Steven Wang (MIT). The platform itself — the M0–M7 build below: scraper + data lake,
as-of feature store, model zoo, honest walk-forward backtest, risk layer, dashboards and
live ops — is upstream's work and is kept intact. Everything from [Status](#status) down is
the upstream README.

### What this fork adds

- **A 13-factor "residual" feature group, wired into the pipeline** (`src/hkjc/features/`,
  `config/features.yaml`, +108 lines of tests in `tests/test_features_asof.py`). A standalone
  study (`reports/w456.py`: 13 handicapping factors in one conditional logit with the market
  as a control) was promoted into a first-class, ablatable group in three sub-groups:
  - `pace_sectional` — early-pace pressure, late-section relative speed, "closed into a hot
    pace", led-and-held — from the per-200m sectional archive, **lagged one run and rolled over
    the horse's last 4 runs** (a sectional describes the run it belongs to, so it is only legal
    for later races).
  - `weight_dynamics` — body-weight change vs last run / vs career average, "heavier and
    fresh" (>8 lb and >45 days off), sharp drop.
  - `class_deploy` — class-drop flag × the trainer's as-of strike rate *on their earlier
    class-drop runners only*, keyed by canonical connection id.

  Threaded through `build_design` / `load_model_data` / `train_production_model --residual` /
  `hkjc ablate --group residual|nlp` so race-day rebuilds a matching design and the leakage
  canary rides through the augmented fit. **Result:** betas reproduce the study (`late_rel3`
  +0.152, t = 3.5), canary stays clean (0.050 → 0.054), and on 14,434 OOS races log-loss
  2.2485 → 2.2481 (−0.0004), model-only WIN ROI −16.30% → −16.80%, market-blend WIN ROI
  −32.81% → −31.62%. Marginal at best — the honest verdict is unchanged: **no edge past the
  takeout.** Full write-up in
  [`CLAUDE.md`](CLAUDE.md#13-factor-residual-group-post-m7----wired-into-the-pipeline).

  *A note on honesty:* the first cut reported a larger gain (log-loss −0.0025, ROI +0.15pp).
  Reviewing the code found `pace_press` was standardised against the mean/std of *all* races
  in its bucket — future seasons included — a distributional leak the outcome-level canary
  cannot detect. Making the normalisation strictly as-of (expanding window) erased most of
  the improvement. That correction, and the train/serve-skew fixes on the race-day path
  (null class read as "Class 3"; card body weight fetched but dropped), are in the follow-up
  commit and are now pinned by tests.
- **Kelly test hardening** (`tests/test_risk_kelly.py`): the scipy reference optimiser used to
  sanity-check the closed-form simultaneous Kelly could stall at 0 when the optimum sits near
  the `sum(f) = 1` boundary (a Hypothesis-found case where the closed form was right), failing
  CI. It now uses a finite penalty instead of `-inf`, an extra start seeded near the candidate,
  and accepts feasible non-converged iterates.
- **macOS race-day automation** (`scripts/`): a `launchd` plist + installer that runs the
  live-odds logger over a meeting window (with `caffeinate` so idle sleep can't stall polls),
  plus `todays_meeting.py`. Upstream shipped only the Windows Task Scheduler script.
- **Draw-bias study** (`reports/draw_bias.py`, null result): strictly as-of, shrunk win / place
  share of each barrier per (venue × surface × distance), with the closing-line market as a
  control. The raw bias is real and matches racecourse lore (Sha Tin 1000m: inside draws −10–18%,
  outside +13–22% vs field-size expectation) but once the market is controlled for it carries
  no residual information (`draw_win_bias` t = 0.8, `draw_place_bias` t = 0.5; baseline
  `draw_rel` itself only t = 2.4). Walk-forward EV-gated betting finds ~200 bets in 14 seasons
  with a CI spanning −39% to +23%. Not wired into the pipeline — three t < 1 columns would only
  add noise. Third honest null after NLP and the residual group.
- **Barrier-trial backfill + `trial_signal` feature group** (the strongest of the four
  studies so far). The `btresult` landing page only lists ~one season, but HKJC still serves
  per-date trial pages back to 2010-11, so `hkjc scrape-trials --since` now enumerates the gap
  day by day: **8,022 → 83,903 trial rows, 1 → 16 seasons**. `reports/trial_signal.py` then
  asks what a horse *did* in its trials since its last race, with the market **and the layoff**
  as controls (trials cluster around spells). Two factors survive: `bt_n_between` (trials
  between races, **t = 3.8**) and `bt_easy_win` (won a trial "easily", **t = −2.6** — the
  market over-backs it). Wired in as an ablatable 6-column group (`src/hkjc/features/build.py::
  _add_trial_signal`, strictly prior via `join_asof(allow_exact_matches=False)`, tests pin
  the as-of rules). **Ablation on 14,434 OOS races:** log-loss 2.2465 → **2.2377 (−0.0089)**,
  ~7× the NLP group's gain and 20× the residual group's; top-1 +0.0012; market-blend WIN ROI
  −32.9% → −30.7%; canary clean (0.051). Yet model-only WIN ROI −17.0% → −18.0%: better
  probabilities, still **no edge past the takeout**. The backfill also moved the *baseline*
  (its `had_recent_trial` went from one season to sixteen): 2.2485 → 2.2465.
- **Research scripts** (`research/`, `reports/`): the exploration behind the residual group —
  WIN / PLACE / quinella backtests (Harville, EV-ratio, expanding calibration), LightGBM
  ablations and regularisation sweeps, calibration checks, data-quality diagnostics. Kept for
  provenance outside the lint/type/test gate; see [`research/README.md`](research/README.md).
- **Repo hygiene for publishing**: research dirs excluded from ruff, secrets moved to env vars,
  gitignore for dated outputs, this README section.

See [`PLAN.md`](PLAN.md) for the build plan (critique, phased roadmap, data scope, schema,
scraper strategy) and [`CLAUDE.md`](CLAUDE.md) for working conventions and current state.

## Status

| Milestone | Scope | State |
|---|---|---|
| **M0** | Foundations: env, config, logging, storage layout, tooling, CLI | ✅ done |
| **M1** | Incremental scraper + storage | ✅ done — backfill stored (1,697 meetings, 2006–2026) |
| **M2** | Features + baseline conditional-logit + honest backtest | ✅ done |
| **M3** | Model zoo (GBMs + LambdaMART + tabular NNs) + calibration + market blend | ✅ done |
| **M4** | NLP track (English): comments-on-running -> lagged signals + ablation | ✅ done |
| **M5** | Risk / staking sweeps (Kelly variants, caps, rounding, multi-bankroll) | ✅ done |
| **M6** | UI: read-only FastAPI + React/Vite/TS dashboards | ✅ done |
| **M7** | Live ops: GraphQL odds logger + race-day prediction pipeline | ✅ done |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                 # create .venv and install pinned deps from uv.lock
uv run hkjc --help      # CLI entry point
uv run hkjc doctor      # resolved config, paths, locked data scope
uv run pytest           # test suite
uv run ruff check . && uv run ruff format .   # lint + format
uv run mypy             # strict type-check
uv run pre-commit install   # one-time: git hooks
```

## Data layer (M1)

Scrape HKJC + open data into a local DuckDB + partitioned-Parquet lake. Every fetch is
recorded in a `_scrape_manifest`, so re-runs are **idempotent** (frozen past pages fetch
zero rows). Politeness: concurrency-capped, rate-limited, retried, cached.

```bash
# Historical results backfill (fixtures calendar enumerates meetings back to ~2006).
uv run hkjc backfill --since 2024-09-01      # bound it; omit --since for full history (~2006)
uv run hkjc scrape --date 2026-06-03         # a single meeting

# Profiles + alternative sources (run after results so ids/dates are known).
uv run hkjc scrape-horses                    # horse bio + form records
uv run hkjc scrape-people                    # jockey + trainer season stats
uv run hkjc scrape-weather --since-year 2006 # HKO daily-climate temperatures
uv run hkjc scrape-trials                    # barrier trials
uv run hkjc scrape-trackwork                 # trackwork (gallop) records
uv run hkjc scrape-sectionals                # per-200m sectional times (#7)
uv run hkjc scrape-text                      # comments-on-running per race (#9, NLP)
uv run hkjc scrape-holidays                  # HK public holidays

uv run hkjc data-health                      # coverage report (meetings/races/rows by season)
```

**What's collected:** results (finish order, SP/win-odds, running positions, finish times,
full WIN→QUARTET dividends), going + rail position, horse profiles (locked bio block +
per-run form), jockey/trainer season stats, HKO daily temperatures, barrier trials,
trackwork, public holidays. See `PLAN.md` §0 for the locked scope (alt sources
3,4,5,7,9,11,14; #6 vet-list and #8 gear-change declarations were dropped as forward-only).
Per-200m **sectionals (#7)** are captured via `hkjc scrape-sectionals` (run it to backfill).

> A full ~20-season backfill is a multi-hour crawl (~1,600 meetings × ~11 pages). It's
> idempotent and cached, so bound the first run with `--since` or run it overnight.

## Features + backtest (M2)

Build the as-of feature store, then run an honest, time-ordered walk-forward backtest of the
PL-strength conditional-logit baseline (WIN softmax + Harville PLACE).

```bash
uv run hkjc features build    # -> features_runner (one as-of row per runner; ~197k rows)
uv run hkjc backtest          # walk-forward, two ROIs, calibration PNG, leakage canary
```

Every feature is computable from `event_time ≤ race_off_time`; the closing-line SP is walled
off from the model and a deterministic **leakage canary** must score ~0. The backtest reports
**two ROIs** (PLAN §1A): a conservative *model-only* (selections from model probability alone)
and an optimistic *market-blended* (positive-EV bets at the SP line). Payouts use the stored
final dividends. The baseline's honest model-only WIN ROI sits around the ~17.5% takeout —
i.e. **no edge beyond the market**, the expected starting point (PLAN §1F); beating it is the
job of M3+.

## Model zoo + leaderboard (M3)

Train the whole zoo (LightGBM, XGBoost, CatBoost, LambdaMART, MLP, FT-Transformer, ensemble)
behind one `ProbabilityModel` interface and rank them on the same honest walk-forward.

```bash
uv run hkjc features build    # rebuild as features v1 (adds the debut-age heuristic)
uv run hkjc train             # walk-forward leaderboard (log-loss / top-1 / ECE / two ROIs)
uv run hkjc train --models catboost,logit --seasons 5   # quick subset over recent seasons
uv run hkjc tune --model catboost --trials 15           # Optuna HPO (min walk-forward log-loss)
```

GBMs use the GPU when present (`HKJC_FORCE_CPU=1` forces CPU; CI is CPU-only); runs are logged
to a local MLflow sqlite store with the feature-store data hash for reproducibility. The
honest takeaway holds across the zoo: **every model loses ≈ the takeout — no edge beyond the
market yet.** A clean finding is that models trained on the *grouped within-race* likelihood
(logit, the NNs) are much better calibrated than the pointwise GBMs, which is what the
calibration layer (temperature/isotonic/Platt) is for.

> Heavy jobs: run them via the venv directly (`.venv/bin/hkjc`, or `.venv/Scripts/hkjc.exe`
> on Windows) — `uv run` re-syncs the env on each call and can contend with a running job.

## NLP track (M4)

Scrape English race text and fold it in as a **lagged** feature group (a comment describes a
run, so it is only a feature for the horse's *later* races — PLAN §1C), then ablate it.

```bash
uv run hkjc scrape-text       # comments-on-running per race (corunning, #9)
uv run hkjc features nlp      # encode comments -> lexicon flags + MiniLM anchor-similarities
uv run hkjc features build    # rebuild as v2 (joins + lags the nlp_text group)
uv run hkjc ablate            # walk-forward logit with vs without the NLP group (marginal effect)
```

Signals = spaCy rules/lexicon (interpretable trouble/ran-on/easing/… counts) + MiniLM sentence
embeddings reduced to a few interpretable anchor-similarity scores. The group is **ablatable**
and kept out of the baseline. With a full text backfill the ablation quantifies NLP's marginal
ROI/log-loss; on a small pilot it is ~0 (low lagged coverage), as expected.

## Risk / staking sweeps (M5)

Reuse the walk-forward OOS predictions and sweep **staking methods × bankrolls** to size value
bets honestly: flat, fixed-fraction, full and fractional Kelly (incl. the exact within-race
**correlated/simultaneous** Kelly), under per-race 10% / per-day 25% caps and legal HK$10
rounding, at HK$1k/10k/50k/100k.

```bash
uv run hkjc risk sweep                       # full sweep -> comparison table + CSV/Parquet + ROI PNG
uv run hkjc risk sweep --pools win           # WIN only (default win,place)
uv run hkjc risk sweep --rebate-rate 0.1     # assume a 10% losing-turnover rebate above HK$10k
```

The honest takeaway holds here too: **no staking rule manufactures an edge** — every method
loses ≈ the takeout (best is fractional Kelly λ≈0.05–0.10 at ~−15%; flat/fixed −22% to −30%),
with wide, overlapping CIs. What staking *does* change is structural, and the sweep surfaces the
two headline effects: the **HK$10 granularity** loss (at HK$1,000 flat/fixed can place *no*
legal diversified bets; Kelly loses ~98% of intended stake to rounding, falling to ~23% at
HK$100k) and the **HK$10k rebate threshold** (crossed on 0 days at HK$1k–10k, but 16–17 days at
HK$100k). Pari-mutuel pool dilution is negligible (a HK$10k cap is <0.2% of HKJC's pools) and is
not modelled; HKJC's real rebate schedule is parameterised, not fabricated. Outputs land in
`data/processed/risk/`.

## Dashboards (M6)

A local, **read-only** FastAPI backend + a React/Vite/TS dashboard suite — it surfaces
recommendations and **never places a bet** (there is no write/bet endpoint).

```bash
uv run hkjc serve              # FastAPI backend on http://127.0.0.1:8000 (/api/*)

# In a second terminal (needs Node.js LTS — not a Python dep; install once):
cd ui && npm install           # first run only
npm run dev                    # Vite dev server on http://localhost:5173 (proxies /api -> :8000)
```

Four dashboards, all on real M2–M5 output (race-day on a mocked card, flagged **MOCK** until
the M7 live logger lands): **Data Health** (coverage + meetings/season + recent races),
**Backtest Explorer** (policy ROIs + WIN calibration curve + the M5 staking sweep),
**Experiment Compare** (the model-zoo leaderboard + ROI-vs-takeout bars), and **Race Day**
(value/stake recommendations). The backend reads the DuckDB views live and the persisted
`processed/` snapshots (`run_backtest`/`run_leaderboard`/`run_sweep` write them) — no training
happens in a request. The frontend (`ui/`) builds with `npm run build` (`tsc` + `vite`); the
Python CI stays Python-only.

## Live ops + race day (M7)

The race-day system: a live-odds GraphQL logger, a forward card capture, and a pipeline that
turns the card + live odds into staking *recommendations*. **It logs and recommends only — there
is no bet/submit path.**

```bash
uv run hkjc train-production --model logit   # fit a model on all history, persist for inference
uv run hkjc log-odds --date 2026-06-21 --venue ST --rounds 120 --interval 30   # log odds snapshots
uv run hkjc race-day --date 2026-06-21 --venue ST   # card -> predict -> blend -> value -> Kelly
```

The live WIN/PLACE odds come from HKJC's public GraphQL gateway via the **exact whitelisted
queries** the bet.hkjc.com app uses (an arbitrary query is rejected). `race-day` fetches the
card, builds as-of features for its runners, predicts WIN/PLACE with the persisted model, blends
the live odds, flags positive-EV value, and sizes fractional-Kelly stakes — writing a
recommendation card to `data/processed/raceday/` that the **Race Day dashboard** renders. Snapshots
are deduplicated on `lastUpdateTime` into the `live_odds_snapshots` view.
`scripts/register_raceday_task.ps1` registers a Windows Task Scheduler job to run it at a cutoff.

## Configuration

YAML under [`config/`](config/), loaded and validated via `pydantic-settings`
(`src/hkjc/common/config.py`). `HKJC_`-prefixed env vars override YAML (nested via `__`,
e.g. `HKJC_RISK__BANKROLL=5000`). `config/local.yaml` is a gitignored override layer.

## Layout

```
config/            # YAML: paths, sources, features, risk, backtest, models
src/hkjc/
  common/          # config, logging, keys, time (HKT)
  data/            # scrape · parse · store (DuckDB+Parquet) · weather · holidays · live/ (M7 GraphQL)
  features/        # as-of feature store · canary · design matrix (M2/M3) · nlp/ lagged text (M4)
  models/          # ProbabilityModel · logit · place (M2) · gbm · nn · ensemble · calibrate · blend (M3)
  backtest/        # walk-forward engine · pari-mutuel sim · metrics · bootstrap · dataset (M2/M3)
  experiments/     # leaderboard · MLflow tracking · Optuna tuning · NLP ablation (M3/M4)
  risk/            # kelly · staking · rebate · simulate · sweep · report (M5)
  api/             # FastAPI: app · routes · schemas · service (M6, read-only)
  cli.py           # Typer entry point (`hkjc`)
ui/                # React + Vite + TS dashboards (M6; node_modules/dist gitignored)
tests/             # pytest suite
fixtures/          # checked-in HTML/JSON for offline parser tests
research/          # exploratory one-off scripts (outside the quality gate; see research/README.md)
reports/           # standalone studies (e.g. the 13-factor w456.py) + their outputs
data/              # gitignored data lake: raw/ processed/ cache/ live_odds/ mlruns/
```

## Responsible use

This repository exists to study how efficient a pari-mutuel market is and how to evaluate
predictive models without fooling yourself. It never places bets, holds no credentials, and
its own results show no exploitable edge. Nothing here is betting or financial advice.
