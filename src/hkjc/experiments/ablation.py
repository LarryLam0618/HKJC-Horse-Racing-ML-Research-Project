"""Feature-group ablation (PLAN.md §2 M4 exit criterion, reused post-M7).

Runs the *same* model through the *same* honest walk-forward twice -- without and with one
ablatable feature group -- and reports the marginal change in log-loss / top-1 / ROI. That
delta is M4's deliverable for ``nlp`` (does the comment-on-running signal help?) and the
acceptance test for ``residual``: the pace / weight-dynamics / class-deploy block ported from
the 13-factor study (reports/w456.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from hkjc.backtest.dataset import load_model_data
from hkjc.backtest.engine import BacktestResult
from hkjc.common.config import AppConfig, get_config
from hkjc.experiments.runner import evaluate_model
from hkjc.models.logit import ConditionalLogit

GROUPS: tuple[str, ...] = ("nlp", "residual")


@dataclass(frozen=True, slots=True)
class AblationResult:
    baseline: BacktestResult
    with_group: BacktestResult
    group: str = "nlp"


def run_ablation(
    cfg: AppConfig | None = None,
    *,
    group: str = "nlp",
    market_weight: float | None = None,
    ev_threshold: float | None = None,
    max_test_seasons: int | None = None,
    seed: int = 0,
) -> AblationResult:
    """Walk-forward the conditional logit with and without ``group``; return both results.

    ``group`` is ``"nlp"`` (the lagged comment signals, M4) or ``"residual"`` (the pace /
    weight-dynamics / class-deploy block, the 13-factor study).
    """
    if group not in GROUPS:
        msg = f"unknown ablation group {group!r}; expected one of {list(GROUPS)}"
        raise ValueError(msg)
    cfg = cfg or get_config()
    market_weight = cfg.models.market_blend_weight if market_weight is None else market_weight
    ev_threshold = cfg.risk.ev_threshold if ev_threshold is None else ev_threshold

    results: list[BacktestResult] = []
    for include in (False, True):
        data = load_model_data(
            cfg,
            include_nlp=include and group == "nlp",
            include_residual=include and group == "residual",
        )
        n_seasons = len(set(data.season.tolist()))
        min_train = 1 if max_test_seasons is None else max(1, n_seasons - max_test_seasons)
        results.append(
            evaluate_model(
                ConditionalLogit,
                data.numeric(),
                data,
                market_weight=market_weight,
                ev_threshold=ev_threshold,
                stake=cfg.risk.min_bet,
                min_train_seasons=min_train,
                seed=seed,
                cfg=cfg,
            )
        )
    return AblationResult(baseline=results[0], with_group=results[1], group=group)


def format_ablation(result: AblationResult) -> str:
    """ASCII table: baseline vs +group on log-loss / top-1 / model-only WIN ROI, with deltas."""
    b, n = result.baseline, result.with_group
    rows = [
        ("log-loss", b.win_log_loss, n.win_log_loss, "{:+.4f}"),
        ("top-1 hit", b.top1_hit_rate, n.top1_hit_rate, "{:+.4f}"),
        ("model-only WIN ROI", b.policies["model_win"].roi, n.policies["model_win"].roi, "{:+.2%}"),
        (
            "market-blend WIN ROI",
            b.policies["blend_win"].roi,
            n.policies["blend_win"].roi,
            "{:+.2%}",
        ),
    ]
    label = f"+{result.group}"
    out = [
        f"{result.group} ablation (logit, {b.n_oos_races} OOS races, seasons "
        f"{b.test_span[0]}..{b.test_span[1]}):\n",
        f"  {'metric':<22}{'baseline':>12}{label:>12}{'delta':>12}",
        "  " + "-" * 58,
    ]
    for name, base_v, grp_v, fmt in rows:
        is_pct = fmt.endswith("%}")
        bv = f"{base_v:.2%}" if is_pct else f"{base_v:.4f}"
        gv = f"{grp_v:.2%}" if is_pct else f"{grp_v:.4f}"
        out.append(f"  {name:<22}{bv:>12}{gv:>12}{fmt.format(grp_v - base_v):>12}")
    return "\n".join(out)
