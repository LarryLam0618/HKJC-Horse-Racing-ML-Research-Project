#!/bin/bash
# M7 live-odds logger — race-day driver (macOS).
#
# Detects today's HKJC meeting (fixtures calendar), then polls WIN/PLACE odds for the
# whole meeting window, deduplicated on lastUpdateTime. Append-only into the
# live_odds_snapshots DuckDB view — this is the forward data collection for the
# "odds resonance" signal (see reports/resonance_signal.md).
#
# Scheduled by scripts/com.hkjc.oddslogger.plist (fires ~11:30 and ~17:30 HKT daily).
# Runs nothing on non-race days. Logs only; never places a bet.
#
# Knobs (env): HKJC_ODDS_ROUNDS (default 220), HKJC_ODDS_INTERVAL secs (default 90).
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

HKJC="$PROJECT_DIR/.venv/bin/hkjc"
PY="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/odds_logger.log"

ts() { TZ=Asia/Hong_Kong date "+%Y-%m-%d %H:%M:%S %Z"; }
say() { echo "[$(ts)] $*" >>"$LOG"; }

# Only one logger at a time — the two daily triggers can otherwise overlap on a day meeting.
LOCK="$LOG_DIR/.oddslogger.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    say "another logger (pid $(cat "$LOCK")) is already running — exiting"
    exit 0
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

TODAY="$(TZ=Asia/Hong_Kong date +%F)"
HOUR="$(TZ=Asia/Hong_Kong date +%-H)"

VENUE="$("$PY" scripts/todays_meeting.py 2>>"$LOG")"
if [ -z "$VENUE" ]; then
    say "no meeting today ($TODAY) — nothing to do"
    exit 0
fi

# Happy Valley meetings are night fixtures (~19:15 first race). Skip the ~11:30 run for
# HV so we don't hammer a market that hasn't opened; the ~17:30 run will catch it.
if [ "$VENUE" = "HV" ] && [ "$HOUR" -lt 15 ]; then
    say "$TODAY HV is a night meeting — skipping the midday run, evening run will cover it"
    exit 0
fi

ROUNDS="${HKJC_ODDS_ROUNDS:-220}"
INTERVAL="${HKJC_ODDS_INTERVAL:-90}"

say "$TODAY $VENUE — starting logger (rounds=$ROUNDS interval=${INTERVAL}s, ~$((ROUNDS*INTERVAL/3600))h)"
# caffeinate -i: keep the Mac awake for the whole meeting window so scheduled polls don't
# stall on idle sleep (does nothing extra if the Mac is already kept awake / plugged in).
CAFFEINATE=""
command -v caffeinate >/dev/null 2>&1 && CAFFEINATE="caffeinate -i"
$CAFFEINATE "$HKJC" log-odds --date "$TODAY" --venue "$VENUE" \
    --rounds "$ROUNDS" --interval "$INTERVAL" >>"$LOG" 2>&1
RC=$?
say "$TODAY $VENUE — logger exited rc=$RC"
exit "$RC"
