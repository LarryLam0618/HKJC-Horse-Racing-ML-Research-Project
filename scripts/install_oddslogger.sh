#!/bin/bash
# One-time install of the M7 odds-logger launchd job (macOS).
# Re-run any time to pick up script changes. Uninstall:
#   launchctl bootout gui/$(id -u)/com.hkjc.oddslogger
#   rm ~/Library/LaunchAgents/com.hkjc.oddslogger.plist
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.hkjc.oddslogger"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"

[ -x "$PROJECT_DIR/.venv/bin/hkjc" ] || { echo "run 'uv sync' first — no .venv/bin/hkjc"; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/logs"
sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$PROJECT_DIR/scripts/com.hkjc.oddslogger.plist" > "$DEST"
chmod +x "$PROJECT_DIR/scripts/log_odds_day.sh"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$DEST"
launchctl enable "gui/$(id -u)/$LABEL"

echo "installed: $DEST"
echo "fires ~11:30 and ~17:30 (Mac local time — set the Mac to Hong Kong time)."
echo "check:   launchctl print gui/$(id -u)/$LABEL | grep -E 'state|last exit'"
echo "run now: launchctl kickstart -k gui/$(id -u)/$LABEL   (then tail logs/odds_logger.log)"
