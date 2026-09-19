"""Is there an HKJC meeting *today* (HKT), and at which venue — with races still to run?

Prints ``ST`` or ``HV`` to stdout and exits 0 when today is a race day and the meeting
still has at least one non-finished race. Prints nothing and exits 1 otherwise.

Venue comes from the fixtures calendar (``(1)`` = Sha Tin, ``(2)`` = Happy Valley in the
distance tokens) — reliable, and independent of the GraphQL gateway's "fall back to the
live meeting" behaviour. The GraphQL card is then queried only to check the meeting is
not already over.

Used by ``scripts/log_odds_day.sh``. Read-only; logs nothing, bets nothing.
"""

from __future__ import annotations

import re
import sys

import httpx
from selectolax.parser import HTMLParser

from hkjc.common.config import get_config
from hkjc.common.time import now_hkt
from hkjc.data.live.graphql import HEADERS, LiveClient

_FINISHED_RACE = {"RESULT", "RESULTS", "ABANDONED", "CLOSED"}


def _fixture_venue(day) -> str | None:
    cfg = get_config()
    url = f"{cfg.sources.hkjc_base_url}/fixture?calyear={day.year}&calmonth={day.month:02d}"
    try:
        html = httpx.get(url, headers={"user-agent": HEADERS["user-agent"]}, timeout=20.0).text
    except httpx.HTTPError:
        return None
    for cell in HTMLParser(html).css("td.calendar"):
        span = cell.css_first("span")
        if span is None or span.text(strip=True) != str(day.day):
            continue
        text = cell.text(separator=" ", strip=True)
        codes = set(re.findall(r"\d{3,4}\((\d)\)", text))
        if "1" in codes:
            return "ST"
        if "2" in codes:
            return "HV"
        return None
    return None


def _has_unfinished_races(day, venue: str) -> bool:
    try:
        with LiveClient() as client:
            card = client.card(day, venue)
    except Exception:
        return True  # network hiccup -> don't suppress the run
    if card is None or card.status != "DEFINED":
        return False
    if not card.races:
        return True  # card not published in detail yet, but meeting is defined
    return any((r.status or "").upper() not in _FINISHED_RACE for r in card.races)


def main() -> int:
    day = now_hkt().date()
    venue = _fixture_venue(day)
    if venue is None:
        return 1
    if not _has_unfinished_races(day, venue):
        return 1
    print(venue)
    return 0


if __name__ == "__main__":
    sys.exit(main())
