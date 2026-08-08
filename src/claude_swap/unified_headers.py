"""Usage source that reads utilization from unified rate-limit response headers.

Upstream issue realiti4/claude-swap#220: ``GET /api/oauth/usage`` rate-limits
per account on that account's own inference activity, so the busiest accounts,
the ones most worth watching, answer 429 and never report. The same token
still gets an answer from ``POST /v1/messages``, and that response carries the
same utilization as ``anthropic-ratelimit-unified-*`` headers, on a 429 as well
as a 200.

Two things a caller must not get wrong:

- The headers are FRACTIONS (``0.11`` means 11%), while cswap's ``pct`` fields,
  and the shape ``oauth.build_usage_result`` consumes, are PERCENTAGES.
  ``parse_unified_headers`` scales before handing off, so its output renders
  through every existing display path.
- The probe spends the account's own quota: a real completion with
  ``max_tokens=1``, roughly 10 tokens including the prompt. Every caller must
  bound how often it fires, and ``usage.headerFallback`` can switch it off.

``PROBE_MODEL`` is pinned to a dated snapshot rather than an alias, so what the
probe costs and which limits it counts against cannot change under us. Dated
snapshots do get retired and nothing here would notice: the 404 reads as
"probe failed", the 429 falls back to its original error, and the #220 rescue
goes quiet with only a DEBUG line. Keeping it current is a maintenance
obligation. Any cheap model the account can call works; only the response
headers are read.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone

from claude_swap import oauth

_logger = logging.getLogger("claude-swap")

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
PROBE_MODEL = "claude-haiku-4-5-20251001"
MESSAGES_API_VERSION = "2023-06-01"

_WINDOWS = (("5h", "five_hour"), ("7d", "seven_day"))

_OVERAGE_PREFIX = "anthropic-ratelimit-unified-overage"


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive lookup by header name.

    HTTP header names are case-insensitive by spec, but a caller's mapping is
    not guaranteed to normalize case. A linear scan is fine: only a handful of
    fixed names are ever looked up.
    """
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def _reset_iso(raw: str) -> str | None:
    """Epoch-seconds string to an ISO 8601 string ending in ``Z``, or None."""
    try:
        epoch = int(raw)
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (OverflowError, OSError, ValueError):
        return None


def _utilization_pct(raw: str) -> float | None:
    """A utilization fraction as a percentage, or None when it is not usable.

    ``float()`` accepts ``"nan"``, ``"inf"`` and negatives, and each stored as
    a percentage reads downstream as a measurement: nan loses every comparison,
    an infinity poisons the reset math, a negative invents headroom. Rejected
    here so nothing past this module has to defend against them.
    """
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value * 100


def _overage_extra_usage(headers: Mapping[str, str]) -> dict | None:
    """The ``extra_usage``-shaped view of the overage headers, or None.

    ``anthropic-ratelimit-unified-overage-utilization`` is the pay-as-you-go
    spend axis, the one issue #194's last-resort rotation tier runs on. Since
    issue #220 makes the probe the only usage source for a subscription-
    exhausted account, and that is precisely the account #194 was written
    for, the tier could never fire until these headers were mapped: the
    endpoint's ``extra_usage`` object never reached a probed account.

    Captured live from an account whose subscription window was spent::

        anthropic-ratelimit-unified-overage-status: rejected
        anthropic-ratelimit-unified-overage-utilization: 1.06
        anthropic-ratelimit-unified-overage-disabled-reason: org_spend_cap_reached

    Utilization is a fraction like every other unified header, so ``1.06``
    is 106% and goes through the same ``_utilization_pct`` boundary the window
    headers do.
    ``status`` and ``disabled-reason`` are independent signals, not
    restatements of the number: an org can reject overage at any
    utilization, so both travel to ``oauth._spend_entry`` and are honored
    by ``oauth.spend_headroom`` on their own.

    ``is_enabled`` is True whenever these headers appeared at all, because
    that is what they prove: the account HAS an overage axis. Whether it can
    currently serve is what ``status``/``disabled_reason`` answer, and
    reporting them is the whole point of building the entry for a capped
    account instead of dropping it. The headers carry no credit amounts, so
    the entry has no money line -- see ``oauth._spend_entry`` for how that
    half became optional. None when no utilization header is present or it
    does not parse: no spend entry then, which callers read as unknown and
    therefore never as a rotation target.
    """
    raw_util = _header(headers, f"{_OVERAGE_PREFIX}-utilization")
    if raw_util is None:
        return None
    pct = _utilization_pct(raw_util)
    if pct is None:
        return None
    extra_usage: dict = {"is_enabled": True, "utilization": pct}
    status = _header(headers, f"{_OVERAGE_PREFIX}-status")
    if status:
        extra_usage["status"] = status
    reason = _header(headers, f"{_OVERAGE_PREFIX}-disabled-reason")
    if reason:
        extra_usage["disabled_reason"] = reason
    return extra_usage


def parse_unified_headers(headers: Mapping[str, str]) -> dict | None:
    """Normalize ``anthropic-ratelimit-unified-*`` headers into cswap's usage shape.

    Reads the ``5h``/``7d`` ``utilization`` and ``reset`` headers
    case-insensitively, scales utilization to a percentage (see the module
    docstring), then delegates the final shape to
    ``oauth.build_usage_result``. The pay-as-you-go overage headers ride along
    as an ``extra_usage`` payload (see :func:`_overage_extra_usage`), so a
    probed account carries the same ``spend`` entry the endpoint would have
    produced. None when nothing at all survives, matching that function's own
    empty-result contract.

    A window whose ``utilization`` header is missing or unusable is skipped
    rather than defaulted to 0, and the result is marked ``partial`` so
    ``oauth.account_headroom`` reports UNKNOWN instead of the surviving
    window's headroom: the number is still worth showing, but one window's
    headroom is not the account's. Verified live on an account at 5h 0% / 7d
    100%, where dropping the weekly header made it read as 100% headroom, the
    fleet's preferred rotation target. Unlike the usage endpoint, a header set
    can lose a window in transit, so incompleteness here is no evidence that
    the window does not exist.
    """
    endpoint_shaped: dict = {}
    for prefix, key in _WINDOWS:
        raw_util = _header(headers, f"anthropic-ratelimit-unified-{prefix}-utilization")
        if raw_util is None:
            continue
        pct = _utilization_pct(raw_util)
        if pct is None:
            continue
        window: dict = {"utilization": pct}
        raw_reset = _header(headers, f"anthropic-ratelimit-unified-{prefix}-reset")
        if raw_reset is not None:
            resets_at = _reset_iso(raw_reset)
            if resets_at is not None:
                window["resets_at"] = resets_at
        endpoint_shaped[key] = window
    overage = _overage_extra_usage(headers)
    if overage is not None:
        endpoint_shaped["extra_usage"] = overage
    if not endpoint_shaped:
        return None
    result = oauth.build_usage_result(endpoint_shaped)
    if result is None:
        return None
    if any(key not in endpoint_shaped for _, key in _WINDOWS):
        result["partial"] = True
    return result


def probe_usage(access_token: str, timeout_s: float = 30.0) -> dict | None:
    """Spend a 1-token completion to read live utilization off response headers.

    ``POST /v1/messages`` rate-limits far more loosely than ``GET
    /api/oauth/usage`` (issue #220), so it answers, with a 200 or a 429 that
    still carries the headers, on exactly the accounts the usage endpoint goes
    silent on. Sends the same bearer and beta headers as
    ``oauth.request_usage_data``, plus ``anthropic-version``, which
    ``/v1/messages`` 400s without and the usage endpoint never required.

    None on any transport failure, on a response with no unified headers, and
    on every error status but 429. Only a 429 reports the account's own
    rate-limit state; any other status describes the request or the server (a
    dead token, a retired model, an outage), so unified-looking headers on one
    are not evidence of a healthy account.
    """
    body = json.dumps(
        {
            "model": PROBE_MODEL,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "anthropic-beta": oauth.OAUTH_BETA_HEADER,
        "anthropic-version": MESSAGES_API_VERSION,
        "Content-Type": "application/json",
        "User-Agent": "claude-swap/1.0",
    }
    req = urllib.request.Request(
        MESSAGES_URL, data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return parse_unified_headers(resp.headers)
    except urllib.error.HTTPError as e:
        if e.code != 429 or not e.headers:
            return None
        return parse_unified_headers(e.headers)
    except Exception as e:
        _logger.debug("Unified-header probe failed: %r", e)
        return None
