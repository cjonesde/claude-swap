"""Tests for the unified-header usage source."""

from __future__ import annotations

from claude_swap.unified_headers import parse_unified_headers

HEALTHY = {
    "anthropic-ratelimit-unified-status": "allowed",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-5h-reset": "1786140000",
    "anthropic-ratelimit-unified-5h-utilization": "0.11",
    "anthropic-ratelimit-unified-7d-status": "allowed",
    "anthropic-ratelimit-unified-7d-reset": "1786640400",
    "anthropic-ratelimit-unified-7d-utilization": "0.1",
    "anthropic-ratelimit-unified-representative-claim": "five_hour",
}

EXHAUSTED = {
    "anthropic-ratelimit-unified-status": "rejected",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-5h-reset": "1786137000",
    "anthropic-ratelimit-unified-5h-utilization": "0.0",
    "anthropic-ratelimit-unified-7d-status": "rejected",
    "anthropic-ratelimit-unified-7d-reset": "1786204800",
    "anthropic-ratelimit-unified-7d-utilization": "1.0",
    "anthropic-ratelimit-unified-overage-status": "rejected",
    "anthropic-ratelimit-unified-overage-utilization": "1.06",
    "anthropic-ratelimit-unified-overage-disabled-reason": "org_spend_cap_reached",
    "anthropic-ratelimit-unified-representative-claim": "seven_day",
}


def test_healthy_scales_fractions_to_percentages():
    out = parse_unified_headers(HEALTHY)
    assert out["five_hour"]["pct"] == 11.0
    assert out["seven_day"]["pct"] == 10.0


def test_reset_epoch_becomes_display_fields():
    out = parse_unified_headers(HEALTHY)
    assert out["five_hour"]["resets_at"] == "2026-08-07T22:00:00Z"
    assert out["five_hour"]["countdown"]
    assert out["five_hour"]["clock"]


def test_exhausted_account_reports_100_not_an_error():
    out = parse_unified_headers(EXHAUSTED)
    assert out["seven_day"]["pct"] == 100.0
    assert out["five_hour"]["pct"] == 0.0


def test_absent_headers_return_none():
    assert parse_unified_headers({}) is None
    assert parse_unified_headers({"content-type": "application/json"}) is None


def test_header_lookup_is_case_insensitive():
    upper = {k.upper(): v for k, v in HEALTHY.items()}
    assert parse_unified_headers(upper)["five_hour"]["pct"] == 11.0


def test_partial_windows_are_allowed():
    only5h = {k: v for k, v in HEALTHY.items() if "-7d-" not in k}
    out = parse_unified_headers(only5h)
    assert out["five_hour"]["pct"] == 11.0
    assert "seven_day" not in out


def test_malformed_values_are_skipped_not_fatal():
    bad = dict(HEALTHY, **{"anthropic-ratelimit-unified-5h-utilization": "banana"})
    out = parse_unified_headers(bad)
    assert "five_hour" not in out
    assert out["seven_day"]["pct"] == 10.0


class TestOverageHeadersBecomeTheSpendEntry:
    """The pay-as-you-go axis issue #194's last-resort tier runs on. It reads
    ``oauth.spend_headroom``, which reads the ``spend`` entry, which only the
    endpoint's ``extra_usage`` object used to produce -- and issue #220 means
    a subscription-exhausted account never gets an endpoint answer, so the
    tier could not fire in the one scenario it exists for. These headers
    carry the data, captured live from such an account.
    """

    def test_overage_utilization_is_a_fraction_like_the_windows(self):
        out = parse_unified_headers(EXHAUSTED)
        assert out["spend"]["pct"] == 106.0

    def test_status_and_reason_travel_as_their_own_signals(self):
        """Independent of the number: they must arrive intact so
        ``spend_headroom`` can veto a low utilization an org still rejects."""
        out = parse_unified_headers(EXHAUSTED)
        assert out["spend"]["status"] == "rejected"
        assert out["spend"]["disabled_reason"] == "org_spend_cap_reached"

    def test_the_probe_reports_no_credit_amounts(self):
        """The headers carry no ``used_credits``/``monthly_limit``, so the
        entry has the utilization axis and no money line."""
        spend = parse_unified_headers(EXHAUSTED)["spend"]
        assert "used" not in spend
        assert "limit" not in spend

    def test_a_response_without_overage_headers_has_no_spend_entry(self):
        assert "spend" not in parse_unified_headers(HEALTHY)

    def test_a_malformed_overage_utilization_yields_no_spend_entry(self):
        """Unknown, never guessed: no entry reads as unknown spend, which
        callers must not promote to a rotation target."""
        bad = dict(
            EXHAUSTED,
            **{"anthropic-ratelimit-unified-overage-utilization": "banana"},
        )
        out = parse_unified_headers(bad)
        assert "spend" not in out
        assert out["seven_day"]["pct"] == 100.0

    def test_overage_headers_alone_still_report_the_spend_axis(self):
        """A response carrying the overage headers and no window headers is
        still worth a usage dict: the windows are unknown, the spend axis is
        measured."""
        overage_only = {
            k: v for k, v in EXHAUSTED.items() if "-overage-" in k
        }
        out = parse_unified_headers(overage_only)
        assert out == {
            "spend": {
                "pct": 106.0,
                "status": "rejected",
                "disabled_reason": "org_spend_cap_reached",
            }
        }
