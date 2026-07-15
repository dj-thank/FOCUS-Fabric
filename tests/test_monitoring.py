from focus_fabric.monitoring import DriftSentinel


def test_drift_sentinel_triggers_on_sustained_miscoverage() -> None:
    sentinel = DriftSentinel(
        target_miscoverage=0.05,
        confidence=0.90,
        window=200,
        minimum_audits=50,
    )
    for index in range(200):
        sentinel.observe(true_error=1.0 if index % 2 == 0 else 0.0, certificate_upper=0.1)
    assert sentinel.triggered
    assert sentinel.action() == "strict_fallback_and_recompile"


def test_drift_sentinel_does_not_trigger_when_covered() -> None:
    sentinel = DriftSentinel(
        target_miscoverage=0.10,
        confidence=0.90,
        window=200,
        minimum_audits=50,
    )
    for _ in range(200):
        sentinel.observe(true_error=0.05, certificate_upper=0.1)
    assert not sentinel.triggered
