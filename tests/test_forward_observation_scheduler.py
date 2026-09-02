from __future__ import annotations

from pathlib import Path


def test_scheduler_script_invokes_only_forward_pipeline() -> None:
    script = Path("scripts/run_daily_forward_observation.ps1").read_text(encoding="utf-8")
    assert "src.daily_forward_pipeline" in script
    assert "paper_trade_alpha" not in script
    assert "paper_fill_tracker" not in script
    assert "live" not in script.lower()
    assert "$pipelineExitCode = $LASTEXITCODE" in script
    assert "if ($pipelineExitCode -ne 0)" in script
