from collections import Counter

from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask


def _scan_signature(scan) -> tuple:
    peaks = tuple(
        (
            round(peak.freq_mhz, 6),
            round(peak.power_dbm, 6),
            round(peak.width_mhz, 6),
            peak.source,
        )
        for peak in scan.detected_peaks
    )
    ground_truth = tuple(
        (
            state.channel_def.id,
            round(state.center_freq, 6),
            state.active_this_scan,
            round(state.power_dbm, 6),
        )
        for state in scan.ground_truth
    )
    interference = tuple(round(freq, 6) for freq in scan.interference_freqs)
    return peaks, ground_truth, interference


def _stage_signatures(task: BlindSpectrumMonitoringTask) -> list[list[tuple]]:
    signatures = [_scan_signature(scan) for scan in task._scans]
    if not task._schedule_stages:
        return [signatures]

    stage_slices: list[list[tuple]] = []
    offset = 0
    for stage in task._schedule_stages:
        count = int(stage["num_instances"])
        stage_slices.append(signatures[offset : offset + count])
        offset += count
    return stage_slices


def _build_task() -> BlindSpectrumMonitoringTask:
    return BlindSpectrumMonitoringTask(
        schedule="default",
        corpus_id="mixed_grid_lifecycle",
    )


def test_prepare_run_zero_preserves_canonical_order():
    canonical = _build_task()
    canonical.reset()

    rollout_zero = _build_task()
    rollout_zero.prepare_run(0)
    rollout_zero.reset()

    assert _stage_signatures(rollout_zero) == _stage_signatures(canonical)
    assert [scan.scan_idx for scan in rollout_zero._scans] == list(
        range(len(rollout_zero._scans))
    )


def test_prepare_run_shuffles_within_stage_without_replacement():
    canonical = _build_task()
    canonical.reset()
    canonical_stages = _stage_signatures(canonical)

    rollout_one = _build_task()
    rollout_one.prepare_run(1)
    rollout_one.reset()
    rollout_one_stages = _stage_signatures(rollout_one)

    assert len(rollout_one_stages) == len(canonical_stages)
    assert [scan.scan_idx for scan in rollout_one._scans] == list(
        range(len(rollout_one._scans))
    )

    stages_with_reordered_scans = 0
    for canonical_stage, rollout_stage in zip(canonical_stages, rollout_one_stages):
        assert len(canonical_stage) == len(rollout_stage)
        assert Counter(canonical_stage) == Counter(rollout_stage)
        if canonical_stage != rollout_stage:
            stages_with_reordered_scans += 1

    assert stages_with_reordered_scans > 0

    rollout_one_again = _build_task()
    rollout_one_again.prepare_run(1)
    rollout_one_again.reset()
    assert _stage_signatures(rollout_one_again) == rollout_one_stages
