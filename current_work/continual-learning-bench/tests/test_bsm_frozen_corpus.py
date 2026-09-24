from collections import Counter

from src.tasks.blind_spectrum_monitoring.corpus import (
    build_rollout_corpus,
    scan_from_row,
    write_scan_corpus,
)
from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask


def _scan_signature(scan) -> tuple:
    peaks = tuple(
        (
            peak.peak_id,
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
    return (
        scan.scan_id,
        scan.sensor_id,
        scan.detector_version,
        scan.integration_time_ms,
        round(scan.estimated_noise_floor_dbm, 6),
        peaks,
        ground_truth,
        interference,
    )


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


def _write_corpus(tmp_path) -> str:
    rows, metadata = build_rollout_corpus("default")
    jsonl_path = tmp_path / "mixed_grid_lifecycle.jsonl"
    metadata_path = tmp_path / "mixed_grid_lifecycle_metadata.json"
    write_scan_corpus(
        rows,
        metadata,
        jsonl_path=jsonl_path,
        metadata_path=metadata_path,
    )
    return str(jsonl_path)


def test_frozen_corpus_matches_seed_generated_rollout(tmp_path):
    dataset_path = _write_corpus(tmp_path)

    canonical_rows, _ = build_rollout_corpus("default")
    canonical_signatures = [
        _scan_signature(scan_from_row(row)) for row in canonical_rows
    ]

    frozen = BlindSpectrumMonitoringTask(
        schedule="default",
        dataset_path=dataset_path,
    )
    frozen.reset()
    frozen_signatures = [_scan_signature(scan) for scan in frozen._scans]

    assert frozen_signatures == canonical_signatures
    assert "corpus:default" in frozen._current_instance_id()


def test_schedule_declared_corpus_loads_without_explicit_corpus_id():
    """The ``default`` schedule declares the shipped corpus at stage level, so
    users can run it with ``--schedule default`` without also passing
    ``corpus_id``.
    """

    task = BlindSpectrumMonitoringTask(schedule="default")
    task.reset()

    assert task._resolved_corpus_id == "mixed_grid_lifecycle"
    assert task._resolved_dataset_path is not None
    assert task._resolved_dataset_path.name == "mixed_grid_lifecycle.jsonl"
    assert len(task._scans) == 90
    assert "corpus:mixed_grid_lifecycle" in task._current_instance_id()


def test_shipped_corpus_loads_via_corpus_id():
    """The shipped ``bsm_mixed_grid_*`` configs reference the
    ``mixed_grid_lifecycle`` corpus by id; loading via ``corpus_id`` should
    locate the on-disk corpus, populate ``_resolved_corpus_id``, and emit
    instance ids tagged with ``corpus:mixed_grid_lifecycle``.
    """

    task = BlindSpectrumMonitoringTask(
        schedule="default",
        corpus_id="mixed_grid_lifecycle",
    )
    task.reset()

    assert task._resolved_corpus_id == "mixed_grid_lifecycle"
    assert task._resolved_dataset_path is not None
    assert task._resolved_dataset_path.name == "mixed_grid_lifecycle.jsonl"
    assert len(task._scans) == 90
    assert "corpus:mixed_grid_lifecycle" in task._current_instance_id()


def test_frozen_corpus_prepare_run_preserves_stage_contents(tmp_path):
    dataset_path = _write_corpus(tmp_path)

    canonical = BlindSpectrumMonitoringTask(
        schedule="default",
        dataset_path=dataset_path,
    )
    canonical.reset()
    canonical_stages = _stage_signatures(canonical)

    rollout_one = BlindSpectrumMonitoringTask(
        schedule="default",
        dataset_path=dataset_path,
    )
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


def test_frozen_corpus_populates_scan_and_peak_ids(tmp_path):
    dataset_path = _write_corpus(tmp_path)

    task = BlindSpectrumMonitoringTask(
        schedule="default",
        dataset_path=dataset_path,
    )
    task.reset()

    scan_ids = [scan.scan_id for scan in task._scans]
    peak_ids = [peak.peak_id for scan in task._scans for peak in scan.detected_peaks]

    assert all(scan_ids)
    assert len(scan_ids) == len(set(scan_ids))
    assert peak_ids
    assert all(peak_ids)
    assert len(peak_ids) == len(set(peak_ids))
