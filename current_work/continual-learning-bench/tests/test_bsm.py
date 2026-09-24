"""Tests for the BSM task (no LLM or Docker calls)."""

import json
from pathlib import Path

import pytest

from src.tasks.blind_spectrum_monitoring.dgp import (
    ChannelDef,
    SpectrumDGP,
)
from src.tasks.blind_spectrum_monitoring.task import (
    BlindSpectrumMonitoringTask,
    ScanReport,
    Transmitter,
    _score_report,
)

BSM_DIR = Path("src/tasks/blind_spectrum_monitoring")


# ---------------------------------------------------------------------------
# DGP tests
# ---------------------------------------------------------------------------


class TestSpectrumDGP:
    def test_generate_produces_correct_count(self):
        dgp = SpectrumDGP(channels=[ChannelDef(id=1, slot=0)], n_instances=5, seed=0)
        scans = dgp.generate()
        assert len(scans) == 5

    def test_detected_peaks_present_for_active_channels(self):
        dgp = SpectrumDGP(
            channels=[ChannelDef(id=1, slot=0, power_dbm=-30)],
            n_instances=20,
            seed=42,
            p_active=1.0,
            p_miss=0.0,
            p_interference=0.0,
            p_false_alarm=0.0,
        )
        for scan in dgp.generate():
            channel_peaks = [p for p in scan.detected_peaks if p.source == "channel"]
            assert len(channel_peaks) == 1

    def test_no_peaks_for_inactive_channels(self):
        dgp = SpectrumDGP(
            channels=[ChannelDef(id=1, slot=0, power_dbm=-30)],
            n_instances=20,
            seed=42,
            p_active=0.0,
            p_interference=0.0,
            p_false_alarm=0.0,
        )
        for scan in dgp.generate():
            assert len(scan.detected_peaks) == 0

    def test_ground_truth_has_all_channels(self):
        channels = [ChannelDef(id=i, slot=i) for i in range(5)]
        dgp = SpectrumDGP(channels=channels, n_instances=1, seed=0)
        scan = dgp.generate()[0]
        assert len(scan.ground_truth) == 5
        ids = {cs.channel_def.id for cs in scan.ground_truth}
        assert ids == {0, 1, 2, 3, 4}

    def test_center_jitter_shifts_centers(self):
        ch = ChannelDef(id=1, slot=2, power_dbm=-30)
        dgp_no_jitter = SpectrumDGP(
            channels=[ch], n_instances=10, seed=42, center_jitter=0.0, W=10, G=2
        )
        dgp_jitter = SpectrumDGP(
            channels=[ch], n_instances=10, seed=42, center_jitter=5.0, W=10, G=2
        )
        nominal = ch.center_freq(10.0, 2.0)
        no_jitter_centers = [
            s.ground_truth[0].center_freq for s in dgp_no_jitter.generate()
        ]
        jitter_centers = [s.ground_truth[0].center_freq for s in dgp_jitter.generate()]
        assert all(c == pytest.approx(nominal) for c in no_jitter_centers)
        assert not all(c == pytest.approx(nominal, abs=0.01) for c in jitter_centers)

    def test_deterministic_with_same_seed(self):
        kwargs = dict(channels=[ChannelDef(id=1, slot=0)], n_instances=3, seed=99)
        a = SpectrumDGP(**kwargs).generate()
        b = SpectrumDGP(**kwargs).generate()
        for sa, sb in zip(a, b):
            assert len(sa.detected_peaks) == len(sb.detected_peaks)
            for pa, pb in zip(sa.detected_peaks, sb.detected_peaks):
                assert pa.freq_mhz == pb.freq_mhz
                assert pa.power_dbm == pb.power_dbm

    def test_interference_frequencies_in_band(self):
        dgp = SpectrumDGP(
            channels=[ChannelDef(id=1, slot=0)],
            n_instances=50,
            seed=0,
            p_interference=0.8,
            band_width=100.0,
        )
        for scan in dgp.generate():
            for f in scan.interference_freqs:
                assert 0 <= f <= 100.0

    def test_peaks_sorted_by_frequency(self):
        dgp = SpectrumDGP(
            channels=[ChannelDef(id=i, slot=i) for i in range(5)],
            n_instances=10,
            seed=42,
            p_interference=0.5,
        )
        for scan in dgp.generate():
            freqs = [p.freq_mhz for p in scan.detected_peaks]
            assert freqs == sorted(freqs)

    def test_miss_probability(self):
        dgp = SpectrumDGP(
            channels=[ChannelDef(id=1, slot=0, power_dbm=-30)],
            n_instances=200,
            seed=42,
            p_active=1.0,
            p_miss=0.5,
            p_interference=0.0,
            p_false_alarm=0.0,
        )
        n_detected = sum(
            1
            for scan in dgp.generate()
            if any(p.source == "channel" for p in scan.detected_peaks)
        )
        assert 60 < n_detected < 140  # ~100 expected

    def test_false_alarm_peaks(self):
        dgp = SpectrumDGP(
            channels=[],
            n_instances=200,
            seed=42,
            p_interference=0.0,
            p_false_alarm=1.0,
        )
        n_fa = sum(len(scan.detected_peaks) for scan in dgp.generate())
        assert n_fa == 200  # exactly one false alarm per scan


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


def _tx(
    center: float,
    bw: float = 10.0,
    active: bool = True,
    power: float = -30.0,
) -> Transmitter:
    return Transmitter(
        center_freq=center,
        bandwidth=bw,
        currently_active=active,
        estimated_power=power,
    )


class TestScoreReport:
    """Tests for long-run availability IoU scoring from transmitter reports."""

    W = 10.0
    G = 2.0
    BAND_WIDTH = 40.0

    def _score(self, chs, report, W=None, G=None, band_width=None):
        W = W or self.W
        G = G or self.G
        band_width = band_width or self.BAND_WIDTH
        return _score_report(
            report=report,
            all_latent_channel_defs=chs,
            W=W,
            G=G,
            band_width=band_width,
        )

    def test_perfect_report(self):
        """Exact occupied-map report should imply perfect availability IoU."""
        chs = [ChannelDef(id=1, slot=0), ChannelDef(id=2, slot=1)]
        report = ScanReport(
            transmitters=[
                _tx(chs[0].center_freq(self.W, self.G)),
                _tx(chs[1].center_freq(self.W, self.G)),
            ]
        )
        result = self._score(chs, report)
        assert result["score"] == 1.0
        assert result["n_matched"] == 2
        assert result["n_gt"] == 2

    def test_empty_report_overclaims_availability(self):
        chs = [ChannelDef(id=1, slot=0)]
        result = self._score(chs, ScanReport(transmitters=[]))
        assert result["score"] == pytest.approx(0.75, abs=0.0001)

    def test_missing_transmitter_penalizes_score(self):
        chs = [ChannelDef(id=1, slot=0), ChannelDef(id=2, slot=1)]
        report = ScanReport(transmitters=[_tx(chs[0].center_freq(self.W, self.G))])
        result = self._score(chs, report)
        assert result["score"] == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_center_offset_reduces_iou(self):
        chs = [ChannelDef(id=1, slot=0)]
        gt_center = chs[0].center_freq(self.W, self.G)
        report = ScanReport(transmitters=[_tx(gt_center + 5.0)])
        result = self._score(chs, report)
        assert 0.0 < result["score"] < 1.0

    def test_bandwidth_mismatch_reduces_iou(self):
        chs = [ChannelDef(id=1, slot=0)]
        gt_center = chs[0].center_freq(self.W, self.G)
        report = ScanReport(transmitters=[_tx(gt_center, bw=5.0)])
        result = self._score(chs, report)
        assert 0.0 < result["score"] < 1.0

    def test_extra_false_transmitter_penalizes_score(self):
        chs = [ChannelDef(id=1, slot=0)]
        gt_center = chs[0].center_freq(self.W, self.G)
        report = ScanReport(
            transmitters=[
                _tx(gt_center),
                _tx(20.0, bw=4.0),
            ]
        )
        result = self._score(chs, report)
        assert 0.0 < result["score"] < 1.0

    def test_activity_flag_does_not_affect_score(self):
        chs = [ChannelDef(id=1, slot=0)]
        gt_center = chs[0].center_freq(self.W, self.G)
        r_active = ScanReport(transmitters=[_tx(gt_center, active=True)])
        r_inactive = ScanReport(transmitters=[_tx(gt_center, active=False)])
        assert (
            self._score(chs, r_active)["score"] == self._score(chs, r_inactive)["score"]
        )

    def test_heterogeneous_bandwidths(self):
        """Exact occupied reports with mixed bandwidths should still score 1.0."""
        chs = [
            ChannelDef(id=1, slot=0),
            ChannelDef(id=6, slot=1, bandwidth_override=5.0),
        ]
        report = ScanReport(
            transmitters=[
                _tx(chs[0].center_freq(self.W, self.G), bw=10.0),
                _tx(chs[1].center_freq(self.W, self.G), bw=5.0),
            ]
        )
        result = self._score(chs, report)
        assert result["score"] == 1.0


# ---------------------------------------------------------------------------
# Variant and rollout loading
# ---------------------------------------------------------------------------


class TestVariantLoading:
    def test_all_variants_are_valid_json(self):
        variant_dir = BSM_DIR / "variants"
        for p in variant_dir.glob("*.json"):
            data = json.loads(p.read_text())
            assert "id" in data
            assert "defaults" in data
            assert "channels" in data["defaults"]
            assert data["id"] == p.stem

    def test_variants_share_difficulty_params(self):
        variant_dir = BSM_DIR / "variants"
        for p in variant_dir.glob("*.json"):
            data = json.loads(p.read_text())
            d = data["defaults"]
            assert d["channel_power_dbm"] == -40, f"{p.name}: wrong channel_power_dbm"
            assert d["noise_sigma"] == 4, f"{p.name}: wrong noise_sigma"
            assert d["p_interference"] == 0.2, f"{p.name}: wrong p_interference"
            assert d["center_jitter"] == 0.5, f"{p.name}: wrong center_jitter"


class TestRolloutLoading:
    def test_all_rollouts_are_valid_json(self):
        rollout_dir = BSM_DIR / "schedules"
        for p in rollout_dir.glob("*.json"):
            data = json.loads(p.read_text())
            assert "id" in data
            assert "stages" in data
            assert len(data["stages"]) >= 1
            assert data["id"] == p.stem

    def test_rollout_stages_have_explicit_seeds(self):
        rollout_dir = BSM_DIR / "schedules"
        for p in rollout_dir.glob("*.json"):
            data = json.loads(p.read_text())
            for i, stage in enumerate(data["stages"]):
                assert "seed" in stage.get("schedule", {}), (
                    f"Rollout {p.name} stage {i} must declare an explicit seed"
                )

    def test_rollout_variants_exist(self):
        variant_dir = BSM_DIR / "variants"
        variant_ids = {p.stem for p in variant_dir.glob("*.json")}
        rollout_dir = BSM_DIR / "schedules"
        for p in rollout_dir.glob("*.json"):
            data = json.loads(p.read_text())
            for stage in data["stages"]:
                assert stage["variant"] in variant_ids, (
                    f"Rollout {p.name} references missing variant '{stage['variant']}'"
                )


# ---------------------------------------------------------------------------
# Task integration (no LLM)
# ---------------------------------------------------------------------------


class TestTaskInit:
    def test_scan_report_requires_transmitters_field(self):
        with pytest.raises(Exception):
            ScanReport.model_validate({})

    def test_variant_init(self):
        task = BlindSpectrumMonitoringTask(variant="five_ch_wide")
        assert task.W == 15.0
        assert task.G == 9.0
        assert len(task._channel_defs) == 13
        assert task.num_instances == 20

    def test_rollout_init(self):
        task = BlindSpectrumMonitoringTask(
            schedule="default",
            corpus_id="mixed_grid_lifecycle",
        )
        assert task.num_instances == 90
        assert len(task._schedule_stages) == 3

    def test_rollout_sets_total_instances(self):
        task = BlindSpectrumMonitoringTask(
            schedule="default",
            corpus_id="mixed_grid_lifecycle",
        )
        assert task.num_instances == 90  # 30 + 30 + 30

    def test_scan_generation(self):
        task = BlindSpectrumMonitoringTask(variant="five_ch_wide")
        task._generate_scans()
        assert len(task._scans) == 20
        for scan in task._scans:
            assert len(scan.ground_truth) == 13
            assert isinstance(scan.detected_peaks, list)

    def test_scan_query_contains_peaks(self):
        task = BlindSpectrumMonitoringTask(variant="five_ch_wide")
        query = task.reset()
        assert "Detected peaks:" in query.prompt
        assert "scan_id:" in query.prompt
        assert "timestamp_utc:" in query.prompt
        assert "sensor_id:" in query.prompt
        assert "peak_id:" in query.prompt
        assert "MHz" in query.prompt
        assert "dBm" in query.prompt
        assert query.response_schema is ScanReport

    def test_scan_query_no_channel_hints(self):
        """Fully latent: prompt should not reveal channel count or true IDs."""
        task = BlindSpectrumMonitoringTask(variant="five_ch_wide")
        query = task.reset()
        assert "previously known" not in query.prompt
        assert "channel_ids" not in query.prompt
        assert "source:" not in query.prompt

    def test_feedback_is_minimal(self):
        """Feedback should be a simple acknowledgment, no per-channel status."""
        task = BlindSpectrumMonitoringTask(variant="five_ch_wide")
        task.reset()
        scan = task._scans[0]
        fb = task._render_feedback(scan)
        assert "recorded" in fb.lower()
        assert "ACTIVE" not in fb
        assert "INACTIVE" not in fb

    def test_no_docker_imports(self):
        """Verify that the task module has no Docker dependencies."""
        import src.tasks.blind_spectrum_monitoring.task as mod

        source = Path(mod.__file__).read_text()
        assert "DockerEnvironment" not in source
        assert "minisweagent" not in source
        assert "tempfile" not in source
