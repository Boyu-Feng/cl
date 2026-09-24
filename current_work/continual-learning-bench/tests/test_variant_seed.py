"""Tests for seed handling in variant validation and rollout resampling."""

import unittest
from typing import Any

from src.tasks.variants import TaskVariantSpec, validate_variant_defaults
from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask
from src.runs import RunMode, derive_run_task_params


class VariantSeedValidationTests(unittest.TestCase):
    """validate_variant_defaults should always exempt seed."""

    def _make_spec(self, defaults: dict[str, Any]) -> TaskVariantSpec:
        return TaskVariantSpec(
            id="test",
            display_name="Test",
            description="test variant",
            defaults=defaults,
        )

    def test_seed_override_accepted(self):
        spec = self._make_spec({"seed": 42, "K": 7})
        validate_variant_defaults(spec, {"seed": 99, "K": 7})

    def test_non_seed_override_rejected(self):
        spec = self._make_spec({"seed": 42, "K": 7})
        with self.assertRaises(ValueError):
            validate_variant_defaults(spec, {"seed": 42, "K": 999})

    def test_seed_exempt_even_with_init_defaults(self):
        spec = self._make_spec({"seed": 42, "K": 7})
        validate_variant_defaults(
            spec,
            {"seed": 44, "K": 7},
            init_defaults={"seed": 42, "K": 5},
        )


class BSMVariantResampleTests(unittest.TestCase):
    """End-to-end: BSM task accepts resampled seeds with a variant."""

    def test_variant_uses_resampled_seed(self):
        task = BlindSpectrumMonitoringTask(variant="full_grid_active", seed=44)
        self.assertEqual(task.seed, 44)

    def test_variant_uses_default_seed_when_unmodified(self):
        task = BlindSpectrumMonitoringTask(variant="full_grid_active")
        self.assertEqual(task.seed, 42)

    def test_derive_then_construct(self):
        base_params = {"seed": 42, "variant": "full_grid_active"}
        derived = derive_run_task_params(base_params, RunMode.RESAMPLE, 3)
        task = BlindSpectrumMonitoringTask(
            **{
                k: v
                for k, v in derived.items()
                if k not in {"run_index", "rollout_index"}
            }
        )
        self.assertEqual(task.seed, 42)


class DeriveTaskParamsSeedTests(unittest.TestCase):
    """_derive_task_params interacts correctly with variant seeds."""

    def test_resample_preserves_seed_for_variant_params(self):
        params = {"seed": 42, "variant": "some_variant"}
        derived = derive_run_task_params(params, RunMode.RESAMPLE, 3)
        self.assertEqual(derived["seed"], 42)
        self.assertEqual(derived["variant"], "some_variant")
        self.assertEqual(derived["rollout_index"], 3)

    def test_replicate_preserves_seed_for_variant_params(self):
        params = {"seed": 42, "variant": "some_variant"}
        derived = derive_run_task_params(params, RunMode.REPLICATE, 3)
        self.assertEqual(derived["seed"], 42)


if __name__ == "__main__":
    unittest.main()
