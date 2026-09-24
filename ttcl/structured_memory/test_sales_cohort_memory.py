"""Tests use invented public interactions, never benchmark truth fixtures."""

import json
import unittest

from ttcl.structured_memory.cohort_memory import CohortMemory
from ttcl.structured_memory.sales_memory import SalesMemory


class SalesMemoryTests(unittest.TestCase):
    def test_public_catalog_joins_observations_and_settled_feedback(self):
        memory = SalesMemory()
        memory.observe(
            "",
            {"command": "cat data/furniture.json data/locations.json"},
            '[{"product_id":1,"product_name":"Chair","category":"Seating","price":20}]\n[{"store_id":7,"city":"North"}]',
            instance_id="a",
            instance_complete=False,
        )
        memory.observe(
            "- `data/sales.csv` — columns: product_id, store_id, sale_year, quantity",
            {"command": "cat data/sales.csv"},
            "product_id,store_id,sale_year,quantity\n1,7,2026,100\n",
            instance_id="a",
            instance_complete=True,
        )
        # Public elapsed-year feedback is in the next prompt, not an outcome.
        feedback = {
            "feedback_year": 2027,
            "entries": [
                {
                    "locality": "North",
                    "furniture_name": "Chair",
                    "predicted": 110,
                    "actual": 120,
                    "error": 10,
                }
            ],
        }
        query = "Previous feedback\n```json\n" + json.dumps(feedback) + "\n```"
        for _ in range(2):
            memory.observe(query, {}, "", instance_id="b", instance_complete=False)
        state = memory.state_dict()
        self.assertEqual(len(state["annual_panel"]), 2)
        self.assertEqual(state["annual_panel"][0]["product"], "Chair")
        self.assertAlmostEqual(
            state["observed_trends"][0]["endpoint_annual_growth"], 0.2
        )
        self.assertEqual(state["settled_error_by_locality"][0]["n_settled"], 1)
        self.assertEqual(
            state["settled_error_by_locality"][0]["mean_signed_error"], -10
        )

    def test_submitted_forecasts_not_actuals_and_unknown_ids_not_merged(self):
        memory = SalesMemory()
        memory.observe(
            "",
            {
                "predictions": [
                    {
                        "locality": "North",
                        "furniture_name": "Chair",
                        "year": 2030,
                        "items_sold": 99999,
                    }
                ]
            },
            "Forecast submitted",
            instance_id="a",
            instance_complete=True,
        )
        self.assertEqual(memory.state_dict()["annual_panel"], [])
        memory.observe(
            "",
            {},
            '[{"product_id":1,"store_id":7,"year":2026,"quantity":2},{"item_code":1,"store_id":7,"year":2026,"quantity":8}]',
            instance_id="b",
            instance_complete=True,
        )
        rows = memory.state_dict()["annual_panel"]
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["product"], rows[1]["product"])

    def test_public_unique_id_overlap_can_link_different_export_headers(self):
        memory = SalesMemory()
        memory.observe(
            "",
            {},
            '[{"furniture_id":"P1","furniture_name":"Chair"},{"location_id":"L1","locality":"North"}]',
            instance_id="a",
            instance_complete=False,
        )
        memory.observe(
            "",
            {},
            "product_id,store_id,sale_year,quantity\nP1,L1,2026,10",
            instance_id="a",
            instance_complete=True,
        )
        row = memory.state_dict()["annual_panel"][0]
        self.assertEqual((row["product"], row["locality"]), ("Chair", "North"))
        self.assertEqual(len(row["inferred_joins"]), 2)
        # A collision is not resolved by choosing the first item.
        memory.observe(
            "",
            {},
            '[{"item_code":"P1","description":"Different chair"}]',
            instance_id="b",
            instance_complete=True,
        )
        self.assertEqual(
            memory.state_dict()["annual_panel"][0]["product"], "product_id:P1"
        )

    def test_context_is_deterministic_bounded_and_independent_copy(self):
        memory = SalesMemory(max_chars=1200)
        for i in range(40):
            memory.observe(
                "",
                {},
                json.dumps(
                    {
                        "product_name": "Chair" + str(i),
                        "city": "North",
                        "year": 2026,
                        "quantity": i,
                    }
                ),
                instance_id=str(i),
                instance_complete=True,
            )
        context = memory.context()
        self.assertLessEqual(len(context), 1200)
        self.assertTrue(json.loads(context)["context_truncated"])
        self.assertEqual(context, memory.context())
        state = memory.state_dict()
        state["annual_panel"].clear()
        self.assertEqual(len(memory.state_dict()["annual_panel"]), 40)


class CohortMemoryTests(unittest.TestCase):
    def test_scoped_instruments_bias_and_sample_survival(self):
        memory = CohortMemory()
        query = "## Study 1/2: Study A\n**Enrollment:** Hospital sample\n**Region(s) in this dataset:** North\n**Patients:** 20"
        metadata = "=== Study Info ===\n  study_name: A\n  enrollment_brief: Hospital sample\n\n=== Columns ===\n  score: Instrument A  units=points  (Not directly interchangeable with instrument B)"
        memory.observe(
            query,
            {"tool_call": {"tool": "get_database_metadata"}},
            metadata,
            instance_id="a",
            instance_complete=False,
        )
        result = "=== Per-Group Survival ===\n\n  low: n=12 (60.0%)  S(12m)=0.9000  S(24m)=0.8000  S(36m)=0.7000\n  high: n=8 (40.0%)  S(12m)=0.8000  S(24m)=0.6000  S(36m)=0.4000\n\n=== Cohort Group Decomposition (1 observable, 1 unobservable) ===\n  cohort_01 (n=10): high=2 (20%), low=8 (80%)\n\nUnobservable cohorts (1, not in predict_cohort_survival):\n  cohort_02"
        memory.observe(
            query,
            {
                "tool_call": {
                    "tool": "estimate_survival_by_group",
                    "group_expression": "CASE WHEN score < 5 THEN 'high' ELSE 'low' END",
                }
            },
            result,
            instance_id="a",
            instance_complete=True,
        )
        memory.observe(
            "## Study 2/2: Study B\n**Enrollment:** Volunteers",
            {"tool_call": {"tool": "get_database_metadata"}},
            "=== Columns ===\n  score: Instrument B  units=points",
            instance_id="b",
            instance_complete=True,
        )
        state = memory.state_dict()
        self.assertEqual(len(state["studies"]), 2)
        first, second = state["studies"]
        self.assertIn("Instrument A", first["columns"]["score"])
        self.assertIn("Instrument B", second["columns"]["score"])
        self.assertEqual(first["brief"]["enrollment"], "Hospital sample")
        evidence = first["survival_evidence"][0]
        self.assertEqual(evidence["groups"][0]["n"], 12)
        self.assertEqual(
            evidence["cohort_composition"][0]["groups"][0]["fraction"], 0.2
        )
        self.assertEqual(evidence["unobservable_cohorts"], ["cohort_02"])
        self.assertEqual(second["survival_evidence"], [])

    def test_semantic_cohort_identifiers_are_not_assumed_numbered(self):
        memory = CohortMemory()
        result = "=== Cohort Group Decomposition (1 observable, 1 unobservable) ===\n  exposure_high_age_old (n=10): low=8 (80%), high=2 (20%)\nUnobservable cohorts (1, not scored):\n  marker_low"
        memory.observe(
            "",
            {"tool": "estimate_survival_by_group", "group_expression": "risk"},
            result,
            instance_id="a",
            instance_complete=True,
        )
        evidence = memory.state_dict()["studies"][0]["survival_evidence"][0]
        self.assertEqual(
            evidence["cohort_composition"][0]["cohort_id"], "exposure_high_age_old"
        )
        self.assertEqual(evidence["unobservable_cohorts"], ["marker_low"])

    def test_sample_fit_and_predictions_are_not_population_truth(self):
        memory = CohortMemory()
        memory.observe(
            "",
            {"tool": "estimate_survival_by_group", "group_expression": "'a'"},
            "=== Per-Group Survival ===\n  a: n=10 (100%)  S(12m)=0.9000  S(24m)=0.8000  S(36m)=0.7000\n=== Cohort Group Decomposition ===\n  cohort_01 (n=10): a=10 (100%)",
            instance_id="a",
            instance_complete=False,
        )
        result = "=== Per-Group Survival ===\n  a: S(12m)=0.9000  S(24m)=0.8000  S(36m)=0.7000\n=== Observable Cohort Fit (1 cohorts) ===\n  cohort_01 (n=10):  model=0.900/0.800/0.700  KM=0.800/0.700/0.600  KL=0.0123"
        memory.observe(
            "",
            {"tool": "predict_cohort_survival", "group_expression": "'a'"},
            result,
            instance_id="a",
            instance_complete=False,
        )
        memory.observe(
            "",
            {"cohort_01__s12": 0.99},
            "Report submitted. Moving to study 2",
            instance_id="a",
            instance_complete=True,
        )
        evidence = memory.state_dict()["studies"][0]["survival_evidence"][0]
        self.assertEqual(
            evidence["observable_cohort_fit"][0]["sample_km"], [0.8, 0.7, 0.6]
        )
        self.assertIn("not population truth", evidence["scope"])
        self.assertEqual(evidence["groups"][0]["n"], 10)
        self.assertEqual(evidence["cohort_composition"][0]["n"], 10)
        self.assertNotIn("0.99", memory.context())

    def test_bounded_parseable_sql_evidence_and_error_do_not_overwrite(self):
        memory = CohortMemory(max_chars=1600)
        sql_result = "grp | n\n--------+---------\nx | 20\ny | 30\n(2 rows)"
        for i in range(8):
            memory.observe(
                "",
                {
                    "tool": "query_sql",
                    "sql": "SELECT grp, COUNT(*) AS n FROM patients GROUP BY grp",
                },
                sql_result,
                instance_id=str(i),
                instance_complete=False,
            )
        memory.observe(
            "",
            {"tool": "query_sql", "sql": "SELECT missing FROM patients"},
            "SQL ERROR: no such column: missing",
            instance_id="7",
            instance_complete=True,
        )
        studies = memory.state_dict()["studies"]
        self.assertEqual(
            studies[-1]["sql_evidence"][0]["rows"], [["x", "20"], ["y", "30"]]
        )
        self.assertEqual(len(studies[-1]["errors"]), 1)
        context = memory.context()
        self.assertLessEqual(len(context), 1600)
        self.assertTrue(json.loads(context)["context_truncated"])


if __name__ == "__main__":
    unittest.main()
