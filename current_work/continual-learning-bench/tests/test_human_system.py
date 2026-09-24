import unittest
from typing import Optional
from unittest.mock import patch

from pydantic import BaseModel, Field

from src.interface import Query
from src.systems.human.system import Human


class SampleResponse(BaseModel):
    reasoning: str = Field(description="Why you chose this answer")
    yield_prediction: float = Field(description="Predicted yield")


class OptionalFieldResponse(BaseModel):
    action: str = Field(description="Action to take")
    amount: Optional[int] = Field(default=None, description="Optional amount")


class HumanSystemTests(unittest.TestCase):
    def test_respond_supports_guided_input(self):
        system = Human()
        query = Query(prompt="Predict a yield.", response_schema=SampleResponse)

        with patch(
            "builtins.input",
            side_effect=[
                "just guessing",
                "88",
            ],
        ):
            response = system.respond(query)

        self.assertEqual(response.action.reasoning, "just guessing")
        self.assertEqual(response.action.yield_prediction, 88.0)

    def test_respond_retries_after_invalid_guided_input(self):
        system = Human()
        query = Query(prompt="Predict a yield.", response_schema=SampleResponse)

        with patch(
            "builtins.input",
            side_effect=[
                "first try",
                "not a number",
                "42",
            ],
        ):
            response = system.respond(query)

        self.assertEqual(response.action.reasoning, "first try")
        self.assertEqual(response.action.yield_prediction, 42.0)

    def test_guided_input_allows_blank_optional_field(self):
        system = Human()
        query = Query(prompt="Choose an action.", response_schema=OptionalFieldResponse)

        with patch(
            "builtins.input",
            side_effect=[
                "CHECK",
                "",
            ],
        ):
            response = system.respond(query)

        self.assertEqual(response.action.action, "CHECK")
        self.assertIsNone(response.action.amount)


if __name__ == "__main__":
    unittest.main()
