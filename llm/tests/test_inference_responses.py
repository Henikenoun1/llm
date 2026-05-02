import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tounsi_llm.inference import (
    _build_grounding_context,
    _get_fallback_response,
    _is_garbage_response,
    _render_controlled_response,
)


class InferenceResponseTests(unittest.TestCase):
    def test_order_tracking_response_is_controlled(self) -> None:
        response = _render_controlled_response(
            intent="order_tracking",
            slots={"num_client": "5007", "order_id": "ORD-ABC12345"},
            missing_slots=[],
            tool_name="track_order",
            tool_result={
                "status": "ok",
                "order_id": "ORD-ABC12345",
                "order_status": "EN_FABRICATION",
                "agence": "Aouina",
                "next_slot": "10:00",
            },
            rag_results=[],
            auto_execute_tool=True,
            runtime_mode="speak",
            target_script="arabic",
        )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertIn("ORD-ABC12345", response)
        self.assertIn("Aouina", response)
        self.assertNotIn("{", response)
        self.assertNotIn("source", response.lower())

    def test_availability_response_does_not_dump_catalog(self) -> None:
        response = _render_controlled_response(
            intent="availability_inquiry",
            slots={"reference": "25YXSU"},
            missing_slots=[],
            tool_name="check_availability",
            tool_result={
                "status": "ok",
                "reference": "25YXSU",
                "catalog_match": True,
                "official_rag_match": True,
                "lens_matches": [
                    {
                        "code": "25YXSU",
                        "name": "VARILUX X DESIGN ORMA SUN",
                        "material": "ORMA",
                        "diameter": "70",
                        "text": "Programme : VARILUX X DESIGN ORMA SUN",
                    }
                ],
            },
            rag_results=[],
            auto_execute_tool=True,
            runtime_mode="speak",
            target_script="arabizi",
        )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertIn("25YXSU", response)
        self.assertIn("confirmation stock", response.lower())
        self.assertNotIn("programme :", response.lower())

    def test_grounding_context_uses_summary_not_raw_dump(self) -> None:
        context = _build_grounding_context(
            intent="availability_inquiry",
            slots={"reference": "25YXSU"},
            missing_slots=[],
            tool_result={
                "status": "ok",
                "reference": "25YXSU",
                "lens_matches": [
                    {
                        "code": "25YXSU",
                        "name": "VARILUX X DESIGN ORMA SUN",
                        "material": "ORMA",
                        "diameter": "70",
                        "text": "Programme : VARILUX X DESIGN ORMA SUN",
                    }
                ],
            },
            rag_results=[
                {
                    "score": 0.99,
                    "text": "Programme : VARILUX X DESIGN ORMA SUN",
                    "metadata": {"code": "25YXSU", "nom": "VARILUX X DESIGN ORMA SUN", "diametre": "70"},
                }
            ],
            memory_hits=[{"text": "suivi commande precedent"}],
            session_state={"active_intent": "availability_inquiry"},
        )

        self.assertIn("[tool_summary]", context)
        self.assertIn("[retrieval_summary]", context)
        self.assertNotIn("Programme : VARILUX X DESIGN ORMA SUN", context)
        self.assertNotIn("[tool_result]", context)

    def test_fallback_respects_arabizi(self) -> None:
        response = _get_fallback_response("unclear", target_script="arabizi")
        self.assertTrue(any(char.isalpha() for char in response))
        self.assertNotIn("عاود", response)

    def test_get_num_client_response_acknowledges_detected_value(self) -> None:
        response = _render_controlled_response(
            intent="get_num_client",
            slots={"num_client": "5007"},
            missing_slots=[],
            tool_name=None,
            tool_result=None,
            rag_results=[],
            auto_execute_tool=False,
            runtime_mode="speak",
            target_script="arabizi",
        )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertIn("5007", response)
        self.assertNotIn("3atini num client", response.lower())

    def test_greeting_response_softens_social_check_in(self) -> None:
        response = _render_controlled_response(
            intent="greeting",
            user_text="chhalek",
            slots={},
            missing_slots=[],
            tool_name=None,
            tool_result=None,
            rag_results=[],
            auto_execute_tool=False,
            runtime_mode="speak",
            target_script="arabizi",
        )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertIn("hamdoullah", response.lower())
        self.assertIn("n3awnek", response.lower())

    def test_agent_identity_response_mentions_name_and_role(self) -> None:
        response = _render_controlled_response(
            intent="agent_identity",
            slots={},
            missing_slots=[],
            tool_name=None,
            tool_result=None,
            rag_results=[],
            auto_execute_tool=False,
            runtime_mode="speak",
            target_script="arabizi",
        )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertIn("Mohsen", response)
        self.assertIn("SIVO", response)
        self.assertIn("suivi commande", response.lower())

    def test_current_date_response_uses_tunisia_date(self) -> None:
        response = _render_controlled_response(
            intent="current_date",
            slots={},
            missing_slots=[],
            tool_name=None,
            tool_result=None,
            rag_results=[],
            auto_execute_tool=False,
            runtime_mode="speak",
            target_script="arabizi",
        )

        self.assertIsNotNone(response)
        assert response is not None
        today = datetime.now(ZoneInfo("Africa/Tunis")).strftime("%d/%m/%Y")
        self.assertIn(today, response)

    def test_clarify_need_response_does_not_ask_num_client_first(self) -> None:
        response = _render_controlled_response(
            intent="clarify_need",
            slots={},
            missing_slots=[],
            tool_name=None,
            tool_result=None,
            rag_results=[],
            auto_execute_tool=False,
            runtime_mode="speak",
            target_script="arabizi",
        )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertIn("suivi commande", response.lower())
        self.assertNotIn("num client", response.lower())

    def test_garbage_response_flags_prompt_leak(self) -> None:
        leaked = """score=0.2611; agence=Ariana\n[context]\nnum_client=3310\n[agent]\nChenna 9olli tawa"""
        self.assertTrue(_is_garbage_response(leaked))


if __name__ == "__main__":
    unittest.main()
