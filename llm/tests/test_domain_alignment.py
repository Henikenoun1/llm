import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tounsi_llm.inference import (
    _preferred_response_script,
    _strip_leaked_english,
    extract_slots,
    infer_intent,
    route_to_tool,
)
from src.tounsi_llm.rag import VectorRAGRetriever
from src.tounsi_llm.tools import ToolRegistry
from src.tounsi_llm.validation import validate_domain_assets


class DomainAlignmentTests(unittest.TestCase):
    def test_tracking_request_extracts_core_slots(self) -> None:
        text = "bonjour 5007 نحب نعرف وين واصلة commande 70007"
        slots = extract_slots(text)
        intent = infer_intent(text, extracted_slots=slots)
        tool_name, _, missing = route_to_tool(intent, slots)

        self.assertEqual(intent, "order_tracking")
        self.assertEqual(slots.get("num_client"), "5007")
        self.assertEqual(slots.get("order_id"), "70007")
        self.assertEqual(tool_name, "track_order")
        self.assertEqual(missing, [])

    def test_greeting_with_inline_num_client_is_detected_in_arabizi(self) -> None:
        text = "3aslema 4560 nheb commande progressive 1.67 marron"
        slots = extract_slots(text)
        intent = infer_intent(text, extracted_slots=slots)

        self.assertEqual(slots.get("num_client"), "4560")
        self.assertEqual(intent, "create_order")

    def test_maak_marker_with_inline_num_client_is_detected_in_arabic_and_arabizi(self) -> None:
        for text in [
            "aslema maak 4580 nheb suivi commande ORD-ABC12345",
            "عسلامة معاك 4580 نحب suivi commande ORD-ABC12345",
        ]:
            with self.subTest(text=text):
                slots = extract_slots(text)
                intent = infer_intent(text, extracted_slots=slots)

                self.assertEqual(slots.get("num_client"), "4580")
                self.assertEqual(slots.get("order_id"), "ORD-ABC12345")
                self.assertEqual(intent, "order_tracking")

    def test_create_order_tool_returns_non_destructive_draft(self) -> None:
        registry = ToolRegistry()
        result = registry.create_order(
            num_client="3310",
            product="Varilux liberty short",
            material="orma",
            treatment="crizal prevencia",
            diameter="70",
        )

        self.assertEqual(result["status"], "draft")
        self.assertTrue(result["needs_confirmation"])
        self.assertFalse(result["submitted"])
        self.assertEqual(result["num_client"], "3310")

    def test_reference_confirmation_uses_known_examples(self) -> None:
        registry = ToolRegistry()
        result = registry.confirm_reference("16 83")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reference"], "1683")
        self.assertGreaterEqual(result["seen_count"], 1)

    def test_reference_confirmation_can_use_external_rag_catalog(self) -> None:
        registry = ToolRegistry()
        result = registry.confirm_reference("25YXSU")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reference"], "25YXSU")
        self.assertEqual(result["source"], "official_rag_catalogue_verres")

    def test_delivery_schedule_intent_and_tool_routing(self) -> None:
        text = "planning livraison agence aouina secteur marsa / lac"
        slots = extract_slots(text)
        intent = infer_intent(text, extracted_slots=slots)
        tool_name, tool_args, missing = route_to_tool(intent, slots)

        self.assertEqual(intent, "delivery_schedule")
        self.assertEqual(tool_name, "get_delivery_schedule")
        self.assertEqual(missing, [])
        self.assertIn("agence", tool_args)

    def test_extract_slots_can_use_official_rag_for_delivery_terms(self) -> None:
        text = "suivi commande CMD2612345678 client 3310 livraison agence Ben Arous secteur mourouj"
        slots = extract_slots(text)

        self.assertEqual(slots.get("agence"), "Ben Arous")
        self.assertEqual(slots.get("secteur"), "mourouj")
        self.assertEqual(slots.get("order_id"), "CMD2612345678")

    def test_extract_slots_can_detect_rag_lens_code_without_reference_keyword(self) -> None:
        text = "25YXSU dispo tawa?"
        slots = extract_slots(text)
        intent = infer_intent(text, extracted_slots=slots)

        self.assertEqual(slots.get("lens_code"), "25YXSU")
        self.assertEqual(slots.get("reference"), "25YXSU")
        self.assertEqual(intent, "availability_inquiry")

    def test_delivery_schedule_picks_next_slot_after_requested_time(self) -> None:
        registry = ToolRegistry()
        result = registry.get_delivery_schedule(
            agence="Aouina",
            secteur="marsa / lac",
            requested_slot="12:01",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["next_slot"], "15h30")
        self.assertEqual(result["delivery_target"], "optician_agency")

    def test_track_order_attaches_agency_delivery_context(self) -> None:
        registry = ToolRegistry()
        result = registry.track_order(
            num_client="3310",
            order_id="CMD2612345678",
            agence="Aouina",
            secteur="marsa / lac",
            requested_slot="12:01",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["delivery_target"], "optician_agency")
        self.assertEqual(result["delivery_schedule"]["next_slot"], "15h30")

    def test_validation_has_no_eval_tool_mismatch(self) -> None:
        VectorRAGRetriever(refresh=True)
        report = validate_domain_assets(write_report=False)

        self.assertEqual(report["eval_cases"]["tool_mismatches"], [])
        self.assertFalse(any("Intent missing tool mapping" in issue for issue in report["issues"]))
        self.assertTrue(
            next(
                check["passed"]
                for check in report["preflight_readiness"]["checks"]
                if check["name"] == "coherence_issues"
            )
        )
        self.assertEqual(report["rag"]["kb_file_count"], 0)
        self.assertGreaterEqual(report["rag_training"]["sft_conversation_count"], 100)
        self.assertGreaterEqual(report["rag_training"]["self_sup_text_count"], 200)

    def test_preferred_response_script_matches_user_style(self) -> None:
        self.assertEqual(_preferred_response_script("aslema nheb prix progressive"), "arabizi")
        self.assertEqual(_preferred_response_script("عسلامة نحب نعرف السوم"), "arabic")

    def test_strip_leaked_english_keeps_arabizi(self) -> None:
        text = "aslema nheb na3ref prix w statut mta3 commande"
        cleaned = _strip_leaked_english(text, target_script="arabizi")
        self.assertEqual(cleaned, text)


if __name__ == "__main__":
    unittest.main()
