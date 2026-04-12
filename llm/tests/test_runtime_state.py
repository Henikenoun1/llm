import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tounsi_llm.config import DOMAIN_CFG
from src.tounsi_llm.corrections import LiveCorrectionStore
from src.tounsi_llm.inference import _recover_missing_slots_from_turn, _resolve_turn_state
from src.tounsi_llm.memory import ConversationMemoryStore


class RuntimeStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._memory_backup = copy.deepcopy(DOMAIN_CFG.get("memory", {}))
        base = Path(self._tmp.name)
        DOMAIN_CFG["memory"] = {
            "history_state_path": str(base / "session_state.json"),
            "conversation_log_path": str(base / "conversations.jsonl"),
            "learning_buffer_path": str(base / "learning_buffer.jsonl"),
            "pending_learning_path": str(base / "learning_pending.jsonl"),
            "feedback_log_path": str(base / "feedback_log.jsonl"),
            "approved_dpo_feedback_path": str(base / "feedback_dpo.jsonl"),
            "ratings_log_path": str(base / "ratings_log.jsonl"),
            "admin_corrections_path": str(base / "admin_corrections.jsonl"),
            "top_k": 3,
        }

    def tearDown(self) -> None:
        DOMAIN_CFG["memory"] = self._memory_backup
        self._tmp.cleanup()

    def test_resolve_turn_state_carries_open_form_slots(self) -> None:
        intent, slots = _resolve_turn_state(
            "fi tunis",
            "unknown",
            {"city": "Tunis"},
            {
                "active_intent": "order_creation",
                "open_form": True,
                "slots": {"product": "progressive", "index": "1.67"},
            },
        )
        self.assertEqual(intent, "create_order")
        self.assertEqual(
            slots,
            {
                "product": "progressive",
                "index": "1.67",
                "city": "Tunis",
            },
        )

    def test_feedback_promotes_approved_learning_examples(self) -> None:
        store = ConversationMemoryStore()
        session_id = "session-test"

        store.update_session_state(
            session_id,
            intent="order_tracking",
            slots={"order_id": "ORD-ABC12345"},
            missing_slots=[],
            tool_call={"name": "track_order", "args": {"order_id": "ORD-ABC12345"}},
            tool_result={"status": "ok", "order_id": "ORD-ABC12345"},
        )
        store.append_exchange(
            session_id,
            "suivi ORD-ABC12345",
            "commande en cours",
            model_variant="prod",
            metadata={
                "intent": "order_tracking",
                "slots": {"order_id": "ORD-ABC12345"},
                "tool_call": {"name": "track_order", "args": {"order_id": "ORD-ABC12345"}},
                "tool_result": {"status": "ok", "order_id": "ORD-ABC12345"},
            },
        )

        result = store.capture_feedback(
            session_id,
            reviewer_id="agent-1",
            corrected_response="commande prête pour livraison",
            approve_for_training=True,
        )

        self.assertTrue(result["approved_sft"])
        self.assertTrue(result["approved_dpo"])

        stats = store.learning_stats()
        self.assertEqual(stats["pending_candidates"], 1)
        self.assertEqual(stats["approved_sft"], 1)
        self.assertEqual(stats["approved_dpo"], 1)
        self.assertEqual(stats["feedback_events"], 1)

    def test_live_admin_correction_matches_runtime_query(self) -> None:
        store = LiveCorrectionStore()
        store.add_correction(
            pattern_text="suivi commande ORD-ABC12345",
            corrected_response="Commande validée par l'admin.",
            intent="order_tracking",
            runtime_mode="speak",
        )

        match = store.find_best(
            user_text="suivi commande ORD-ABC12345",
            intent="order_tracking",
            slots={"order_id": "ORD-ABC12345"},
            runtime_mode="speak",
        )

        self.assertIsNotNone(match)
        self.assertEqual(match["corrected_response"], "Commande validée par l'admin.")

    def test_short_reply_recovers_num_client_from_missing_slots(self) -> None:
        recovered = _recover_missing_slots_from_turn(
            text="5007",
            extracted_slots={},
            session_state={"missing_slots": ["num_client"]},
        )
        self.assertEqual(recovered.get("num_client"), "5007")

    def test_short_reply_recovers_order_id_from_missing_slots(self) -> None:
        recovered = _recover_missing_slots_from_turn(
            text="CMD70007",
            extracted_slots={},
            session_state={"missing_slots": ["order_id"]},
        )
        self.assertEqual(recovered.get("order_id"), "CMD70007")


if __name__ == "__main__":
    unittest.main()
