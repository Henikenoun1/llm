import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tounsi_llm.config import CFG, DOMAIN_CFG
from src.tounsi_llm.evaluation import evaluate_processed_data
from src.tounsi_llm.storage import DatabaseBackend


class EvaluationStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._memory_backup = copy.deepcopy(DOMAIN_CFG.get("memory", {}))
        self._db_url_backup = CFG.database_url
        self.backend = None
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
        CFG.database_url = f"sqlite:///{base / 'test.db'}"

    def tearDown(self) -> None:
        if self.backend is not None:
            self.backend.close()
        DOMAIN_CFG["memory"] = self._memory_backup
        CFG.database_url = self._db_url_backup
        self._tmp.cleanup()

    def test_database_backend_records_entities(self) -> None:
        self.backend = DatabaseBackend()
        if not self.backend.enabled:
            self.skipTest("database backend dependencies are not installed")

        self.backend.save_session_states(
            {
                "session-1": {
                    "history": [{"role": "user", "content": "hello"}],
                    "last_active": 1.0,
                    "state": {"active_intent": "greeting", "slots": {}},
                }
            }
        )
        self.backend.record_conversation(
            {
                "session_id": "session-1",
                "timestamp": 1.0,
                "model_variant": "prod",
                "user": "hello",
                "assistant": "salem",
                "metadata": {"intent": "greeting"},
            }
        )
        self.backend.record_learning_example({"session_id": "session-1", "timestamp": 1.0}, status="approved", source="test")
        self.backend.record_rating({"session_id": "session-1", "timestamp": 1.0, "verdict": "good", "notes": ""})
        self.backend.record_admin_correction(
            {
                "timestamp": 1.0,
                "pattern_text": "hello",
                "normalized_pattern": "hello",
                "intent": "greeting",
                "runtime_mode": "speak",
                "action": "replace",
                "corrected_response": "salem",
                "reviewer_id": "admin",
                "notes": "",
                "slots": {},
            }
        )

        counts = self.backend.counts()
        self.assertEqual(counts["session_states"], 1)
        self.assertEqual(counts["conversations"], 1)
        self.assertEqual(counts["learning_examples"], 1)
        self.assertEqual(counts["ratings"], 1)
        self.assertEqual(counts["admin_corrections"], 1)

    def test_processed_data_eval_has_expected_sections(self) -> None:
        report = evaluate_processed_data()
        self.assertIn("processed_data", report)
        self.assertIsInstance(report["processed_data"], dict)


if __name__ == "__main__":
    unittest.main()
