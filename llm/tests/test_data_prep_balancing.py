import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tounsi_llm.config import CFG
from src.tounsi_llm.data_prep import (
    _collapse_sft_conversation_entries,
    _materialize_weighted_train_conversations,
    format_sft_conversation,
)


class SFTDataPrepBalancingTests(unittest.TestCase):
    def test_collapse_deduplicates_same_conversation_across_sources(self) -> None:
        conversation = format_sft_conversation(
            [
                {"role": "user", "content": "num client 3310 suivi commande CMD70007"},
                {"role": "assistant", "content": "نثبتلك statut w délai taqribi."},
            ]
        )
        entries = [
            {"messages": conversation, "source": "sft", "weight": 4},
            {"messages": conversation, "source": "slot_bootstrap", "weight": 8},
        ]

        collapsed = _collapse_sft_conversation_entries(entries)

        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["weight"], 8)
        self.assertEqual(set(collapsed[0]["sources"]), {"sft", "slot_bootstrap"})

    def test_materialized_train_repeats_are_capped(self) -> None:
        conversation = format_sft_conversation(
            [
                {"role": "user", "content": "aslema nheb create_order progressive"},
                {"role": "assistant", "content": "مدلي num client وبعد نكمل slots."},
            ]
        )

        materialized = _materialize_weighted_train_conversations(
            [{"messages": conversation, "source": "slot_bootstrap", "weight": 99}]
        )

        self.assertEqual(len(materialized), CFG.max_sft_train_repeats)


if __name__ == "__main__":
    unittest.main()
