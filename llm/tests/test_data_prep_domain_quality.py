import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tounsi_llm.data_prep import (
    _build_rejected_dpo_response,
    _dpo_pair_ok,
    conversation_domain_ok,
    format_sft_conversation,
)


class DataPrepDomainQualityTests(unittest.TestCase):
    def test_generic_chitchat_is_not_considered_call_center_domain(self) -> None:
        conversation = format_sft_conversation(
            [
                {"role": "user", "content": "شنوة تعمل الليلة؟"},
                {"role": "assistant", "content": "مازلت ما قررتش."},
            ]
        )

        self.assertFalse(conversation_domain_ok(conversation, source="sft"))

    def test_optical_call_center_conversation_is_domain_relevant(self) -> None:
        conversation = format_sft_conversation(
            [
                {"role": "user", "content": "aslema num client 3310 nheb suivi commande CMD2612345678"},
                {"role": "assistant", "content": "حاضر، نثبت dossier ونعطيك statut + délai taqribi lel agence."},
            ]
        )

        self.assertTrue(conversation_domain_ok(conversation, source="sft"))

    def test_dpo_pair_must_stay_domain_relevant(self) -> None:
        generic_pair = {
            "prompt": "شنوة تعمل الليلة؟",
            "chosen": "مازلت ما قررتش.",
            "rejected": "أكيد كل شي confirmé.",
        }

        self.assertFalse(_dpo_pair_ok(generic_pair))

    def test_synthetic_rejected_dpo_response_stays_business_specific(self) -> None:
        prompt = "Client: aslema num client 3310 nheb suivi commande CMD2612345678"
        chosen = "حاضر، نثبت dossier ونعطيك statut + délai taqribi lel agence."
        rejected = _build_rejected_dpo_response(prompt, chosen)

        self.assertNotEqual(rejected, chosen)
        self.assertTrue(any(token in rejected.lower() for token in ["commande", "livraison", "stock", "catalog"]))
        self.assertTrue(_dpo_pair_ok({"prompt": prompt, "chosen": chosen, "rejected": rejected}))


if __name__ == "__main__":
    unittest.main()
