# Changement Ahmed - Resume push

## Conclusion directe

- SFT reste base sur SFTTrainer (fine-tuning supervise).
- DPO reste base sur DPOTrainer (prompt/chosen/rejected).

## Ce qui a change
- Nettoyage des elements lies au sentiment dans la preparation des donnees et la config.
- Alignement des intents runtime: ajout de order_creation en plus de create_order dans la config domaine.
- Separation plus claire ingestion/preparation:
  - nouveau module data_sources.py pour download + extraction + manifest.
  - nouveau module data_audit.py pour audit des datasets bruts.
- Pipeline train enrichi:
  - ajout du stage audit.
  - ajout d options self-sup (max_steps, max_seq_len, fresh_adapter).
- Enrichissement SFT avec un fichier local d intents operateur:
  - data/config/commandes_intents.jsonl.

## Data utilisees
- SFT:
  - wghezaiel/SFT-Tunisian-Derja
  - abdouuu/tunisian_chatbot_data (role sft)
  - data/config/commandes_intents.jsonl
- DPO:
  - wghezaiel/DPO-Tunisian-Derja
  - hamzabouajila/machine-translation-en-tn-msa-dpo-v2
  - data/history/feedback_dpo.jsonl (si approuve)
- Self-supervised:
  - abdouuu/tunisian_chatbot_data
  - data/tounsi_raw/data_full_local.jsonl
  - linagora/linto-dataset-text-ar-tn
  - chaymafourati/tunizi
  - samfatnassi/Tunisian-Railway-Dialogues
  - linagora/Tunisian_Derja_Dataset (TunSwitchTunisiaOnly, TuDiCOI)
  - hamzabouajila/tunisian-derja-unified-raw-corpus


