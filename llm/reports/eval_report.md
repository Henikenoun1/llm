# Evaluation Report – Tounsi Call Center LLM

Generated: 2026-03-07 20:35:22

## Summary Metrics
```json
{
  "overall_task_success": 0.975,
  "per_intent_task_success": {
    "appointment_booking": 1.0,
    "order_creation": 0.9210526315789473,
    "store_info": 1.0,
    "price_inquiry": 0.9354838709677419,
    "order_tracking": 1.0
  },
  "n_tool_examples": 200,
  "tool_name_accuracy": 0.96,
  "tool_json_validity": 1.0,
  "tool_schema_validity": 0.985,
  "tool_required_fields_accuracy": 0.985,
  "slot_f1_micro": 1.0,
  "price_hallucination_rate": 0.06451612903225806,
  "policy_mismatch_rate": 0.0,
  "price_groundedness_pass_rate": 0.935483870967742,
  "lexical_jaccard_mean": 0.9986666666666667,
  "embedding_similarity_mean": 0.999330461025238,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "pii_leakage_rate": 0.0,
  "uncertainty_escalation_accuracy": 0.0,
  "uncertainty_cases": 0
}
```

## Threshold Table

| Metric | Value | Acceptable | Target | Status |
|---|---:|---:|---:|---|
| overall_task_success | 0.9750 | 0.8000 | 0.9200 | TARGET |
| tool_name_accuracy | 0.9600 | 0.9000 | 0.9700 | OK |
| tool_json_validity | 1.0000 | 0.9500 | 0.9950 | TARGET |
| tool_schema_validity | 0.9850 | 0.9000 | 0.9800 | TARGET |
| tool_required_fields_accuracy | 0.9850 | 0.9000 | 0.9800 | TARGET |
| slot_f1_micro | 1.0000 | 0.8500 | 0.9500 | TARGET |
| price_groundedness_pass_rate | 0.9355 | 0.9500 | 0.9950 | FAIL |
| price_hallucination_rate | 0.0645 | 0.0500 | 0.0050 | FAIL |
| policy_mismatch_rate | 0.0000 | 0.0500 | 0.0100 | TARGET |
| pii_leakage_rate | 0.0000 | 0.0100 | 0.0000 | TARGET |
| uncertainty_escalation_accuracy | 0.0000 | 0.8500 | 0.9500 | FAIL |

## Task Success by Intent

| Intent | Rate |
|---|---:|
| appointment_booking | 1.0000 |
| order_creation | 0.9211 |
| order_tracking | 1.0000 |
| price_inquiry | 0.9355 |
| store_info | 1.0000 |