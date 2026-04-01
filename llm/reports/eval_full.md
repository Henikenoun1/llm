# Evaluation Report

## Data Summary

### self_sup_train
- examples: 23
- avg_tokens: 19.65
- avg_chars: 100.57
- duplicate_examples: 0
- clean_tounsi_rate: 100.0

### self_sup_val
- examples: 2
- avg_tokens: 11
- avg_chars: 57
- duplicate_examples: 0
- clean_tounsi_rate: 100.0

### sft_train
- examples: 23
- avg_turns: 3
- avg_user_tokens: 14.83
- avg_assistant_tokens: 11.52
- assistant_clean_tounsi_rate: 100.0
- duplicate_examples: 0

### sft_val
- examples: 1
- avg_turns: 3
- avg_user_tokens: 39
- avg_assistant_tokens: 33
- assistant_clean_tounsi_rate: 100.0
- duplicate_examples: 0

### sft_test
- examples: 1
- avg_turns: 3
- avg_user_tokens: 8
- avg_assistant_tokens: 6
- assistant_clean_tounsi_rate: 100.0
- duplicate_examples: 0

### dpo_train
- examples: 6616
- avg_prompt_tokens: 12.85
- avg_chosen_tokens: 8.84
- avg_rejected_tokens: 9.62
- malformed_examples: 0

### dpo_val
- examples: 735
- avg_prompt_tokens: 12.95
- avg_chosen_tokens: 9.02
- avg_rejected_tokens: 9.15
- malformed_examples: 0

## Inference Summary

- cases_total: 4
- cases_scored: 4
- errors: 0
- intent_accuracy: 100.0
- tool_accuracy: 75.0
- human_review_accuracy: 50.0
- slot_precision: 0.6667
- slot_recall: 0.6667
- slot_f1: 0.6667
- response_rougeL_f1: 0.0
- response_rouge1_f1: 0.0
- response_rouge2_f1: 0.0
- response_bleu: 0.0172
- response_chrf: 0.0038
- tounsi_rate: 100.0
- chinese_rate: 0.0
- keyword_pass_rate: 100.0
- forbidden_pass_rate: 100.0
- avg_fusha_markers: 0.0
- avg_latency_ms: 93.48

## Recommendations

- Track intent accuracy, slot F1, tool accuracy and human review accuracy together.
- Use ROUGE-L, BLEU and chrF only for cases with stable reference responses.
- Keep separate eval suites for collect mode and autonomous mode.
- Review error cases in the JSON report before promoting a new production adapter.
