"""
Central project configuration.

The goal of this module is to make the project reusable across domains by
moving as much behavior as possible behind files and structured config.
Changing the business context later should mostly mean changing files inside
`data/config`, the official business corpus under `data/rag`, the structured
tool-support data under `data/kb`, and the training datasets.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "tounsi_raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
KB_DIR = DATA_DIR / "kb"
RAG_DIR = DATA_DIR / "rag"
EVAL_DIR = DATA_DIR / "eval"
CONFIG_DIR = DATA_DIR / "config"
HISTORY_DIR = DATA_DIR / "history"

ARTIFACTS_DIR = ROOT / "artifacts"
ADAPTERS_DIR = ARTIFACTS_DIR / "adapters"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
CACHE_DIR = ARTIFACTS_DIR / "cache"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"
SCHEMAS_DIR = ARTIFACTS_DIR / "schemas"

REPORTS_DIR = ROOT / "reports"

for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    KB_DIR,
    RAG_DIR,
    EVAL_DIR,
    CONFIG_DIR,
    HISTORY_DIR,
    ARTIFACTS_DIR,
    ADAPTERS_DIR,
    CHECKPOINTS_DIR,
    CACHE_DIR,
    MANIFESTS_DIR,
    SCHEMAS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


SEED = 42


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


seed_everything()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPORTS_DIR / "run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("tounsi_llm")


def detect_runtime() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "cuda": False,
        "gpu_count": 0,
        "gpus": [],
        "bf16": False,
        "fp16": False,
    }
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu_count"] = torch.cuda.device_count()
            for idx in range(info["gpu_count"]):
                props = torch.cuda.get_device_properties(idx)
                info["gpus"].append(
                    {
                        "name": props.name,
                        "vram_gb": round(props.total_memory / (1024**3), 1),
                    }
                )
            info["bf16"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            info["fp16"] = True
    except Exception:
        pass
    return info


RUNTIME = detect_runtime()


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load JSON config %s: %s", path, exc)
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL line in %s", path)
    return rows


DEFAULT_DOMAIN_CONFIG: dict[str, Any] = {
    "domain_name": "generic call center",
    "organization": "your company",
    "assistant_name": "agent",
    "default_language": "tounsi",
    "supported_languages": ["tounsi", "fr", "code-switch"],
    "system_prompt": (
        "إنت agent تخدم في مركز نداء. جاوب باللهجة التونسية كي يلزم، "
        "وكون واضح ومهني ومختصر. إذا ما تعرفش معلومة قولها بصراحة."
    ),
    "system_prompts": {
        "default": (
            "إنت agent تخدم في مركز نداء. جاوب باللهجة التونسية كي يلزم، "
            "وكون واضح ومهني ومختصر. إذا ما تعرفش معلومة قولها بصراحة."
        ),
        "arabic": (
            "جاوب باللهجة التونسية بالحروف العربية. اسأل سؤال توضيحي واحد في كل مرة "
            "وما تخترعش معلومات تجارية."
        ),
        "arabizi": (
            "Reply in Tunisian Arabizi with a professional call-center tone. "
            "Do not invent business facts."
        ),
        "code_switch": (
            "استعمل darja تونسية مع termes metier بالفرنسية بطريقة طبيعية ومختصرة."
        ),
    },
    "humanized_prompt": (
        "إنت عامل حقيقي في مركز النداء، موش بوت. احكي بطبيعة، اسأل على "
        "المعطيات الناقصة، وما تخترعش معلومات."
    ),
    "datasets": [],
    "intent_keywords": {},
    "business_keywords": [],
    "required_slots": {},
    "intent_to_tool": {},
    "tool_execution_policy": {},
    "product_aliases": {},
    "location_aliases": {},
    "slot_patterns": {},
    "out_of_domain_keywords": [],
    "franco_map": {},
    "tool_descriptions": {},
    "runtime_modes": {"default": "speak"},
    "rag_source_dirs": ["data/rag"],
    "memory": {
        "history_state_path": "data/history/session_state.json",
        "conversation_log_path": "data/history/conversations.jsonl",
        "learning_buffer_path": "data/history/learning_buffer.jsonl",
        "pending_learning_path": "data/history/learning_pending.jsonl",
        "feedback_log_path": "data/history/feedback_log.jsonl",
        "approved_dpo_feedback_path": "data/history/feedback_dpo.jsonl",
        "ratings_log_path": "data/history/ratings_log.jsonl",
        "admin_corrections_path": "data/history/admin_corrections.jsonl",
        "top_k": 3,
    },
    "api": {
        "allowed_origins": ["http://localhost:4200", "http://127.0.0.1:4200"],
        "require_api_key": False,
    },
    "database": {
        "url": "",
        "echo": False,
    },
}


DOMAIN_CONFIG_PATH = CONFIG_DIR / "domain.json"
FEW_SHOTS_PATH = CONFIG_DIR / "few_shots.jsonl"
DOMAIN_CFG = _load_json(DOMAIN_CONFIG_PATH, DEFAULT_DOMAIN_CONFIG)
FEW_SHOTS_CFG = _load_jsonl(FEW_SHOTS_PATH)


@dataclass
class ProjectConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_len: int = 1536
    use_4bit: bool = True
    target_runtime_profile: str = "t4_16gb"

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    batch_size: int = 1
    grad_accum_steps: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    fp16: bool = True
    bf16: bool = False

    self_sup_learning_rate: float = 6e-5
    sft_learning_rate: float = 8e-5
    dpo_learning_rate: float = 4e-6
    epochs_self_sup: int = 2
    epochs_sft: int = 3
    epochs_dpo: int = 1
    self_sup_max_seq_len: int = 1024
    sft_max_seq_len: int = 1536
    dpo_max_seq_len: int = 1024
    self_sup_target_texts: int | None = 104000
    self_sup_max_steps: int = 1800
    sft_max_steps: int | None = 2400
    dpo_max_steps: int = 600
    self_sup_continue_from_adapter: bool = False
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    save_total_limit: int = 1
    early_stopping_patience: int = 2
    min_self_sup_train_rows: int = 25000
    min_sft_train_rows: int = 8000
    min_dpo_train_rows: int = 3000

    temperature: float = 0.35
    top_p: float = 0.9
    max_new_tokens: int = 220
    repetition_penalty: float = 1.1

    retrieval_top_k: int = 4
    few_shot_top_k: int = 3
    memory_top_k: int = 3
    max_history_messages: int = 24
    session_ttl_seconds: int = 60 * 60 * 24

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_chunk_size: int = 380
    rag_chunk_overlap: int = 60
    rag_refresh_on_startup: bool = True
    preload_models_on_startup: bool = False
    require_api_key: bool = False
    api_key_env: str = "CALL_CENTER_API_KEY"
    api_key: str | None = field(default=None, repr=False)
    database_url_env: str = "CALL_CENTER_DATABASE_URL"
    database_url: str | None = field(default=None, repr=False)
    database_echo: bool = False
    database_required_for_production: bool = False
    cors_allowed_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:4200",
            "http://127.0.0.1:4200",
        ]
    )
    strict_variant_resolution: dict[str, bool] = field(
        default_factory=lambda: {
            "prod": True,
        }
    )

    stage_output_dirs: dict[str, str] = field(
        default_factory=lambda: {
            "self_sup": "self_sup",
            "sft": "sft",
            "dpo": "dpo",
            "production": "production",
        }
    )
    variant_fallbacks: dict[str, list[str]] = field(
        default_factory=lambda: {
            "prod": ["production", "sft", "v2", "self_sup"],
            "dpo": ["dpo", "v2_dpo", "sft", "v2", "self_sup"],
            "sft": ["sft", "v2", "self_sup"],
            "self_sup": ["self_sup"],
            "base": [],
        }
    )

    def __post_init__(self) -> None:
        if not RUNTIME.get("cuda"):
            self.bf16 = False
            self.fp16 = False
        elif not RUNTIME.get("bf16"):
            self.bf16 = False
            self.fp16 = True

        rag_cfg = DOMAIN_CFG.get("rag", {})
        self.embedding_model = rag_cfg.get("embedding_model", self.embedding_model)
        self.rag_chunk_size = int(rag_cfg.get("chunk_size", self.rag_chunk_size))
        self.rag_chunk_overlap = int(rag_cfg.get("chunk_overlap", self.rag_chunk_overlap))
        self.retrieval_top_k = int(rag_cfg.get("top_k", self.retrieval_top_k))

        memory_cfg = DOMAIN_CFG.get("memory", {})
        self.memory_top_k = int(memory_cfg.get("top_k", self.memory_top_k))

        api_cfg = DOMAIN_CFG.get("api", {})
        cors_origins = api_cfg.get("allowed_origins")
        if isinstance(cors_origins, list) and cors_origins:
            self.cors_allowed_origins = [str(origin) for origin in cors_origins]
        self.require_api_key = bool(api_cfg.get("require_api_key", self.require_api_key))
        api_key = os.getenv(self.api_key_env) or api_cfg.get("api_key")
        self.api_key = str(api_key) if api_key else None

        database_cfg = DOMAIN_CFG.get("database", {})
        self.database_echo = bool(database_cfg.get("echo", self.database_echo))
        self.database_required_for_production = bool(
            database_cfg.get("required_for_production", self.database_required_for_production)
        )
        database_url = os.getenv(self.database_url_env) or database_cfg.get("url")
        self.database_url = str(database_url) if database_url else None

    def adapter_dir_for_stage(self, stage: str) -> Path:
        return ADAPTERS_DIR / self.stage_output_dirs.get(stage, stage)

    def adapter_candidates_for_variant(self, variant: str, allow_fallback: bool = True) -> list[Path]:
        names = self.variant_fallbacks.get(variant, [variant])
        if not allow_fallback and names:
            names = names[:1]
        return [ADAPTERS_DIR / name for name in names]

    def should_allow_variant_fallback(self, variant: str) -> bool:
        return not self.strict_variant_resolution.get(variant, False)

    def resolve_adapter_dir(self, variant: str, allow_fallback: bool | None = None) -> Path | None:
        if allow_fallback is None:
            allow_fallback = self.should_allow_variant_fallback(variant)
        for candidate in self.adapter_candidates_for_variant(variant, allow_fallback=allow_fallback):
            if (candidate / "adapter_config.json").exists():
                return candidate
        return None

    def available_variants(self) -> dict[str, str | None]:
        return {
            variant: (
                str(self.resolve_adapter_dir(variant))
                if self.resolve_adapter_dir(variant)
                else None
            )
            for variant in self.variant_fallbacks
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload.get("api_key"):
            payload["api_key"] = "***"
        return payload


CFG = ProjectConfig()
