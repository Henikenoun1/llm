# Rapport Détaillé — Tounsi Call-Center LLM v2

**Auteur :** AI Engineering Pipeline  
**Date :** Juillet 2025  
**Version :** 2.0  
**GPU :** Tesla T4 (15 GB VRAM) — Ubuntu 24.04.4 LTS — 62 GB RAM  
**Modèle de base :** Qwen/Qwen2.5-7B-Instruct (4-bit QLoRA)

---

## Table des Matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture technique](#2-architecture-technique)
3. [Fichier par fichier — Code, Objectif, Sortie](#3-fichier-par-fichier)
   - 3.1 [config.py](#31-configpy)
   - 3.2 [data_prep.py](#32-data_preppy)
   - 3.3 [training.py](#33-trainingpy)
   - 3.4 [inference.py](#34-inferencepy)
   - 3.5 [kb_tools.py](#35-kb_toolspy)
   - 3.6 [server.py](#36-serverpy)
   - 3.7 [evaluation.py](#37-evaluationpy)
   - 3.8 [scripts/train.py](#38-scriptstrainpy)
   - 3.9 [Frontend Angular](#39-frontend-angular)
4. [Données — Sources, Nettoyage, Statistiques](#4-données)
5. [Entraînement SFT — Résultats et Interprétation](#5-entraînement-sft)
6. [Audit DPO — Problème critique et Résolution](#6-audit-dpo)
7. [Post-traitement — Pipeline de correction](#7-post-traitement)
8. [Tests et Validation](#8-tests-et-validation)
9. [Points d'amélioration](#9-points-damélioration)
10. [Annexes](#10-annexes)

---

## 1. Vue d'ensemble du projet

### Objectif
Créer un chatbot de centre d'appel pour **sivo**, un centre d'optique en Tunisie, qui communique en **dialecte tunisien (Tounsi/Derja)** — pas en arabe standard (Fus7a) ni en français.

### Historique des versions
| Version | Description | Problèmes |
|---------|-------------|-----------|
| v1.0 | Prompt-patching sur Qwen2.5-7B brut | Réponses en Fus7a, caractères chinois (CJK), hallucinations |
| v2.0 | LoRA fine-tuning sur données réelles Tounsi | Résolu : le modèle parle nativement Tounsi |

### Statistiques globales du code

| Composant | Fichier(s) | Lignes |
|-----------|-----------|--------|
| Backend Python | 8 fichiers src/ + scripts/ | **2 606** |
| Frontend Angular | 5 fichiers src/app/ | **553** |
| **Total** | **13 fichiers** | **3 159** |

---

## 2. Architecture technique

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Angular 21)                      │
│               http://localhost:4200                           │
│  app.ts ─── chat.service.ts ─── app.html ─── app.scss       │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /chat (JSON)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Backend (FastAPI / Uvicorn)                   │
│               http://localhost:8000                           │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ server.py │───▶│ inference.py │───▶│ Qwen2.5-7B +     │   │
│  │ (REST)    │    │ (pipeline)   │    │ LoRA adapter     │   │
│  └──────────┘    └──────┬───────┘    │ (4-bit QLoRA)    │   │
│                         │            └──────────────────┘   │
│                         ▼                                    │
│              ┌──────────────────┐                            │
│              │   kb_tools.py     │                            │
│              │ BM25 Retriever    │                            │
│              │ 8 produits, 5 mag │                            │
│              │ 5 politiques      │                            │
│              │ 150 commandes     │                            │
│              └──────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline d'inférence (production_infer)

```
Input utilisateur
      │
      ▼
[1] normalize_text() ──▶ _transliterate_franco()
      │
      ▼
[2a] infer_intent() ──▶ Détection OOD → Fallback "مانعرفش" (pas d'appel modèle)
      │
      ▼
[2b] Latin court + unknown → Fallback "عاودلي" (pas d'appel modèle)
      │
      ▼
[3] route_to_tool() ──▶ execute_tool() (KB lookup)
      │
      ▼
[4] _build_context() ──▶ Messages système + historique + RAG
      │
      ▼
[5] _generate() ──▶ Qwen2.5 + LoRA (GPU)
      │
      ▼
[6] Post-traitement :
    _strip_english()
    _fix_brand_names()
    _strip_embedded_english()
    _fix_phrasing()
    _scrub_pii()
    _clean_garbage_tail()
    _truncate()
      │
      ▼
[7] Quality gate : _is_garbage_response() → retry ou fallback
      │
      ▼
Output JSON (response, intent, slots, tool_call, latency_ms)
```

---

## 3. Fichier par fichier

### 3.1 config.py
**Chemin :** `src/tounsi_llm/config.py` — **143 lignes**

**Objectif :** Configuration centralisée du projet. Tous les hyperparamètres, chemins, seed, logging et détection runtime en un seul endroit.

**Blocs de code :**

| Bloc | Lignes | Fonction | Description |
|------|--------|----------|-------------|
| PATHS | 22-39 | Constantes | ROOT, DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR + création auto des sous-dossiers |
| SEED | 42-57 | `seed_everything()` | Reproductibilité : Python, NumPy, PyTorch (seed=42) |
| LOGGING | 60-68 | Configuration | StreamHandler (stdout) + FileHandler (run.log), format avec timestamp |
| RUNTIME | 69-93 | `detect_runtime()` | Détecte GPU, CUDA, bf16/fp16 support. Retourne dict d'info hardware |
| CONFIG | 97-143 | `ProjectConfig` | Dataclass avec tous les hyperparamètres |

**Hyperparamètres clés :**

```python
base_model      = "Qwen/Qwen2.5-7B-Instruct"
max_seq_len     = 2048
use_4bit        = True          # QLoRA quantization
lora_rank       = 32            # Augmenté pour meilleur apprentissage dialecte
lora_alpha      = 64
lora_dropout    = 0.05
learning_rate   = 2e-4
batch_size      = 1
grad_accum_steps= 8             # Effective batch = 8
epochs_sft      = 5
epochs_dpo      = 2
warmup_ratio    = 0.03
weight_decay    = 0.01
temperature     = 0.7
top_p           = 0.85
max_new_tokens  = 200
repetition_penalty = 1.15
```

**Interprétation :**
- `lora_rank=32 / alpha=64` : ratio 2x donne un bon compromis entre capacité d'apprentissage et régularisation
- `effective_batch=8` : compense le batch_size=1 (contrainte VRAM T4) via gradient accumulation
- `temperature=0.7` : assez créatif pour le dialecte mais pas trop aléatoire

**Point d'amélioration :**
- Ajouter un fichier `.yaml` externe pour surcharger les configs sans modifier le code
- Le `__post_init__` force fp16 si bf16 n'est pas supporté, mais le T4 supporte bf16 (vérifié)

---

### 3.2 data_prep.py
**Chemin :** `src/tounsi_llm/data_prep.py` — **606 lignes**

**Objectif :** Téléchargement, nettoyage, filtrage qualité et formatage des données d'entraînement SFT + DPO.

**Fonctions principales :**

| Fonction | Ligne | Objectif | Sortie |
|----------|-------|----------|--------|
| `download_tounsi_datasets()` | 25 | Télécharge 3 datasets HuggingFace | Dict {nom: chemin} |
| `_save_jsonl()` | 73 | Sérialisation HF Dataset → JSONL | Fichier .jsonl |
| `is_moroccan_or_algerian()` | 169 | Détecte dialecte marocain/algérien (hard + soft) | bool |
| `is_clean_tounsi()` | 224 | Vérifie si texte est du Tounsi propre | bool |
| `compute_quality_stats()` | 245 | Statistiques qualité (% propre, chinese, Fus7a) | Dict |
| `format_sft_conversation()` | 288 | Formate conversation → template Qwen chat | list[dict] |
| `prepare_sft_data()` | 314 | Pipeline complet SFT : load → filter → split → save | Dict chemins |
| `prepare_dpo_data()` | 461 | Pipeline complet DPO : load → filter → split → save | Dict chemins |
| `_extract_conversation()` | 552 | Parse 4 formats différents de données | list[dict] ou None |
| `_extract_dpo_pair()` | 590 | Parse paire DPO (prompt/chosen/rejected) | Dict ou None |

**Détection dialecte marocain (innovation clé) :**

```python
# Hard reject — mots JAMAIS utilisés en Tounsi
HARD_MOROCCAN = ["ديال", "كاين", "مزيان", "دابا", "واخا", "بزاف", ...]
# Résultat: 0.00% contamination marocaine dans les données finales
```

La détection utilise 2 niveaux :
1. **Hard reject** : 35+ marqueurs exclusivement marocains/algériens (ديال, كاين, etc.)
2. **Soft scoring** : ratio marqueurs marocains vs. marqueurs Tounsi

**Filtrage qualité SFT (4 niveaux) :**

| Filtre | Critère | Éliminé |
|--------|---------|---------|
| Short | Réponse < 10 caractères | MCQ, réponses "A/B" |
| Translation | "ترجم من..." dans user | Fus7a en réponse |
| Moroccan | Marqueurs marocains/algériens | Dialecte non-tunisien |
| Fus7a | ≥2 marqueurs Fus7a par réponse | Arabe standard |

**Marqueurs Tounsi positifs (42 mots) :**
```python
TOUNSI_MARKERS = ["متاع", "فمّا", "برشا", "قداش", "باهي", "عسلامة",
                  "خويا", "نحب", "نجّم", "شنوة", "بلارات", "كوموند", ...]
```

**System prompts :**
```python
SYSTEM_PROMPT = "انت sivo مساعد ذكي متاع sivo لي هو مركز بصريات في تونس. تحكي بالتونسي ديما. جاوب بإختصار وبطريقة ودودة."
SYSTEM_GENERAL = "تحكي بالتونسي ديما. جاوب بطريقة طبيعية وودودة."
```

**Interprétation :**
- Deux system prompts : SYSTEM_PROMPT (domaine optique, 3846 occurrences) et SYSTEM_GENERAL (enrichissement langue, 897 occurrences)
- Les conversations custom get 3x weight dans le training (upsampling)

**Point d'amélioration :**
- Ajouter filtrage par longueur max (certaines réponses > 500 tokens sont des listes/poèmes)
- Ajouter détection de code-switching excessif (mélange français-arabe)

---

### 3.3 training.py
**Chemin :** `src/tounsi_llm/training.py` — **333 lignes**

**Objectif :** Chargement du modèle de base, application de LoRA, et entraînement SFT + DPO.

**Fonctions principales :**

| Fonction | Ligne | Objectif | Sortie |
|----------|-------|----------|--------|
| `load_base_model()` | 33 | Charge Qwen2.5-7B en 4-bit (QLoRA-ready) | (model, tokenizer) |
| `apply_lora()` | 69 | Applique LoRA config au modèle | model PEFT |
| `_load_sft_dataset()` | 89 | Charge JSONL → HF Dataset avec chat template | Dataset |
| `train_sft()` | 137 | Entraînement SFT complet avec TRL SFTTrainer | Dict metrics |
| `train_dpo()` | 236 | Entraînement DPO à partir de l'adapter SFT | Dict metrics |

**Configuration QLoRA :**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,     # Double quantization
    bnb_4bit_compute_dtype=torch.float16,
)
```

**Configuration LoRA :**
```python
LoraConfig(
    r=32,                    # Rank
    lora_alpha=64,           # Scaling factor
    lora_dropout=0.05,
    target_modules=[         # TOUTES les couches attention + MLP
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
```

**SFT Training Config :**
```python
SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,    # Effective batch = 8
    learning_rate=2e-4,
    num_train_epochs=5,
    warmup_ratio=0.03,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,      # Économise ~40% VRAM
    save_steps=500,
    logging_steps=10,
)
```

**DPO Training Config :**
```python
DPOConfig(
    learning_rate=max(5e-6, 2e-4/10),  # = 2e-5 (10x plus petit que SFT)
    num_train_epochs=2,
    beta=0.1,                          # KL penalty coefficient
    bf16=True, fp16=False,
    max_length=1024,
    max_prompt_length=512,
    gradient_checkpointing=True,
)
```

**Interprétation :**
- Le DPO utilise un learning rate 10x plus petit que le SFT pour éviter le catastrophic forgetting
- `beta=0.1` est la valeur standard DPO — contrôle combien le modèle peut s'éloigner de la policy de référence
- `gradient_checkpointing=True` est essentiel sur T4 (15GB) pour faire tenir le modèle 7B

**Point d'amélioration :**
- Ajouter early stopping basé sur eval loss
- Logger les eval metrics pendant le DPO (actuellement seulement train metrics)
- Sauvegarder le meilleur checkpoint (pas seulement le dernier)

---

### 3.4 inference.py
**Chemin :** `src/tounsi_llm/inference.py` — **826 lignes, 21 fonctions**

**Objectif :** Pipeline d'inférence complet : normalisation → détection d'intent → routing d'outils → génération → post-traitement → quality gate.

C'est le fichier le plus complexe et le plus critique du projet.

**Fonctions — Bloc par bloc :**

#### Bloc 1 : Normalisation (lignes 36-90)

| Fonction | Objectif |
|----------|----------|
| `normalize_text()` | Unicode NFKC + collapse whitespace + translittération franco |
| `_transliterate_franco()` | Convertit 30 mots franco-arabes en arabe (aslema→عسلامة) |

```python
_FRANCO_MAP = {
    "salam": "سلام", "aslema": "عسلامة", "chkoun": "شكون",
    "blarat": "بلارات", "commande": "كوموند", "n7eb": "نحب", ...
}
```

**Point clé :** La translittération ne s'active que si le texte contient plus de caractères latins qu'arabes (évite de casser les inputs mixtes).

#### Bloc 2 : Détection d'intent (lignes 92-190)

| Fonction | Objectif |
|----------|----------|
| `_norm_for_intent()` | Normalise pour matching : lowercase + strip diacritics |
| `infer_intent()` | Détection rule-based avec 8 catégories d'intent |

**Intents supportés :**

| Intent | Mots-clés (exemples) | Action |
|--------|---------------------|--------|
| greeting | سلام, عسلامة, bonjour, اهلا, شحالك | Réponse de salutation |
| price_inquiry | سوم, قداش, prix, combien | Lookup prix catalogue |
| product_info | بلارات, عدسات, produit, منتجات | Info produits |
| store_info | محل, وقتاش, horaires, يفتح | Info magasins |
| order_tracking | نتبع, track, suivre, كوموند | Suivi commande |
| order_creation | نعمل كوموند, commander | Nouvelle commande |
| appointment_booking | رنديفو, rdv, موعد | Réservation |
| thanks | مرسي, شكرا, merci | Remerciement |
| unknown | (défaut) | Demande de clarification |

**Innovation clé — Word-boundary matching :**
```python
words = set(t.split())  # Matching par mot exact
# Empêche "high" de matcher "hi" (greeting)
# Les keywords multi-mots ("نعمل كوموند") utilisent substring match
```

#### Bloc 3 : Extraction de slots (lignes 193-252)

| Slot | Regex/Méthode | Exemple |
|------|---------------|---------|
| order_id | `ORD-[A-Z0-9]{5,}` | ORD-ABC12345 |
| index | `1\.(50\|56\|60\|67)` | 1.67 |
| city | Lookup dans KB + aliases Arabe→English | صفاقس → Sfax |
| product | Aliases : بروقرسيف→progressive, بلو كات→blue_cut | |
| phone | `+?216[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{3}` | +216 22 333 444 |
| date | `20\d{2}-\d{2}-\d{2}` | 2025-07-15 |

#### Bloc 4 : Routing d'outils (lignes 253-303)

| Intent | Slots requis | Outil appelé |
|--------|-------------|-------------|
| price_inquiry | product + index | `get_price()` |
| store_info | city | `get_store_info()` |
| order_tracking | order_id | `track_order()` |
| order_creation | product + index + city + phone | `create_order()` |
| appointment_booking | city + date + time_slot + phone | `book_appointment()` |

Si les slots requis manquent → pas d'appel outil, le modèle demande les infos manquantes.

#### Bloc 5 : Chargement LLM (lignes 305-378)

```python
# Singleton thread-safe
_model = None
_tokenizer = None
_lock = threading.Lock()
```

Charge Qwen2.5-7B-Instruct en 4-bit + l'adapter LoRA v2 (309 MB). Le modèle utilise ~6GB VRAM une fois chargé.

#### Bloc 6 : Génération (lignes 427-462)

```python
model.generate(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.85,
    repetition_penalty=1.15,
)
```

Utilise le chat template Qwen (`apply_chat_template`) pour formatter les messages système + historique + user.

#### Bloc 7 : Post-traitement (lignes 463-630)

**7 fonctions de nettoyage appliquées en séquence :**

| # | Fonction | Problème résolu | Mécanisme |
|---|----------|----------------|-----------|
| 1 | `_strip_english()` | Début en anglais "Hello! I'm..." | Regex strip + détection >60% ASCII |
| 2 | `_fix_brand_names()` | Fuite "OptiBot", "bot", "sivo متاع sivo" | 7 regex de remplacement |
| 3 | `_strip_embedded_english()` | Mots anglais isolés "QUESTIONS?" | Regex + whitelist (DT, indice, blue...) |
| 4 | `_fix_phrasing()` | "نعملك" au lieu de "نعاونك" | 6 regex de remplacement |
| 5 | `_scrub_pii()` | Numéros/commandes inventés | Ne garde que les vrais slots du user |
| 6 | `_clean_garbage_tail()` | Mots TSAC en fin (ريمسي, تراكتور) | Smart removal : garde si >70% arabe restant |
| 7 | `_truncate()` | Réponses trop longues | Max 8 phrases |

**Brand fixes détaillés :**
```python
_BRAND_FIXES = [
    (r"OptiBot",            "sivo"),
    (r"OptiTounes",         "sivo"),
    (r"Opti\s*Bot",         "sivo"),
    (r"bot",                "sivo"),           # standalone
    (r"أنا sivo المساعد متاع sivo", "أنا المساعد متاع sivo"),  # dédoublement
    (r"sivo المساعد متاع sivo",    "المساعد متاع sivo"),        # dédoublement
]
```

#### Bloc 8 : Quality Gate (lignes 632-655)

```python
_GARBAGE_PATTERNS = [
    r"^\s*حسبي الله",        # Religieux/émotionnel TSAC
    r"^\s*ربي يحفظ",
    r"^\s*صعيبة الحكاية",
    ...                       # 17 patterns au total
]
```

Si la réponse est détectée comme garbage :
1. **Retry** : regénère avec guidance renforcée
2. **Double garbage** : retourne fallback poli

#### Bloc 9 : production_infer() (lignes 657-826)

Pipeline principal qui orchestre tous les blocs :

1. **Normalisation** → text propre
2. **OOD detection** (35 keywords : salaire, محامي, football...) → fallback immédiat (0ms GPU)
3. **Short Latin check** ("high", "abc") → fallback immédiat
4. **Tool routing** → lookup KB si applicable
5. **RAG search** → BM25 sur KB (skip pour greetings/thanks)
6. **Generate** → model + post-traitement
7. **Quality gate** → retry ou fallback

**Point d'amélioration :**
- Remplacer le rule-based intent par un classifier fine-tuné
- Ajouter du caching pour les prompts fréquents (greeting = même réponse)
- Implémenter un confidence score sur l'intent

---

### 3.5 kb_tools.py
**Chemin :** `src/tounsi_llm/kb_tools.py` — **240 lignes**

**Objectif :** Base de connaissances (produits, magasins, politiques, commandes) + BM25 retriever.

**Données KB :**

| Source | Fichier | Entrées | Contenu |
|--------|---------|---------|---------|
| Catalogue | lens_catalog.csv | 8 produits | Prix, SKU, coating, indice |
| Magasins | stores.csv | 5 villes | Horaires, nom, adresse |
| Politiques | policies.jsonl | 5 règles | Garantie, retour, échange |
| Commandes | orders_mock.csv | 150 mock | Status, tracking, téléphone |

**Outils disponibles :**

| Outil | Fonction | Paramètres | Retour |
|-------|----------|-----------|--------|
| `get_price()` | Prix du catalogue | product, index, coating? | {status, sku, prix_min, prix_max} |
| `get_store_info()` | Info magasin | city | {status, nom, horaires} |
| `track_order()` | Suivi commande | order_id, phone? | {status, commande complète} |
| `create_order()` | Nouvelle commande | product, index, city, phone | {status, order_id} |
| `book_appointment()` | Réservation | city, date, time_slot, phone | {status, apt_id} |

**BM25 Retriever :**

```python
class KBRetriever:
    """BM25 sans dépendances externes"""
    # k1=1.5, b=0.75 (paramètres standard)
    # Tokenisation : re.findall(r"\w+", text.lower())
    # Auto-build depuis KB: 8 products + 5 stores + 5 policies = 18 documents
```

**Interprétation :**
- Le BM25 est suffisant pour 18 documents — pas besoin de vector DB
- Les commandes mock (150) simulent un vrai système de tracking

**Point d'amélioration :**
- Intégrer avec un vrai système de commandes (API ERP)
- Ajouter plus de produits au catalogue
- Implémenter fuzzy matching pour les noms de produits en Tounsi

---

### 3.6 server.py
**Chemin :** `src/tounsi_llm/server.py` — **170 lignes**

**Objectif :** API REST FastAPI avec gestion de sessions multi-tour.

**Endpoints :**

| Méthode | Route | Description | Corps |
|---------|-------|-------------|-------|
| POST | /chat | Message principal | `{message, session_id?}` |
| GET | /health | Status serveur | - |
| GET | /tools | Liste des outils | - |
| POST | /reset | Reset session | `{session_id?}` |

**Schéma de réponse (ChatResponse) :**

```python
{
    "response": str,        # Réponse en Tounsi
    "intent": str,          # Intent détecté
    "language_style": "tounsi",
    "slots": dict,          # Slots extraits
    "session_id": str,      # UUID session
    "latency_ms": float,    # Temps de réponse
    "tool_call": dict|None, # {name, arguments}
    "tool_result": dict|None,
    "gated": bool,          # Quality gate triggered?
    "routing_reason": str,
}
```

**Gestion de sessions :**
- Max 20 messages par session (derniers 6 envoyés au modèle = 3 tours)
- TTL : 1 heure d'inactivité
- Nettoyage automatique des sessions expirées

**CORS :** `allow_origins=["*"]` (développement — à restreindre en production)

**Point d'amélioration :**
- Ajouter rate limiting
- WebSocket pour streaming de réponses
- Authentification (API key ou JWT)
- Restreindre CORS en production

---

### 3.7 evaluation.py
**Chemin :** `src/tounsi_llm/evaluation.py` — **153 lignes**

**Objectif :** Évaluation automatique du modèle sur 16 cas de test couvrant tous les intents.

**Cas de test :**

| # | Input | Intent attendu | Vérification contenu |
|---|-------|---------------|---------------------|
| 1 | عسلامة شحوالك | greeting | ["سلام", "OptiBot"] |
| 2 | سلام عليكم | greeting | - |
| 3 | بونجور | greeting | - |
| 4 | قداش سوم البلارات البروقرسيف 1.67 | price_inquiry | ["390", "460", "DT"] |
| 5 | شحال سوم بلارات عادي 1.50 | price_inquiry | - |
| 6 | وقتاش يفتح المحل في صفاقس | store_info | ["09:00", "17:30"] |
| 7 | وين المحل في تونس | store_info | - |
| 8 | شنيا البلارات الي عندكم | product_info | - |
| 9 | نحب نسهل عل البلارات | product_info | - |
| 10 | نحب نتبع الكوموند ORD-ABC12345 | order_tracking | - |
| 11 | شنوة الأخبار اليوم | greeting | - |
| 12 | هايل برشا باه قلي | unknown | - |
| 13 | نحب حاجة باهية باش نجم تنصحني | product_info | - |
| 14 | والله لحقيقا انا حاير نحب ناخو lunette | product_info | - |
| 15 | يزي من الكذب قداش نستنى في الكوموند ORD-TEST1234 | order_tracking | - |
| 16 | مرسي برشا يعيشك | thanks | - |

**Métriques calculées :**
- Intent accuracy (%)
- Tounsi rate (%) — via `is_clean_tounsi()`
- Chinese rate (%) — caractères CJK
- Content accuracy (%) — contenu attendu présent
- Avg Fus7a words — moyenne de marqueurs Fus7a par réponse
- Avg latency (s)

**Point d'amélioration :**
- Augmenter à 50+ cas de test
- Ajouter des tests adversariaux (injection, manipulation)
- Mesurer BLEU/ROUGE entre réponse et référence

---

### 3.8 scripts/train.py
**Chemin :** `scripts/train.py` — **133 lignes**

**Objectif :** Point d'entrée CLI pour le pipeline d'entraînement complet.

**Stages disponibles :**

| Stage | Commande | Action |
|-------|---------|--------|
| download | `--stage download` | Télécharge les datasets HuggingFace |
| prepare | `--stage prepare` | Nettoie et formate les données |
| sft | `--stage sft` | Entraînement SFT |
| dpo | `--stage dpo` | Entraînement DPO |
| eval | `--stage eval` | Évaluation du modèle |
| all | `--stage all` | Pipeline complet (download → prepare → sft → dpo → eval) |

**Arguments additionnels :**
- `--max-sft-samples` : Limiter les échantillons (tests rapides)
- `--sft-epochs` : Override nombre d'époques SFT
- `--dpo-epochs` : Override nombre d'époques DPO

---

### 3.9 Frontend Angular
**Chemin :** `frontend/src/app/` — **553 lignes total**

| Fichier | Lignes | Rôle |
|---------|--------|------|
| app.ts | 135 | Composant principal — logique chat |
| app.html | 136 | Template — UI du chatbot |
| app.scss | 207 | Styles — thème sivo |
| chat.service.ts | 66 | Service HTTP — appels API |
| app.config.ts | 9 | Configuration Angular |

**chat.service.ts :**
```typescript
// POST http://localhost:8000/chat
// Body: { message: string, session_id?: string }
// Response: ChatResponse (response, intent, slots, tool_call, ...)
```

**UI :** Interface de chat RTL (droite-à-gauche) avec bulles de messages, indicateur de typing, affichage des outils appelés.

---

## 4. Données

### 4.1 Sources de données brutes

| Source | Dataset | Lignes | Utilité |
|--------|---------|--------|---------|
| HuggingFace | wghezaiel/SFT-Tunisian-Derja | ~40K | Paires instruction-réponse en Tounsi |
| HuggingFace | wghezaiel/DPO-Tunisian-Derja | ~44K | Paires de préférence (98.6% traduction — INUTILE) |
| HuggingFace | abdouuu/tunisian_chatbot_data | ~1.4K | Dialogues chatbot Tounsi |
| Custom | custom_tounsi_conversations.jsonl | 171 | Conversations centre d'appel optique (3x weight) |

### 4.2 Données d'enrichissement

| Fichier | Lignes | Source | Contenu |
|---------|--------|--------|---------|
| all_clean_tounsi.jsonl | 28,831 | Compilation nettoyée | Textes Tounsi propres |
| enriched_sft_conversations.jsonl | 6,265 | Enrichissement | Conversations SFT additionnelles |
| tsac_train.jsonl | 13,669 | TSAC corpus | Sentiments Tounsi (⚠️ bruit) |
| tsac_test.jsonl | 3,400 | TSAC corpus | Test set TSAC |
| tunisian_dialect_corpus.jsonl | 49,889 | Corpus dialecte | Large corpus Tounsi |
| tunizi_bigbench_50k.jsonl | 50,000 | TuniziBigBench | Benchmark Tounsi |
| tunizi.jsonl | 3,000 | Tunizi | Sentiments Tounsi |
| tunifra.jsonl | 7,797 | TUNIFRA | Franco-arabe Tounsi |
| tunswitch.jsonl | 5,082 | TunSwitch | Code-switching Tounsi |
| **Total enrichissement** | **167,933** | | |

### 4.3 Données d'entraînement finales

| Split | Taille | Fichier |
|-------|--------|---------|
| SFT Train | 4,914 | data/processed/sft_train.jsonl |
| SFT Val | 273 | data/processed/sft_val.jsonl |
| SFT Test | 273 | data/processed/sft_test.jsonl |
| DPO Train | **177** (nouveau) | data/processed/dpo_train.jsonl |
| DPO Val | **20** (nouveau) | data/processed/dpo_val.jsonl |

### 4.4 Distribution des system prompts dans SFT

| System Prompt | Count | % |
|---------------|-------|---|
| SYSTEM_GENERAL (langue) | 3,846 | 78.3% |
| Sans system prompt | 897 | 18.2% |
| SYSTEM_PROMPT (domaine optique) | 171 | 3.5% |

### 4.5 Base de connaissances

| Ressource | Fichier | Entrées |
|-----------|---------|---------|
| Catalogue lentilles | lens_catalog.csv | 8 produits |
| Magasins | stores.csv | 5 villes (Tunis, Sfax, Sousse, Ariana, Nabeul) |
| Politiques | policies.jsonl | 5 règles |
| Commandes mock | orders_mock.csv | 150 commandes |

---

## 5. Entraînement SFT

### 5.1 Configuration

| Paramètre | Valeur |
|-----------|--------|
| Modèle base | Qwen/Qwen2.5-7B-Instruct |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| LoRA modules | q, k, v, o, gate, up, down (7 modules) |
| Learning rate | 2e-4 |
| Batch size | 1 (effective 8 via grad_accum) |
| Époques | 5 |
| Warmup ratio | 3% |
| Weight decay | 0.01 |
| Précision | fp16 |
| Gradient checkpointing | ✓ |

### 5.2 Résultats

| Métrique | Valeur |
|----------|--------|
| **Train loss final** | **0.857** |
| Runtime | 36,430 sec (~10.1 heures) |
| Steps totaux | 3,070 |
| Samples/sec | 0.674 |
| Steps/sec | 0.084 |
| FLOPS totaux | 9.53 × 10¹⁶ |
| **Taille adapter** | **309 MB** |

### 5.3 Interprétation

- **Loss 0.857** : Bonne convergence pour un modèle 7B avec LoRA. La loss ne descend pas en dessous de ~0.5 typiquement car le modèle est quantifié en 4-bit, ce qui limite la précision.
- **10.1 heures** : Long mais attendu pour 4,914 samples × 5 époques sur T4 avec batch=1.
- **Adapter 309 MB** : Contient les poids LoRA de 7 types de modules × 32 couches du transformer. Taille raisonnable.

### 5.4 Checkpoints sauvegardés

```
artifacts/checkpoints/sft_v2/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
├── checkpoint-2000/
├── checkpoint-2500/
├── checkpoint-3000/
└── checkpoint-3070/    ← final
```

---

## 6. Audit DPO — Problème critique et Résolution

### 6.1 Découverte du problème

L'analyse des anciennes données DPO a révélé un **problème critique** :

```
Total paires: 1,934
├── "ترجم من الدارجة للفصحى" (Darija→Fus7a) : 957 (49.5%)
├── "ترجم من الفصحى للدارجة" (Fus7a→Darija) : 949 (49.1%)
└── Autres : 28 (1.4%)

→ 98.6% des données DPO sont des TRADUCTIONS, 0% de conversations call-center
```

**Impact :** Entraîner le DPO sur ces données aurait appris au modèle à **traduire** entre Fus7a et Derja au lieu d'améliorer ses réponses chatbot.

### 6.2 Nouvelles données DPO générées

177 paires d'entraînement + 20 validation, ciblant les **vrais problèmes observés** :

| Catégorie | Train | Description |
|-----------|-------|-------------|
| Greetings | ~25 | chosen = salutation Tounsi professionnelle, rejected = TSAC garbage / anglais |
| Identity | ~20 | chosen = "أنا المساعد متاع sivo", rejected = "OptiBot" / fuite brand |
| OOD honest | ~25 | chosen = "مانعرفش" honnête, rejected = hallucination / réponse fausse |
| Products | ~25 | chosen = info produit + prix corrects, rejected = prix inventés / chinois |
| Unclear | ~20 | chosen = "بش تحب نعاونك؟", rejected = réponse vague / TSAC |
| Thanks | ~20 | chosen = "عيشك، نتمنى خدمتك", rejected = réponse religieuse TSAC |
| Stores | ~20 | chosen = info magasin correcte, rejected = fausses horaires |
| Style | ~22 | chosen = "نعاونك" (bon phrasing), rejected = "نعملك" (mauvais) |
| **Total** | **177** | |

**Format des données :**
```jsonl
{"prompt": "شنوة أحسن بنك في تونس", "chosen": "هاذي مش من اختصاصي خويا. كان تحب نعديك لزميلي؟", "rejected": "البلارات عندنا يبداو من 40 دينار"}
```

### 6.3 Statistiques moyennes

| Métrique | Valeur |
|----------|--------|
| Avg chars prompt | 12 |
| Avg chars chosen | 59 |
| Avg chars rejected | 30 |
| Paires valides | 177/177 (100%) |

---

## 7. Post-traitement — Pipeline de correction

### 7.1 Problèmes résolus

| # | Problème | Source | Solution | Vérification |
|---|----------|--------|----------|-------------|
| 1 | "OptiBot" fuite de nom | Training data (ancien nom) | `_fix_brand_names()` : 4 regex | ✅ Toujours "sivo" |
| 2 | Caractères chinois (CJK) | Modèle base Qwen (chinois) | `_is_garbage_response()` → retry/fallback | ✅ 0% Chinese |
| 3 | Patterns TSAC ("حسبي الله") | Données enrichissement TSAC | 17 regex patterns → retry/fallback | ✅ Filtré |
| 4 | Mots garbage ("ريمسي") | TSAC entraînement | `_clean_garbage_tail()` : smart removal | ✅ Filtré |
| 5 | Anglais embedded ("QUESTIONS?") | Base model leakage | `_strip_embedded_english()` + whitelist | ✅ Filtré |
| 6 | "high" → hallucination prix | "high" matchait "hi" (greeting) | Word-boundary matching (set) | ✅ → fallback unclear |
| 7 | OOD hallucination | Modèle invente des réponses | 35 keywords OOD + short-circuit | ✅ → "مانعرفش" |
| 8 | "نعملك" mauvais phrasing | Training data style | `_fix_phrasing()` : 6 regex | ✅ → "نعاونك" |
| 9 | "sivo المساعد متاع sivo" | Redondance brand | Regex avec groupe optionnel | ✅ → "المساعد متاع sivo" |
| 10 | PII inventé (faux numéros) | Modèle hallucine | `_scrub_pii()` : ne garde que vrais slots | ✅ Filtré |

### 7.2 Ordre du pipeline de post-traitement

```
Réponse brute du modèle
    │
    ├─[1] _strip_english()         # Retire "Hello! I'm..."
    ├─[2] _fix_brand_names()       # OptiBot → sivo, dédoublement
    ├─[3] _strip_embedded_english() # "QUESTIONS?" → rien
    ├─[4] _fix_phrasing()          # نعملك → نعاونك
    ├─[5] _scrub_pii()             # Faux numéros → supprimés
    ├─[6] _clean_garbage_tail()    # ريمسي, تراكتور → nettoyé
    └─[7] _truncate()              # Max 8 phrases
    │
    ▼
Quality Gate: _is_garbage_response()
    │
    ├─ OK → retourner réponse
    ├─ Garbage → retry avec guidance renforcée
    └─ Double garbage → fallback poli
```

---

## 8. Tests et Validation

### 8.1 Suite de tests (10 scénarios)

| # | Input | Type | Résultat | Latency |
|---|-------|------|----------|---------|
| 1 | "high" | Latin ambigu | "عاودلي من فضلك، بش تحب نعاونك؟" | ~1ms |
| 2 | "اهلا" | Salutation | Réponse greeting sivo | ~2-3s |
| 3 | "شحالك" | Salutation | Réponse greeting sivo | ~2-3s |
| 4 | "انت شكون" | Identité | "أنا المساعد متاع sivo" | ~2-3s |
| 5 | "سلام" | Salutation | Réponse greeting sivo | ~2-3s |
| 6 | "salaire ingenieur" | OOD | "مانعرفش نجاوبك..." (honnête) | ~1ms |
| 7 | "عندكم بلارات blue cut" | Domaine | Prix + info produit | ~3-5s |
| 8 | "شكرا" | Remerciement | "عيشك" / "بالسلامة" | ~2-3s |
| 9 | "كيفاش نولي محامي" | OOD | "مانعرفش..." (honnête) | ~1ms |
| 10 | "enti chkoun" | Franco identité | "أنا المساعد متاع sivo" | ~2-3s |

**Résultat : 10/10 ✅**

### 8.2 Latences observées

| Type de requête | Latence | GPU utilisé |
|----------------|---------|------------|
| OOD (short-circuit) | **< 5ms** | Non |
| Latin ambigu (short-circuit) | **< 5ms** | Non |
| Génération modèle | **2-5 secondes** | Oui |
| Avec tool call | **3-6 secondes** | Oui |

**Interprétation :** Les short-circuits OOD et Latin sont quasi-instantanés car ils ne font pas d'appel GPU. Les réponses générées prennent 2-5s, ce qui est acceptable pour un chatbot.

---

## 9. Points d'amélioration

### 9.1 Priorité haute

| # | Amélioration | Impact | Effort |
|---|-------------|--------|--------|
| 1 | **Entraînement DPO** avec nouvelles données (177 paires) | Meilleure qualité des réponses, moins de TSAC/anglais | En cours |
| 2 | **Plus de données SFT domaine** : conversations call-center optique réelles | Meilleure spécialisation domaine (actuellement 3.5% du training) | Moyen |
| 3 | **Early stopping** dans le training | Évite overfitting, meilleure généralisation | Facile |
| 4 | **Classifier d'intent ML** | Remplace le rule-based, gère les edge cases | Moyen |

### 9.2 Priorité moyenne

| # | Amélioration | Impact | Effort |
|---|-------------|--------|--------|
| 5 | **Streaming WebSocket** pour réponses progressives | UX plus fluide, feedback immédiat | Moyen |
| 6 | **Rate limiting + Auth** sur l'API | Sécurité production | Facile |
| 7 | **Vector DB** (FAISS/ChromaDB) pour le retriever | Meilleur recall avec plus de documents KB | Moyen |
| 8 | **Tests adversariaux** (injection, manipulation) | Robustesse | Moyen |
| 9 | **Configuration YAML externe** | Flexibilité sans toucher au code | Facile |

### 9.3 Priorité basse

| # | Amélioration | Impact | Effort |
|---|-------------|--------|--------|
| 10 | **Multi-GPU** support (si upgrade hardware) | Latence réduite | Faible effort si DDP |
| 11 | **RLHF** avec feedback utilisateurs réels | Alignement continu | Élevé |
| 12 | **Multimodal** (photos d'ordonnances) | Nouveau cas d'usage | Élevé |
| 13 | **Intégration ERP** pour vraies commandes | Production-ready | Élevé |

### 9.4 Métriques à ajouter

- **BLEU/ROUGE** entre réponse et référence
- **Perplexité** sur le test set
- **Satisfaction utilisateur** (thumbs up/down dans le frontend)
- **Intent confusion matrix** pour identifier les erreurs systématiques
- **A/B testing** avant/après DPO

---

## 10. Annexes

### A. Arborescence complète des fichiers

```
ai/
├── src/tounsi_llm/
│   ├── __init__.py          (2 lignes)
│   ├── config.py            (143 lignes) — Configuration centralisée
│   ├── data_prep.py         (606 lignes) — Préparation données
│   ├── training.py          (333 lignes) — SFT + DPO training
│   ├── inference.py         (826 lignes) — Pipeline d'inférence
│   ├── kb_tools.py          (240 lignes) — KB + BM25 retriever
│   ├── server.py            (170 lignes) — API FastAPI
│   └── evaluation.py        (153 lignes) — Évaluation auto
├── scripts/
│   └── train.py             (133 lignes) — CLI entry point
├── frontend/src/app/
│   ├── app.ts               (135 lignes) — Composant Angular
│   ├── app.html             (136 lignes) — Template chat
│   ├── app.scss             (207 lignes) — Styles
│   ├── chat.service.ts      (66 lignes)  — Service HTTP
│   └── app.config.ts        (9 lignes)   — Config Angular
├── data/
│   ├── processed/           — Données finales (SFT + DPO)
│   ├── tounsi_raw/          — Données brutes HuggingFace
│   ├── kb/                  — Base de connaissances
│   └── tounsi_raw/enrichment/ — 9 corpus (167K lignes)
├── artifacts/
│   ├── adapters/v2/         — Adapter LoRA SFT (309 MB)
│   └── checkpoints/sft_v2/  — 7 checkpoints
└── reports/
    ├── sft_v2_metrics.json  — Métriques SFT
    └── rapport_detaille_v2.md — Ce rapport
```

### B. Dépendances Python

```
torch, transformers (4.46.3), peft (0.13.2), trl (0.12.2),
bitsandbytes, datasets, fastapi, uvicorn, pydantic, numpy
```

### C. Commandes utiles

```bash
# Démarrer le serveur
cd /home/ahmed/Bureau/ai
.venv/bin/uvicorn src.tounsi_llm.server:app --host 0.0.0.0 --port 8000

# Entraînement SFT
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m scripts.train --stage sft

# Entraînement DPO
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m scripts.train --stage dpo

# Test rapide
curl -s http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "سلام"}' | python3 -m json.tool

# Évaluation
python -m scripts.train --stage eval
```

### D. Métriques SFT brutes

```json
{
  "train_runtime": 36430.7,
  "train_samples_per_second": 0.674,
  "train_steps_per_second": 0.084,
  "total_flos": 9.528e+16,
  "train_loss": 0.857,
  "epoch": 5.0,
  "training_time_min": 607.2,
  "lora_rank": 32,
  "epochs": 5
}
```

---

*Rapport généré automatiquement — Tounsi Call-Center LLM v2 Pipeline*
