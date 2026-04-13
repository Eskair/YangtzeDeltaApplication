# YangtzeDelta Proposal Analyser — Architecture Review & Upgrade Plan

## 1. Pipeline Architecture

### 1.1 Full Pipeline (8 stages)

```
Upload (PDF/DOCX/PPTX/TXT)
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: prepare_proposal_text                                  │
│   PDF/DOCX/PPTX → full_text.txt + pages.json                   │
│   (OCR via Tesseract, optional vision LLM for figures)          │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: extract_facts_by_chunk                                 │
│   full_text.txt → character-level chunks (1800 chars, 400 lap)  │
│   → LLM fact extraction per chunk → raw_facts.jsonl             │
│   Each fact: {text, dimensions[], type, primary_dimension, meta} │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: build_dimensions_from_facts                            │
│   raw_facts.jsonl → group by 5 dimensions                       │
│   → 1 LLM call per dimension → dimensions_v2.json              │
│   Each dimension: {summary, key_points[], risks[], mitigations[]}│
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: generate_questions                                     │
│   dimensions_v2.json → 1 LLM call per dimension                │
│   → 6–10 questions per dimension with aspect/priority/links_to  │
│   → generated_questions.json (simplified) + detail version      │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: llm_answering                                          │
│   questions + dimension context → OpenAI + optional DeepSeek    │
│   3 variants per question (default/risk/implementation)         │
│   → batch or single-question mode → refine step                 │
│   → merged all_refined_items.json                               │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 6: post_processing  (NO LLM calls — pure heuristics)      │
│   Score each candidate answer on: length, claims, structure,    │
│   alignment, confidence, consistency, penalties                  │
│   → Select best candidate per question                          │
│   → final_payload.json + metrics.json                           │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 7: ai_expert_opinion                                      │
│   metrics.json + final_payload.json → LLM expert commentary     │
│   per dimension (summary/strengths/concerns/recommendations)    │
│   → local verdict logic (GO / HOLD / NO-GO)                     │
│   → ai_expert_opinion.json + .md                                │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 8: generate_final_report                                  │
│   Assembles: Executive Summary + Expert MD + QA Section          │
│   → {pid}_final_report.md                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow Summary

```
PDF ─→ full_text.txt ─→ raw_facts.jsonl ─→ dimensions_v2.json
                                                    │
                                    ┌───────────────┘
                                    ▼
                        generated_questions.json
                                    │
                                    ▼
                        all_refined_items.json  (multi-model, multi-variant)
                                    │
                                    ▼
                        final_payload.json  +  metrics.json
                                    │
                                    ▼
                        ai_expert_opinion.json/.md
                                    │
                                    ▼
                        {pid}_final_report.md
```

### 1.3 Serving Layer

- **FastAPI** server (`src/api/server.py`) orchestrates the pipeline as subprocess calls
- **SSE** (Server-Sent Events) for real-time progress updates to the frontend
- **Single-page HTML** frontend with upload → progress → rendered Markdown report
- **Email delivery** (Resend API or SMTP) for completed reports

---

## 2. System Quality Evaluation

### 2.1 Strengths

| Area | Assessment |
|------|-----------|
| **Anti-hallucination design** | The system shows strong awareness of LLM hallucination risks. Every major prompt includes explicit hard constraints: "do not fabricate institutions, numbers, drugs, etc." The `mark_numeric_suspect` function cross-checks extracted numbers against source chunks. The prompt system consistently requires grounding in `[提案事实]`. |
| **Multi-variant answering** | Stage 5 generates 3 variants per question (default, risk, implementation perspectives), creating a "panel of reviewers" effect that improves coverage and enables downstream selection. |
| **Dual-model architecture** | Using both OpenAI and DeepSeek reduces single-model bias. The post-processing stage includes provider rotation to avoid systematically favoring one model. |
| **Structured intermediate representations** | Each stage produces well-defined JSON artifacts that are both machine-readable and human-inspectable. The `raw_facts.jsonl` → `dimensions_v2.json` → `generated_questions.json` chain creates clear audit trails. |
| **Self-refinement loop** | The `refine_candidate` step in Stage 5 asks the LLM to self-review its output against the source facts, catching some overstatements. |
| **Progressive filtering** | The post-processing pipeline applies multiple quality gates (structure, alignment, overclaim detection, dimension drift, redline detection for fabricated citations) before selecting the best candidate. |
| **Graceful degradation** | The expert opinion stage has a full local-rules fallback when the LLM is unavailable, ensuring the system always produces output. |
| **Score shielding** | The expert opinion prompt deliberately hides numeric scores from the LLM, providing only qualitative hints ("strong/medium/weak") to prevent score leakage from influencing the narrative. |

### 2.2 Weaknesses and Risks

#### A. LLM Hallucination Risks (Still Present)

1. **Fact extraction is one-shot per chunk with limited verification.** While `mark_numeric_suspect` flags numbers, there is no systematic verification that extracted text fragments actually exist in the source. A dedicated string-matching verification pass would significantly reduce phantom facts.

2. **Dimension summaries can drift from facts.** Stage 3 truncates facts to 10,000 characters before sending to the LLM. Important facts near the truncation boundary may be lost, and the LLM may fill gaps with plausible-sounding but unsourced content.

3. **Question generation can create leading questions.** Despite anti-hallucination constraints, the generated questions sometimes embed assumptions about what the proposal contains (e.g., "Based on the proposal's description of market size...") when the proposal may not contain such information.

4. **The `general_insights` field blurs the line between fact and opinion.** While clearly labeled as "industry knowledge," these insights are generated by the same LLM and may contain inaccurate domain claims that downstream stages treat as authoritative baselines.

#### B. Prompt Design Issues

1. **Domain lock-in.** All prompts are hardcoded for "生物医药 & AI" (biomedical & AI) projects. The 5 fixed dimensions (team, objectives, strategy, innovation, feasibility), the type taxonomy (`clinical_design`, `regulatory`, `ai_model`, etc.), and the question aspects are all biomedical-specific. This makes the system unusable for other research domains without extensive prompt rewriting.

2. **Prompt length and redundancy.** The `FACT_PROMPT` alone is ~190 lines. The `QUESTION_PROMPT_TEMPLATE` is ~130 lines. These prompts contain extensive Chinese instructions with many overlapping constraints. This increases token cost, makes iteration difficult, and may actually confuse the model by providing contradictory guidance at different points in the prompt.

3. **Language coupling.** The system prompt language is Chinese, but it requests both Chinese and English question outputs. This forces the LLM to context-switch languages, reducing quality in both. The system prompts, dimension labels, and output format should be decoupled from any specific language.

4. **No prompt versioning or A/B testing infrastructure.** Prompts are embedded as string constants in Python files. There is no mechanism to version, compare, or A/B test prompt variations systematically.

#### C. Scoring Reliability Issues

1. **Heuristic scoring masquerades as meaningful metrics.** The `post_processing.py` scoring system (1,600+ lines) computes elaborate scores using character length, bullet count, Jaccard overlap, and keyword matching. These proxy metrics have no validated correlation with actual review quality. A high score means the answer is long, well-formatted, and uses expected keywords — not that it is correct or insightful.

2. **Dimension weights are arbitrary.** Innovation gets 1.10, feasibility gets 1.20, others get 1.00. These weights have no empirical basis and silently influence the final verdict (GO/HOLD/NO-GO), which could have significant consequences for funding decisions.

3. **The verdict logic is simplistic.** Three hard-coded threshold checks (`score >= 0.62 → GO`, `score < 0.40 → NO-GO`, else `HOLD`) determine a consequential funding recommendation. These thresholds are not calibrated against human reviewer decisions.

4. **Scoring conflates form with substance.** The `looks_structured()` function gives points for bullet formatting. The `alignment` score rewards keyword overlap with pre-defined authority tokens (FDA, ISO, etc.). A well-formatted answer that mentions regulatory terms scores higher than a substantively better answer in plain prose.

5. **No inter-rater reliability measurement.** The system compares model outputs via Jaccard similarity as a proxy for "consistency," but this measures lexical overlap, not semantic agreement. Two answers could use completely different words to express the same conclusion, or use the same words to express different conclusions.

#### D. Architecture Issues

1. **Subprocess-based pipeline.** Each stage runs as a separate Python subprocess (`subprocess.Popen`). This means no shared state, no error recovery mid-pipeline, and significant overhead from repeated Python startup and module loading.

2. **No idempotency or caching.** If Stage 5 fails, rerunning the pipeline starts from Stage 1. There is no checkpointing — all LLM calls (and their costs) are repeated.

3. **Single-threaded per stage.** Dimension-level operations within each stage run sequentially. Stage 5 processes dimensions one at a time despite being embarrassingly parallel.

4. **No test suite.** Zero tests in the repository. The complex scoring logic in `post_processing.py` (1,600 lines of heuristic scoring) has no unit tests, making it impossible to verify correctness or detect regressions.

5. **Global state at import time.** Several modules (`extract_facts_by_chunk.py`, `build_dimensions_from_facts.py`, `web_search.py`) instantiate OpenAI clients as module-level globals, causing side effects on import.

6. **Dead code.** `src/main.py` imports non-existent `backend.chains.*` modules. The `search_by_dimension.py`, `fusion_search.py`, and `build_vector_db.py` modules are not wired into the pipeline. `model_selector.py` is unused by the main pipeline.

7. **In-memory job store.** Server restarts lose all job state. No persistence layer.

---

## 3. High-Impact Improvements

### 3.1 Architectural Improvements

#### 3.1.1 Replace subprocess pipeline with async task graph

```python
# Proposed: pipeline as an async DAG with checkpointing
class PipelineRunner:
    def __init__(self, pid: str, config: PipelineConfig):
        self.pid = pid
        self.config = config
        self.checkpoint = CheckpointStore(pid)

    async def run(self):
        text = await self.checkpoint.get_or_run(
            "prepare_text", prepare_proposal_text, self.pid
        )
        facts = await self.checkpoint.get_or_run(
            "extract_facts", extract_facts, self.pid, text
        )
        dimensions = await self.checkpoint.get_or_run(
            "build_dimensions", build_dimensions, self.pid, facts
        )
        questions = await self.checkpoint.get_or_run(
            "generate_questions", generate_questions, self.pid, dimensions
        )
        # Stage 5: parallel across dimensions
        answers = await asyncio.gather(*[
            self.checkpoint.get_or_run(
                f"answer_{dim}", answer_dimension, dim, questions[dim], dimensions[dim]
            )
            for dim in DIMENSIONS
        ])
        # ...remaining stages
```

**Benefits:** Caching avoids re-running expensive LLM calls. Parallelism across dimensions cuts Stage 5 time by ~5x. In-process execution eliminates subprocess overhead.

#### 3.1.2 Add a verification stage after fact extraction

```
Stage 1.5 (NEW): verify_facts
  For each extracted fact:
    1. Fuzzy-match key entities/numbers against source chunk
    2. Flag facts with <70% entity coverage as "unverified"
    3. Remove facts where key numeric claims can't be found in source
```

This is the single highest-ROI change for hallucination reduction.

#### 3.1.3 Persistent job store

Replace the in-memory `jobs` dict with SQLite or Redis. Even a simple JSON file per job would survive server restarts.

### 3.2 Prompt System Improvements (Domain-Agnostic)

#### 3.2.1 Externalize and parameterize prompts

Move all prompts to a `prompts/` directory as YAML/Jinja2 templates:

```yaml
# prompts/fact_extraction.yaml
system: |
  You are a rigorous fact extractor. Extract atomic facts from text chunks.
  {{constraints}}

user: |
  Domain: {{domain}}
  Dimensions: {{dimensions | join(', ')}}
  Fact types: {{fact_types | join(', ')}}

  === Text chunk ===
  {{chunk_text}}

constraints: |
  1. Only use information present in the current text chunk.
  2. Do not fabricate entities, numbers, or claims.
  3. Each fact must be traceable to the source text.
  ...
```

**Benefits:** Prompts become versionable, testable, and domain-switchable without code changes.

#### 3.2.2 Make dimensions and types configurable

```yaml
# config/domain_biomedical.yaml
dimensions:
  - id: team
    label: "Team & Governance"
    description: "Team composition, expertise, governance"
  - id: objectives
    label: "Project Objectives"
    description: "Goals, milestones, success metrics"
  # ...

fact_types:
  - id: team_member
    maps_to_dimensions: [team]
  - id: clinical_design
    maps_to_dimensions: [objectives, strategy, feasibility]
  # ...

# config/domain_engineering.yaml
dimensions:
  - id: team
    label: "Team & Capabilities"
  - id: technical_approach
    label: "Technical Approach"
  - id: market_fit
    label: "Market Fit & Impact"
  # ...
```

#### 3.2.3 Separate "instruction" from "content" in prompts

Current prompts mix meta-instructions (output format, constraints) with content-specific instructions (what to look for in biomedical proposals). Split these:

```
Base Prompt (reusable across domains):
  - Output format requirements
  - Anti-hallucination constraints
  - Grounding rules

Domain Overlay (swappable):
  - What dimensions to evaluate
  - What entity types to extract
  - What aspects to assess

Proposal Context (per-run):
  - Extracted facts/summaries
  - Generated questions
```

### 3.3 Scoring Improvements

#### 3.3.1 Replace heuristic scoring with calibrated LLM evaluation

The current 1,600-line scoring heuristic should be replaced with a lighter, more reliable approach:

```python
EVAL_PROMPT = """
Rate this answer on a 1-5 scale for each criterion.
You MUST provide a brief justification for each score.

Question: {question}
Source facts: {relevant_facts}
Answer to evaluate: {answer}

Criteria:
1. Factual grounding: Does every claim trace to the source facts?
2. Completeness: Does the answer address all parts of the question?
3. Insight quality: Does the answer provide non-obvious analysis?
4. Actionability: Are recommendations specific and implementable?

Output JSON: {"factual_grounding": {"score": N, "reason": "..."}, ...}
"""
```

Use a separate (ideally different) LLM as evaluator, or use the same model with a dedicated evaluation prompt. Calibrate against a small set of human-graded examples.

#### 3.3.2 Implement answer-fact traceability scoring

For each claim in an answer, compute a grounding score:

```python
def compute_grounding_score(answer_claims: list[str], source_facts: list[str]) -> float:
    grounded = 0
    for claim in answer_claims:
        best_match = max(
            semantic_similarity(claim, fact) for fact in source_facts
        )
        if best_match > THRESHOLD:
            grounded += 1
    return grounded / len(answer_claims) if answer_claims else 0.0
```

This directly measures hallucination rather than using proxies like keyword overlap.

#### 3.3.3 Calibrate verdict thresholds empirically

Current thresholds (0.62 for GO, 0.40 for NO-GO) are arbitrary. Build a calibration dataset:

1. Collect 50+ proposals with known human reviewer decisions
2. Run the pipeline on each
3. Fit thresholds to maximize agreement with human verdicts
4. Re-calibrate periodically

#### 3.3.4 Add confidence intervals to scores

Instead of a single point score, report a range:

```json
{
  "dimension": "team",
  "score": 0.68,
  "confidence_interval": [0.55, 0.78],
  "reliability": "moderate",
  "basis": "5 questions answered, 3 with strong fact grounding"
}
```

This communicates uncertainty honestly and prevents over-reliance on precise-looking numbers.

---

## 4. Upgraded Design

### 4.1 Proposed Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CONFIG LAYER                          │
│  domain.yaml  │  prompts/*.yaml  │  scoring_config.yaml │
└───────────────┴──────────────────┴──────────────────────┘
        │                │                  │
        ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                   CORE PIPELINE                         │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌───────────────────┐   │
│  │ Ingest   │──▶│ Extract  │──▶│ Verify & Cluster  │   │
│  │          │   │ Facts    │   │ Facts             │   │
│  │ PDF/DOCX │   │          │   │ (NEW: traceability│   │
│  │ → text   │   │ chunk →  │   │  check + dedup)   │   │
│  └──────────┘   │ LLM →   │   └────────┬──────────┘   │
│                 │ facts    │            │               │
│                 └──────────┘            ▼               │
│                              ┌──────────────────────┐   │
│                              │ Build Dimensions     │   │
│                              │ (configurable N dims)│   │
│                              └──────────┬───────────┘   │
│                                         │               │
│                    ┌────────────────────┘               │
│                    ▼                                     │
│           ┌────────────────┐                            │
│           │ Generate       │                            │
│           │ Questions      │                            │
│           │ (from config)  │                            │
│           └───────┬────────┘                            │
│                   │                                     │
│                   ▼                                     │
│  ┌────────────────────────────────────────────────┐     │
│  │         Answer & Evaluate  (parallel)          │     │
│  │                                                │     │
│  │  ┌─────────┐ ┌─────────┐      ┌─────────┐     │     │
│  │  │ Model A │ │ Model B │ ...  │ Model N │     │     │
│  │  │ answer  │ │ answer  │      │ answer  │     │     │
│  │  └────┬────┘ └────┬────┘      └────┬────┘     │     │
│  │       └─────┬─────┘                │          │     │
│  │             ▼                      │          │     │
│  │   ┌─────────────────┐             │          │     │
│  │   │  LLM Evaluator  │◀────────────┘          │     │
│  │   │  (grounding +   │                        │     │
│  │   │   quality check) │                        │     │
│  │   └────────┬────────┘                        │     │
│  │            │                                  │     │
│  │            ▼                                  │     │
│  │   Select best per question                    │     │
│  └────────────────────────────────────────────────┘     │
│                   │                                     │
│                   ▼                                     │
│  ┌─────────────────────────────────────────────┐        │
│  │       Expert Synthesis                      │        │
│  │  LLM narrative + local verdict logic        │        │
│  └──────────────────────┬──────────────────────┘        │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────┐        │
│  │       Report Assembly                       │        │
│  │  Executive summary + dimension details      │        │
│  │  + Q&A evidence trail                       │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   SERVING LAYER                         │
│  FastAPI + SSE  │  Job persistence  │  Email delivery   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Module Responsibilities (Upgraded)

| Module | Responsibility | Key Change |
|--------|---------------|------------|
| `config/` | Domain definitions, prompt templates, scoring weights | **NEW:** All domain knowledge externalized |
| `pipeline/runner.py` | Async pipeline orchestrator with checkpointing | **NEW:** Replaces subprocess chain |
| `pipeline/ingest.py` | Document → text extraction | Refactored from `prepare_proposal_text.py` |
| `pipeline/extract.py` | Chunked fact extraction | Simplified from `extract_facts_by_chunk.py` |
| `pipeline/verify.py` | Fact verification against source text | **NEW:** Hallucination reduction |
| `pipeline/dimensions.py` | Facts → dimension summaries | Refactored, config-driven dimensions |
| `pipeline/questions.py` | Dimension summaries → evaluation questions | Prompt-template driven |
| `pipeline/answer.py` | Multi-model Q&A with variants | Parallelized across dimensions |
| `pipeline/evaluate.py` | LLM-based answer evaluation + fact grounding | **NEW:** Replaces heuristic scoring |
| `pipeline/synthesize.py` | Expert narrative + verdict | Simplified from `ai_expert_opinion.py` |
| `pipeline/report.py` | Final Markdown assembly | Simplified from `generate_final_report.py` |
| `api/server.py` | FastAPI endpoints + job management | Add persistent job store |
| `api/models.py` | LLM client factory (OpenAI/DeepSeek/Gemini) | Refactored from scattered init code |

### 4.3 Key Design Principles for the Upgrade

1. **Configuration over code.** Domain knowledge (dimensions, fact types, question aspects, scoring weights) lives in YAML, not in Python string constants.

2. **Verify, don't trust.** Every LLM output passes through a verification step that checks claims against source material.

3. **Measure what matters.** Replace proxy metrics (keyword overlap, bullet count) with direct quality measurement (fact grounding, answer completeness).

4. **Honest uncertainty.** Report confidence intervals, flag information gaps explicitly, and make the basis for every conclusion traceable.

5. **Fail gracefully, resume cheaply.** Checkpoint every stage output. On failure, resume from the last successful stage.

6. **Test the untestable.** Unit-test scoring logic, integration-test the full pipeline against reference proposals, regression-test prompts by comparing outputs across versions.

### 4.4 Migration Priority

| Priority | Change | Impact | Effort |
|----------|--------|--------|--------|
| **P0** | Add fact verification stage (3.1.2) | Directly reduces hallucination | 1 new module (~200 lines) |
| **P0** | Externalize prompts to YAML (3.2.1) | Enables domain-agnostic use | Refactor prompts + add loader |
| **P1** | Replace heuristic scoring with LLM eval (3.3.1) | More reliable quality signal | Replace post_processing core |
| **P1** | Parallelize dimension processing (3.1.1) | 3-5x speedup for Stages 3-5 | Refactor to async |
| **P1** | Add checkpoint/caching (3.1.1) | Saves cost on reruns | Add persistence layer |
| **P2** | Configurable dimensions (3.2.2) | Domain flexibility | Config schema + refactor |
| **P2** | Calibrate verdict thresholds (3.3.3) | More reliable decisions | Requires calibration data |
| **P2** | Add test suite | Prevents regressions | Ongoing investment |
| **P3** | Persistent job store (3.1.3) | Production resilience | Add SQLite/Redis layer |
| **P3** | Remove dead code | Cleaner codebase | Audit + delete |

---

## 5. Summary

The YangtzeDelta system is a thoughtfully designed 8-stage pipeline that demonstrates strong awareness of LLM risks. Its multi-model, multi-variant architecture with progressive filtering is a solid foundation. The three highest-impact improvements are:

1. **Add fact verification** — a post-extraction stage that validates each fact against source text, directly attacking the hallucination problem at its root.

2. **Externalize all domain knowledge** — move dimensions, fact types, prompt templates, and scoring weights to configuration files, making the system usable across research domains.

3. **Replace proxy scoring with grounded evaluation** — swap the 1,600-line heuristic scorer for LLM-based evaluation that directly measures factual grounding and answer quality.

These three changes would transform the system from a domain-specific prototype into a production-grade, domain-agnostic AI research proposal reviewer.
