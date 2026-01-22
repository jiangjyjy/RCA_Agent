# RCA_Agent

A two-stage pipeline for generating and evaluating multi-user CST dialogue responses. Stage 1 creates model outputs on synthetic dialogues; Stage 2 scores them with LLM judges and BERTScore; `stress_test.py` runs a manual case study.

## Requirements
- Python 3.10+
- pip packages: `openai`, `python-dotenv`, `tqdm`, `bert-score` 
- Local model: place `local_models/bert-base-chinese/` to speed up BERTScore; otherwise the HuggingFace name `bert-base-chinese` is used.

## Configure API Access
1. Supply API keys for all models you plan to call.
2. Edit [experiment_suite/backends.py](experiment_suite/backends.py) `MODEL_CONFIG` to fill `api_key`, `base_url`, and `model` for each backbone you will use (e.g., `deepseek-v3.2`, `gpt-4o`, `glm-4.7`, `kimi`, `gpt-5`, `gemini-3`).
3. Edit [rca_framework/config.py](rca_framework/config.py) to set `API_KEY`, `BASE_URL`, and `MODEL_NAME` for the reward/critic model. These are separate from the generation backbones.
4. Concurrency limits: `MAX_CONCURRENT_REQUESTS` in [rca_framework/config.py](rca_framework/config.py) and `API_LIMIT`/`WORKER_LIMIT` in [generation.py](generation.py) guard API rate limits.

## Data Prep
- Place synthetic dialogue JSON files under `synthetic_data/`. Each file should contain `synthetic_id`, `virtual_participants`, and `dialogue` fields. The generator will sample turns automatically.

## How to Run
1. **Stage 1 — Generate candidate replies**
   - Adjust `TARGET_BACKBONE`, `NUM_SAMPLES`, `SEED`, and API limits at the top of [generation.py](generation.py) as needed.
   - Run: `python generation.py`
   - Outputs: `stage1_generation.jsonl` and `viz_data.jsonl`.

2. **Stage 2 — Score with LLM judges & BERTScore**
   - Ensure `stage1_generation.jsonl` exists (from Stage 1) or is pre-generated.
   - Run: `python evaluation.py`
   - Outputs: `experiment_results.jsonl` with scores (`DS_*`, `GLM_*`, `BERTScore`).

3. **Stress test / case study**
   - Edit `TEST_CASE` in [stress_test.py](stress_test.py) if desired.
   - Run: `python stress_test.py`
   - Compares multiple backbones plus the RCA agent on a single crafted scenario and prints LaTeX-ready rows.

## File Map (key components)
- [generation.py](generation.py): Stage 1 orchestration; calls baseline agents and `RcaAgent`; writes generation log.
- [evaluation.py](evaluation.py): Stage 2 scoring; uses `LLMJudge` and BERTScore; writes experiment results.
- [stress_test.py](stress_test.py): One-off real-case comparison across models and RCA.
- [experiment_suite/baselines.py](experiment_suite/baselines.py): Zero-shot, few-shot, and CoT baselines.
- [experiment_suite/judges.py](experiment_suite/judges.py): LLM-based evaluators.
- [experiment_suite/metrics.py](experiment_suite/metrics.py): BERTScore/Distinct/BLEU/ROUGE helpers.
- [rca_framework/agent.py](rca_framework/agent.py): RCA agent with planning → multi-candidate generation → reward reranking.
- [rca_framework/reward_model.py](rca_framework/reward_model.py): Heuristic reward via critic LLM.
- [rca_framework/config.py](rca_framework/config.py): System prompts, weights, and API concurrency settings.

## Notes & Tips
- Missing dependencies: if `bert-score` is not installed, BERTScore defaults to 0.0; install it for proper metrics.
- Logs are append-only; delete `stage1_generation.jsonl` / `experiment_results.jsonl` if you need a clean rerun.
- Ensure rate limits are consistent with your provider; adjust semaphores accordingly to avoid throttling.
