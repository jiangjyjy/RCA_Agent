# A Reflective Cognitive Alignment Framework for Agents in Cognitive Stimulation Care for the Elderly

Two complementary pieces live here:
- **STaR-CS**: end-to-end cleaning, de-identification, skeleton extraction, style analysis, and synthetic dialogue generation for Cognitive Stimulation Therapy (CST) group sessions.
- **RCA_Agent**: reflective cognitive alignment (RCA) agent, baselines, and automatic evaluation pipeline on the synthetic dialogues produced above.

Use this repo as a landing page to route into either workflow.

## Repository Layout
- STaR-CS/ — data cleaning → de-identification → skeleton extraction → style analysis → synthetic dialogue generation
- RCA_Agent/ — RCA agent, baseline generators, judges, metrics, scripts

## Prerequisites
- Python 3.10+
- An OpenAI-compatible API endpoint for all LLM calls (de-identification, skeleton extraction, style analysis, synthesis, RCA planning/generation, LLM judges)
- Recommended: local `bert-base-chinese` under `RCA_Agent/local_models/bert-base-chinese/` to speed up BERTScore; otherwise the HF hub model name is used

## Quick Start
### 1) STaR-CS: Build the Synthetic Dialogue Corpus
1. Install deps: `pip install -r STaR-CS/requirements.txt`
2. Place raw session files under `STaR-CS/Data_Set_25/` (pairs of `.srt` subtitles and matching `.txt` role maps).
3. Set environment variables in `.env` (API keys, base URL) for the stages that call LLMs.
4. Run the pipeline in order:
   - `python STaR-CS/data_cleaning.py` → writes `STaR-CS/output/1_structured_data/`
   - `python STaR-CS/desensitize_data.py` → writes `STaR-CS/output/2_desensitized_final/`
   - `python STaR-CS/extract_skeleton.py` → writes `STaR-CS/output/3_dialogue_skeletons/`
   - `python STaR-CS/analyze_style.py` → writes `STaR-CS/output/Therapist_Style_and_Personality_Analysis.md`
   - `python STaR-CS/virtual_participant_generator.py` → writes `STaR-CS/output/4_synthetic_data/`
5. Adjust top-of-file constants in each script to point to your input/output paths and API endpoint. The README in `STaR-CS/` has the full parameter list.

### 2) RCA_Agent: Generate and Evaluate Responses
1. Install deps (minimal): `pip install openai python-dotenv tqdm bert-score`
2. Configure backbones for generation in [RCA_Agent/experiment_suite/backends.py](RCA_Agent/experiment_suite/backends.py): set `api_key`, `base_url`, and `model` names per backbone.
3. Configure the critic/reward model and concurrency in [RCA_Agent/rca_framework/config.py](RCA_Agent/rca_framework/config.py).
4. Prepare data: place synthetic dialogue JSONs from STaR-CS under `RCA_Agent/synthetic_data/`.
5. Run Stage 1 generation: `python RCA_Agent/generation.py`
   - Produces `RCA_Agent/stage1_generation.jsonl` and `RCA_Agent/viz_data.jsonl`.
6. Run Stage 2 evaluation: `python RCA_Agent/evaluation.py`
   - Produces `RCA_Agent/experiment_results.jsonl` with LLM-judge scores and BERTScore.
7. Optional stress test: `python RCA_Agent/stress_test.py` to compare backbones and the RCA agent on a handcrafted scenario.

## Configuration Pointers
- Rate limits: tune `MAX_CONCURRENT_REQUESTS` and `API_SEMAPHORE` in [RCA_Agent/rca_framework/config.py](RCA_Agent/rca_framework/config.py) plus `API_LIMIT`/`WORKER_LIMIT` near the top of [RCA_Agent/generation.py](RCA_Agent/generation.py).
- Style cards and prompts: defined in [RCA_Agent/rca_framework/config.py](RCA_Agent/rca_framework/config.py).
- Baselines and judges: see [RCA_Agent/experiment_suite/baselines.py](RCA_Agent/experiment_suite/baselines.py) and [RCA_Agent/experiment_suite/judges.py](RCA_Agent/experiment_suite/judges.py).

## Outputs
- STaR-CS: structured JSONs, de-identified JSONs, dialogue skeletons, therapist style report, and labeled synthetic dialogues (all under `STaR-CS/output/`).
- RCA_Agent: generation logs (`stage1_generation.jsonl`, `viz_data.jsonl`) and scored results (`experiment_results.jsonl`).

## Tips
- Logs are append-only; delete the `.jsonl` outputs if you want a clean rerun.
- If BERTScore is missing, scores default to 0.0—install `bert-score` and place the local model to avoid slow downloads.
- Keep provider rate limits in mind when raising concurrency; throttling will slow runs.
