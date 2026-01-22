# STaR-CS

A complete pipeline for processing and synthesizing CST (Cognitive Stimulation Therapy) group dialogues:

- Convert raw subtitles/annotations (SRT + TXT) into standardized JSON (data_cleaning)
- Perform strict PII de-identification with an LLM (desensitize_data)
- Extract structured “Dialogue Skeletons” from dialogues (extract_skeleton)
- Aggregate the therapist’s (H) utterances and produce a style/personality report (analyze_style)
- Generate high-quality, multi-label synthetic dialogues based on the “style card + scene + dialogue skeleton” (virtual_participant_generator)


## Directory Structure

- Data_Set_25/ — Input data root containing sessions’ `.srt` and corresponding `.txt`
- output/
  - 1_structured_data/ — Stage 1: structured JSON output
  - 2_desensitized_final/ — Stage 2: de-identified JSON output
  - 3_dialogue_skeletons/ — Stage 3: extracted dialogue skeletons (JSON)
  - 4_synthetic_data/ — Stage 5: generated synthetic dialogues (JSON)
  - Therapist_Style_and_Personality_Analysis.md — Stage 4: therapist style card


## Requirements

- Python 3.10+
- An OpenAI-compatible inference API service (used for de-identification, skeleton extraction, style analysis, and synthesis)

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare environment variables（.env）:

```
XXX=your_api_key_here
```

> Note: Setting `KEY` as well is only to keep console messages consistent.


## Input Data Preparation

Each session must have a pair of files:

- `*.srt`: subtitle lines with timestamps. Each line is recommended as `H: ...`, `V1: ...`, `V2: ...`, etc. If an entry contains multiple lines, they are parsed line-by-line and appended to the previous turn when appropriate.
- `*.txt`: a same-named role mapping file to provide descriptions for IDs like `H`, `V1`, `V2`. Example lines:
  - `H: Therapist`
  - `V1: Male elder`
  - `V2: Female elder`

See [data_cleaning.py](data_cleaning.py) for parsing rules.


## Configurables (Important)

Before running, adjust the top-level constants in each script according to your local paths:

- [data_cleaning.py](data_cleaning.py)
  - `input_dir = Path("Data_Set_25")`
  - `output_dir = Path("output/1_structured_data")`

- [desensitize_data.py](desensitize_data.py)
  - `BASE_INPUT_DIR = Path("output/1_structured_data")`   # Recommended to use this repository’s standard folder
  - `BASE_OUTPUT_DIR = Path("output/2_desensitized_final")`
  - Requires `API_KEY` and `BASE_URL` in `.env`
  - `NUM_PROCESSES` can be tuned based on CPU cores

- [extract_skeleton.py](extract_skeleton.py)
  - `INPUT_DIR = Path("output/2_desensitized_final")`
  - `OUTPUT_DIR = Path("output/3_dialogue_skeletons")`
  - Requires `API_KEY` and `BASE_URL` in `.env`

- [analyze_style.py](analyze_style.py)
  - `INPUT_DIR = Path("output/2_desensitized_final")`
  - `OUTPUT_FILE = Path("output/Therapist_Style_and_Personality_Analysis.md")`
  - Requires `API_KEY` and `BASE_URL` in `.env`
  - `CHUNK_TARGET_SIZE_TOKENS` can be tuned based on your model context

- [virtual_participant_generator.py](virtual_participant_generator.py)
  - `ANALYSIS_REPORT_PATH = Path("output/Therapist_Style_and_Personality_Analysis.md")`
  - `SKELETON_DIR = Path("output/3_dialogue_skeletons")`
  - `OUTPUT_DIR = Path("output/4_synthetic_data")`
  - Ensure `PATIENT_GEN_MODEL` (participant generation) and `MAIN_GEN_MODEL` (main dialogue generation) are available on your API
  - Adjust `NUM_DIALOGUES_TO_GENERATE` (count) and `NUM_PROCESSES` (parallelism) as needed


## Execution Order

1. Structure raw data
   ```bash
   python data_cleaning.py
   ```
   - Input: `Data_Set_25/**/*.srt` and same-named `*.txt`
   - Output: `output/1_structured_data/**/*.json`

2. De-identification (requires API)
   ```bash
   # Make sure BASE_INPUT_DIR/BASE_OUTPUT_DIR are set as above
   python desensitize_data.py
   ```
   - Input: `output/1_structured_data/**/*.json`
   - Output: `output/2_desensitized_final/**/*.json`

3. Extract dialogue skeletons (requires API)
   ```bash
   # Ensure extract_skeleton.py INPUT_DIR points to 2_desensitized_final
   python extract_skeleton.py
   ```
   - Input: `output/2_desensitized_final/**/*.json`
   - Output: `output/3_dialogue_skeletons/**/*.json`

4. Therapist style analysis (requires API)
   ```bash
   # Ensure analyze_style.py INPUT_DIR points to 2_desensitized_final
   python analyze_style.py
   ```
   - Output: `output/Therapist_Style_and_Personality_Analysis.md`

5. Generate synthetic dialogues (requires API)
   ```bash
   # Ensure virtual_participant_generator.py can read the outputs above
   python virtual_participant_generator.py
   ```
   - Inputs: style card + dialogue skeletons
   - Output: `output/4_synthetic_data/synthetic_dialogue_*.json`


## Artifacts

- `output/1_structured_data/*.json`: Structured session results, including `speaker_map` and per-turn `dialogue`.
- `output/2_desensitized_final/*.json`: Adds `privacy_map` and placeholders for sensitive spans (e.g., `[PERSON_3]`).
- `output/3_dialogue_skeletons/*.json`: Dialogue skeletons for each session (process-level structure).
- `output/Therapist_Style_and_Personality_Analysis.md`: Style/personality report for therapist (H).
- `output/4_synthetic_data/*.json`: New dialogues adhering to the style card and skeletons, labeled with `emotion`, `strategy`, `principles`, and `cognitive_state`.


## Tips & FAQ

- Windows multiprocessing: `multiprocessing.freeze_support()` is used where necessary.
- `pysrt` encoding: If read errors occur, confirm the `.srt` is `utf-8` (or pass the correct encoding to `pysrt.open`).
- Token chunking: `analyze_style.py` uses `tiktoken` for approximate token counting; tune with `CHUNK_TARGET_SIZE_TOKENS`.
- API compatibility: If using an OpenAI-compatible service, ensure `BASE_URL` and model names (e.g., `deepseek-ai/deepseek-v3`, `Qwen/Qwen3-30B`) are available.
- Env var names: To avoid confusion, consider providing both `XXX` and `KEY` in `.env`, or unify the variable names in code.
