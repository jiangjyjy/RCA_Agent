import json
import time
import random
import threading
import concurrent.futures
import traceback
import os
import sys
from tqdm import tqdm
from pathlib import Path

TEST_DATA_DIR = Path("synthetic_data") 
GEN_OUTPUT_LOG = "stage1_generation.jsonl" 
VIZ_LOGGER = None 

NUM_SAMPLES = 50 
SEED = 2026 
TARGET_BACKBONE = "gpt-4o" 

import rca_framework.config

backbone_lower = TARGET_BACKBONE.lower()

if "glm" in backbone_lower:
    API_LIMIT = 6
    WORKER_LIMIT = 6
    print(f"[Config] Detected GLM. Mode: Strict (Limit={API_LIMIT})")

elif "kimi" in backbone_lower:
    API_LIMIT = 10       
    WORKER_LIMIT = 12    
    print(f"[Config] Detected Kimi Instruct. Mode: Moderate (Limit={API_LIMIT})")

elif "thinking" in backbone_lower or "r1" in backbone_lower:
    API_LIMIT = 2
    WORKER_LIMIT = 3
    print(f"[Config] Detected Reasoning Model. Mode: Slow (Limit={API_LIMIT})")

else:
    API_LIMIT = 10
    WORKER_LIMIT = 12
    print(f"[Config] Detected Standard Model. Mode: Fast (Limit={API_LIMIT})")

rca_framework.config.MAX_CONCURRENT_REQUESTS = API_LIMIT
rca_framework.config.API_SEMAPHORE = threading.Semaphore(API_LIMIT)

from rca_framework.agent import RcaAgent
from rca_framework.config import FULL_STYLE_CARD, BASELINE_STYLE_CARD
from experiment_suite.baselines import ZeroShotAgent, FewShotAgent, CoTAgent
from experiment_suite.visual_data import VisualDataLogger

VIZ_LOGGER = VisualDataLogger("viz_data.jsonl")
FILE_LOCK = threading.Lock()

print_lock = threading.Lock()
def safe_print(msg):
    with print_lock:
        print(msg, flush=True)

def get_completed_ids():
    completed = set()
    if not os.path.exists(GEN_OUTPUT_LOG): return completed
    try:
        with open(GEN_OUTPUT_LOG, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if len(data.get('outputs', {})) == 4:
                    completed.add(data['sample_id'])
    except: pass
    return completed

def load_smart_samples():
    safe_print(f"[System] Scanning data with SEED={SEED}...")
    files = list(TEST_DATA_DIR.glob("*.json"))
    if not files: return []
    random.seed(SEED)
    random.shuffle(files)
    samples = []
    target_turn_range = range(5, 26)
    for f in files:
        if len(samples) >= NUM_SAMPLES: break
        try:
            d = json.load(open(f, encoding='utf-8'))
            dialogue = d['dialogue']
            valid_indices = [i for i in range(len(dialogue)-1) 
                             if i in target_turn_range 
                             and dialogue[i]['speaker_id'].startswith('V')
                             and dialogue[i+1]['speaker_id'] == 'H']
            if not valid_indices: continue
            cut_idx = random.choice(valid_indices)
            sample = {
                "id": f"{d['synthetic_id']}_turn{cut_idx}",
                "anchor": json.dumps(d['virtual_participants'], ensure_ascii=False), 
                "history": dialogue[:cut_idx+1],
                "ground_truth": dialogue[cut_idx+1]['text'],
                "turn_depth": cut_idx
            }
            samples.append(sample)
        except: continue
    return samples

def run_generation(sample):
    sid = sample['id']
    history_text = "\n".join([f"{t['speaker_id']}: {t['text']}" for t in sample['history']])
    base_ctx = f"[画像]\n{sample['anchor']}\n[对话]\n{history_text}"
    
    entry = {
        "sample_id": sid,
        "ground_truth": sample['ground_truth'],
        "turn_depth": sample['turn_depth'],
        "anchor": sample['anchor'],
        "history_text": history_text,
        "outputs": {}
    }

    experiments = [
        (ZeroShotAgent, TARGET_BACKBONE, "Base (Zero-Shot)", BASELINE_STYLE_CARD),
        (FewShotAgent,  TARGET_BACKBONE, "Few-Shot",         BASELINE_STYLE_CARD),
        (CoTAgent,      TARGET_BACKBONE, "CoT (No-IVA)",     FULL_STYLE_CARD),
        (None,          TARGET_BACKBONE, "RCA (Ours)",       FULL_STYLE_CARD), 
    ]

    for AgentCls, model_key, name, card in experiments:
        safe_print(f"[{sid}] Generating {name}...")
        try:
            start = time.time()
            if name == "RCA (Ours)":
                agent = RcaAgent(model_key=TARGET_BACKBONE, style_card=card, initial_anchor=sample['anchor'])
                agent.load_history(sample['history'][:-1]) 
                input_turn = sample['history'][-1]
                
                resp_obj, candidates = agent.step(input_turn, return_debug_data=True)
                resp_text = resp_obj['text']
                
                with FILE_LOCK:
                    VIZ_LOGGER.log_thinking_process(sid, "RCA", 
                                                    resp_obj.get('state_estimation'), 
                                                    resp_obj.get('strategic_pivot'), "Action")
                    VIZ_LOGGER.log_reward_surface(f"{sid}_RCA", candidates)
            else:
                agent = AgentCls(model_key, card)
                resp_text = agent.step(base_ctx)
            
            latency = time.time() - start
            entry['outputs'][name] = {"response": resp_text, "latency": latency, "scores": {}}
            
        except Exception as e:
            safe_print(f"[{sid}] Error in {name}: {e}")
            return None

    with FILE_LOCK:
        with open(GEN_OUTPUT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    return sid

def main():
    completed = get_completed_ids()
    safe_print(f"[Stage 1] Found {len(completed)} completed samples.")

    all_samples = load_smart_samples()
    tasks = [s for s in all_samples if s['id'] not in completed]
    
    if not tasks:
        safe_print("[Stage 1] All done!")
        return

    safe_print(f"[Stage 1] Remaining: {len(tasks)}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKER_LIMIT) as executor:
        futures = [executor.submit(run_generation, s) for s in tasks]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            pass
            
    safe_print("[Stage 1] Finished.")

if __name__ == "__main__":
    main()