import json
import time
import threading
import concurrent.futures
import traceback
import os
import random
from tqdm import tqdm
from experiment_suite.judges import LLMJudge
from experiment_suite.metrics import calculate_bertscore

INPUT_LOG = "stage1_generation.jsonl"
OUTPUT_LOG = "experiment_results.jsonl"

MAX_THREAD_WORKERS = 20 

SEMAPHORE_GLM = threading.Semaphore(5)
SEMAPHORE_DS = threading.Semaphore(14)

FILE_LOCK = threading.Lock()

GLOBAL_DATA = [] 
DATA_INDEX = {}

print_lock = threading.Lock()
def safe_print(msg):
    with print_lock:
        print(msg, flush=True)

def load_data_robust(filepath):
    data_list = []
    if not os.path.exists(filepath): return []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line) if isinstance(line, str) else line
                if isinstance(d, str): d = json.loads(d)
                data_list.append(d)
            except: continue
    return data_list

def init_storage():
    global GLOBAL_DATA, DATA_INDEX
    if not os.path.exists(OUTPUT_LOG):
        safe_print(f"[System] Initializing {OUTPUT_LOG} from input...")
        raw_data = load_data_robust(INPUT_LOG)
        if not raw_data:
            safe_print("[Fatal] Input data is empty!")
            exit()
        
        with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
            for d in raw_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    GLOBAL_DATA = load_data_robust(OUTPUT_LOG)
    
    DATA_INDEX = {d['sample_id']: i for i, d in enumerate(GLOBAL_DATA)}
    
    safe_print(f"[System] Storage initialized. Loaded {len(GLOBAL_DATA)} samples.")

def save_score_immediate(sid, model_name, scores):
    with FILE_LOCK:
        if sid not in DATA_INDEX: return
        
        idx = DATA_INDEX[sid]
        
        target_output = GLOBAL_DATA[idx]['outputs'].get(model_name)
        if not target_output: return
        
        if 'scores' not in target_output:
            target_output['scores'] = {}
        
        target_output['scores'].update(scores)
        temp_file = OUTPUT_LOG + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                for d in GLOBAL_DATA:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
            if os.path.exists(OUTPUT_LOG):
                os.remove(OUTPUT_LOG)
            os.rename(temp_file, OUTPUT_LOG)
        except Exception as e:
            safe_print(f"[Save Error] {e}")

def run_micro_task(judge, prefix, sid, model_name, anchor, history, response, gt):
    bert_score = 0.0
    if prefix == "DS":
        try:
            bert_score = calculate_bertscore([gt], [response])
        except: pass

    sem = SEMAPHORE_GLM if "glm" in prefix.lower() else SEMAPHORE_DS
    
    with sem:
        try:
            start_t = time.time()
            
            s1 = judge.evaluate_single_turn(anchor, history, response)
            s2 = judge.evaluate_group_dynamics(history, response, anchor)
            
            merged = {**s1, **s2}
            if "reason" in merged: del merged["reason"]
            
            final_scores = {f"{prefix}_{k}": v for k, v in merged.items()}
            if prefix == "DS":
                final_scores['BERTScore'] = bert_score
            
            save_score_immediate(sid, model_name, final_scores)
            
            duration = time.time() - start_t
            safe_print(f"[{sid}] {model_name} -> {prefix} ({duration:.1f}s)")
            
        except Exception as e:
            safe_print(f"[{sid}] {model_name} -> {prefix} Error: {e}")

def main():
    init_storage()
    
    judge_ds = LLMJudge("deepseek-v3.2")
    judge_glm = LLMJudge("glm-4.7")
    
    tasks = []
    
    for d in GLOBAL_DATA:
        sid = d['sample_id']
        anchor = d['anchor']
        history = d.get('history_text', "")
        if not history:
            history = "\n".join([f"{t['speaker_id']}: {t['text']}" for t in d.get('history', [])])
        gt = d['ground_truth']
        
        for model_name, output in d['outputs'].items():
            if 'response' not in output: continue
            
            resp_text = output['response']
            curr_scores = output.get('scores', {})
            
            if 'DS_SCA' not in curr_scores:
                tasks.append({
                    'func': run_micro_task,
                    'args': (judge_ds, "DS", sid, model_name, anchor, history, resp_text, gt)
                })
            
            if 'GLM_SCA' not in curr_scores:
                tasks.append({
                    'func': run_micro_task,
                    'args': (judge_glm, "GLM", sid, model_name, anchor, history, resp_text, gt)
                })
    
    if not tasks:
        safe_print("[System] All scores are complete! 🎉")
        return

    safe_print(f"[System] Identified {len(tasks)} missing micro-tasks.")
    
    random.shuffle(tasks)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS) as executor:
        futures = [executor.submit(t['func'], *t['args']) for t in tasks]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing"):
            pass

    safe_print("[System] Evaluation Finished.")

if __name__ == "__main__":
    main()