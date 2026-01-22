import os
import math
import threading
from pathlib import Path
from collections import Counter

BERT_LOCK = threading.Lock()

try:
    from bert_score import score as bs_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("[Warning] 'bert-score' lib not found. BERTScore will be 0.0.")

def calculate_bertscore(refs, cands):
    if not HAS_BERTSCORE: return 0.0
    
    with BERT_LOCK:
        try:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            model_dir = project_root / "local_models" / "bert-base-chinese"
            model_path_str = str(model_dir) if model_dir.exists() else "bert-base-chinese"

            P, R, F1 = bs_score(
                cands, 
                refs, 
                model_type=model_path_str, 
                num_layers=12, 
                verbose=False,
                device="cpu" 
            )
            return F1.mean().item()
        except Exception as e:
            print(f"\n[BERTScore Error] {type(e).__name__}: {e}")
            return 0.0

def calculate_distinct(responses):
    if not responses: return 0.0, 0.0
    unigrams = set()
    bigrams = set()
    total_unigrams = 0
    total_bigrams = 0
    for r in responses:
        tokens = list(r)
        if not tokens: continue
        for t in tokens: unigrams.add(t)
        total_unigrams += len(tokens)
        for i in range(len(tokens)-1):
            bigrams.add(tokens[i] + tokens[i+1])
            total_bigrams += len(tokens) - 1
    d1 = len(unigrams) / total_unigrams if total_unigrams > 0 else 0.0
    d2 = len(bigrams) / total_bigrams if total_bigrams > 0 else 0.0
    return d1, d2

def calculate_bleu(reference, candidate):
    ref_tokens = list(reference)
    cand_tokens = list(candidate)
    if not cand_tokens: return 0.0
    c_counts = Counter(cand_tokens)
    r_counts = Counter(ref_tokens)
    overlap = sum((c_counts & r_counts).values())
    precision = overlap / len(cand_tokens)
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(cand_tokens))
    return bp * precision * 100

def calculate_rouge_l(reference, candidate):
    if not candidate or not reference: return 0.0
    m, n = len(reference), len(candidate)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == candidate[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p = lcs / n if n > 0 else 0
    r = lcs / m if m > 0 else 0
    f1 = 0
    if p + r > 0:
        f1 = 2 * p * r / (p + r)
    return f1 * 100