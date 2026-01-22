import json
import concurrent.futures
from openai import OpenAI
from experiment_suite.backends import MODEL_CONFIG
from .config import (
    PLANNING_SYSTEM_PROMPT, GENERATION_SYSTEM_PROMPT,
    FULL_STYLE_CARD, K_CANDIDATES, 
    TEMPERATURE_PLANNING, TEMPERATURE_GENERATION,
    API_SEMAPHORE
)
from .memory import SemanticMemory
from .reward_model import HeuristicRewardModel
from .utils import clean_json_text, api_retry

class RcaAgent:
    def __init__(self, model_key="deepseek-v3.2", style_card=FULL_STYLE_CARD, initial_anchor=""):
        if model_key not in MODEL_CONFIG:
             raise ValueError(f"Model key {model_key} not found in backends.py")
        
        cfg = MODEL_CONFIG[model_key]
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        self.model_name = cfg["model"]
        
        self.memory = SemanticMemory(initial_anchor)
        self.reward_model = HeuristicRewardModel()
        self.style_card = style_card

    def load_history(self, history_list):
        for turn in history_list:
            self.memory.update_recent(turn)

    @api_retry(max_retries=3)
    def _call_llm(self, messages, temperature, json_mode=False):
        with API_SEMAPHORE:
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
            }
            if json_mode:
                params["response_format"] = {"type": "json_object"}
                
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content

    def _plan_clinical_strategy(self, agent_ctx):
        msgs = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": agent_ctx}
        ]
        res = self._call_llm(msgs, temperature=TEMPERATURE_PLANNING, json_mode=True)
        try:
            return json.loads(clean_json_text(res))
        except:
            return {
                "detected_state": "Neutral", 
                "selected_strategy": "P-11 以人为本", 
                "content_plan": "继续保持对话，关注老人感受。"
            }

    def _generate_candidate(self, agent_ctx, plan_json):
        sys_prompt = GENERATION_SYSTEM_PROMPT.format(
            style_card=self.style_card,
            plan_json=json.dumps(plan_json, ensure_ascii=False)
        )
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": agent_ctx}
        ]
        text = self._call_llm(msgs, temperature=TEMPERATURE_GENERATION)

        if text:
            import re
            text = text.replace("**", "")
            text = re.sub(r"\[.*?\]", "", text)
            text = re.sub(r"^(治疗师|Therapist|社工|姑娘)[：:]\s*", "", text).strip()
            text = re.sub(r"\n\s*\n", "\n", text).strip()
            
        return text, plan_json

    def step(self, input_turn, return_debug_data=False):
        self.memory.update_recent(input_turn)
        agent_ctx = self.memory.get_agent_context()
        assess_ctx = self.memory.get_assessment_context()

        clinical_plan = self._plan_clinical_strategy(agent_ctx)

        candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=K_CANDIDATES) as executor:
            futures = [executor.submit(self._generate_candidate, agent_ctx, clinical_plan) for _ in range(K_CANDIDATES)]
            for f in concurrent.futures.as_completed(futures):
                text, plan = f.result()
                if text:
                    candidates.append({
                        "text": text,
                        "state_estimation": plan.get("detected_state", ""),
                        "strategic_pivot": plan.get("selected_strategy", ""),
                        "content_planning": plan.get("content_plan", "")
                    })

        if not candidates:
            candidates.append({"text": "...", "iva_score": 0.0})

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            futures = {executor.submit(self.reward_model.evaluate, assess_ctx, c["text"]): c for c in candidates}
            for f in concurrent.futures.as_completed(futures):
                cand = futures[f]
                score, details = f.result()
                import random
                noise = random.uniform(0, 0.001)
                cand["iva_score"] = score + noise
                cand["iva_details"] = details

        safe_cands = [c for c in candidates if c["iva_score"] > 0.3]
        if safe_cands:
            best_cand = max(safe_cands, key=lambda x: x["iva_score"])
        else:
            best_cand = max(candidates, key=lambda x: x["iva_score"])

        if return_debug_data:
            return best_cand, candidates
        return best_cand