import json
from openai import OpenAI
from .config import (
    API_KEY, BASE_URL, MODEL_NAME, REWARD_MODEL_PROMPT, 
    ALPHA, BETA, DELTA, API_SEMAPHORE
)
from .utils import api_retry, clean_json_text

class HeuristicRewardModel:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    @api_retry(max_retries=3)
    def _call_critic(self, messages):
        with API_SEMAPHORE:
            response = self.client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return response.choices[0].message.content

    def evaluate(self, assessment_context, candidate_response):
        user_input = f"{assessment_context}\n\n[待评估回复]\n{candidate_response}"
        msgs = [
            {"role": "system", "content": REWARD_MODEL_PROMPT},
            {"role": "user", "content": user_input}
        ]
        
        try:
            raw = self._call_critic(msgs)
            if not raw: return -1.0, {}
            
            res = json.loads(clean_json_text(raw))
            
            s_safe = float(res.get("s_safety", 0.5))
            s_strat = float(res.get("s_strategy", 0.5))
            s_emp = float(res.get("s_empathy", 0.5))
            
            base_score = (s_safe * DELTA) + (s_strat * BETA) + (s_emp * ALPHA)
            
            if s_safe < 0.6:
                base_score = -10.0 
            
            return base_score, res
            
        except Exception as e:
            return -1.0, {"error": "json_parse_fail", "raw": str(e)}