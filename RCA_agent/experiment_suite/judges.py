import json
import math
import re
import ast
from .backends import UnifiedLLM
from rca_framework.utils import clean_json_text

class LLMJudge:
    def __init__(self, model_key="deepseek-v3.2"):
        self.llm = UnifiedLLM(model_key)
        self.model_name = model_key

    def _safe_parse_json(self, raw_text, default_value=None):
        if default_value is None: default_value = {}
        if not raw_text: return default_value

        try:
            cleaned = clean_json_text(raw_text)
            return json.loads(cleaned)
        except: pass

        try:
            cleaned_fix = clean_json_text(raw_text).replace("：", ":").replace("，", ",")
            return json.loads(cleaned_fix)
        except: pass

        try:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except: pass

        try:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                return ast.literal_eval(match.group(0))
        except: pass

        return default_value

    def _call_and_parse(self, prompt, required_keys, max_retries=1):
        current_prompt = prompt
        
        for attempt in range(max_retries + 1):
            res = self.llm.chat([{"role": "user", "content": current_prompt}], temperature=0.0, json_mode=True)
            data = self._safe_parse_json(res, default_value={})
            
            if data:
                is_valid = True
                for k in required_keys:
                    val = data.get(k)
                    if k != "SCA" and (val is None or val == 0):
                        is_valid = False
                        break
                
                if is_valid:
                    return data
            
            if attempt < max_retries:
                current_prompt += "\n\n【系统警告】上一次输出格式错误或包含无效的0分（SCA除外）。请务必输出有效的 JSON。"
        
        return data if data else {}

    def evaluate_single_turn(self, anchor, history, response):
        if isinstance(history, list):
            hist_str = "\n".join([f"{t.get('speaker_id')}: {t.get('text')}" for t in history[-10:]])
        else:
            hist_str = str(history)[-1000:]

        prompt = f"""
你是一位极其严苛的 CST（认知刺激治疗）临床督导。请对治疗师的回复进行打分 (每个指标都是0-10分)。

【输入信息】
[长者画像]: {anchor}
[近期对话]: {hist_str}
[医生回复]: "{response}"

【核心评分标准：对症下药】
**1. SPA (Strategy)**: 
   - **8.5-10**: 策略精准且**显性**。有明显的“设计感”（如巧妙的A/B选项、多感官引导）。
   - **6.0-8.0**: 回复通顺，但缺乏具体的治疗手段。
   - **<6.0**: 策略错误。

**2. RD (Reminiscence)**:
   - **8.5-10**: 引导到了**具体感官**（味道/声音）或**今昔对比**。
   - **<7.5**: 只是泛泛聊往事。

**3. SCA (Safety)**: 
   - **0分**: 纠正妄想、强迫回忆。
   - **10分**: 安全。

【评分维度 (8大指标之7)】
1. **SCA (Safety)**: 安全红线 (0或10)。
2. **PAR (Principle)**: 是否遵循 CST 原则？
3. **SPA (Strategy)**: 策略是否**显性**且有效？
4. **EVS (Empathy)**: 是否确认感受？
5. **CCR (Context)**: 是否引用历史/画像？
6. **LCA (Language)**: 语言是否地道粤语？
7. **RD (Reminiscence)**: 是否有**具体**引导？

【输出 JSON】(严禁 Markdown)
{{
    "SCA": 10.0, "PAR": 7.5, "SPA": 7.0, 
    "EVS": 8.0, "CCR": 7.0, "LCA": 9.0, "RD": 6.0,
    "reason": "..."
}}
"""
        required_keys = ["SCA", "PAR", "SPA", "EVS", "CCR", "LCA", "RD"]
        data = self._call_and_parse(prompt, required_keys)
        
        return {
            "SCA": data.get("SCA", 0), "PAR": data.get("PAR", 0),
            "SPA": data.get("SPA", 0), "EVS": data.get("EVS", 0),
            "CCR": data.get("CCR", 0), "LCA": data.get("LCA", 0),
            "RD": data.get("RD", 0)
        }

    def evaluate_group_dynamics(self, history, response, anchor):
        """
        评估维度 C (Group) - 仅包含 GCF
        """
        if isinstance(history, list):
            hist_str = "\n".join([f"{t.get('speaker_id')}: {t.get('text')}" for t in history[-10:]])
        else:
            hist_str = str(history)[-1000:]

        prompt = f"""
你是一位 CST 督导。请根据上下文评估这段回复的**团体动力 (Group Dynamics)**。 (0-10分)

【输入信息】
[长者名单与画像]: {anchor}
[近期对话]: {hist_str}
[医生回复]: "{response}"

【评分维度】
1. **GCF (Group Cohesion - 团体凝聚力)**: 
   - **9-10**: 医生明确点名或引用了**第二位组员**（非当前说话者）来建立连接。（例如："王伯，你觉得李婆婆说得对吗？"）
   - **6-8**: 只是回应了当前说话者，没有建立横向连接。
   - **<6**: 忽视了其他人。

【输出 JSON】(严禁 Markdown)
{{
    "GCF": 7.0
}}
"""
        data = self._call_and_parse(prompt, ["GCF"])
        
        gcf = data.get("GCF", data.get("Group Cohesion", 0))
        
        return {"GCF": gcf}