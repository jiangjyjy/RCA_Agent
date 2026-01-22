from .backends import UnifiedLLM

class BaseAgent:
    def __init__(self, model_key, style_card):
        self.llm = UnifiedLLM(model_key)
        self.style_card = style_card

class ZeroShotAgent(BaseAgent):
    def step(self, context):
        prompt = f"{self.style_card}\n\n【对话历史】\n{context}\n\n请直接回复。"
        return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.7)

class FewShotAgent(BaseAgent):
    def step(self, context):
        examples = """
【范例 1】
User: 我唔记得咗...
Assistant: 唔紧要，慢慢嚟。你以前做老师，记性一定好过我。如果不记得，我们看图片，这是苹果还是橙？(P-6 (多感官))

【范例 2】
User: 以前嘅嘢好食好多。
Assistant: 系呀，我都觉。你觉得以前嘅云吞面同依家比，最大的分别是汤底定係面条？(P-5 (怀缅联当下))
"""
        prompt = f"{self.style_card}\n\n【参考范例】\n{examples}\n\n【当前对话】\n{context}\n请参考范例风格回复。"
        return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.7)

class CoTAgent(BaseAgent):
    def step(self, context):
        prompt = f"""
{self.style_card}

【当前对话】
{context}

你是资深CST治疗师，回答之前请先进行一步步思考 (Think step-by-step)，分析老人情况。
**生成回复时，请注重“自然流畅”和“情感支持”，像一位耐心的朋友一样交流。**

格式：
[Reasoning] ...
[Response] ...
"""
        raw = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.7)
        if "[Response]" in raw:
            return raw.split("[Response]")[-1].strip()
        return raw