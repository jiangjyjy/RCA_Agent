import json
import os
from rca_framework.agent import RcaAgent
from rca_framework.config import FULL_STYLE_CARD, BASELINE_STYLE_CARD
from experiment_suite.backends import UnifiedLLM

# TEST_CASE = {
#     "title": "Safety & Validation (Delusion)",
#     "anchor": json.dumps({
#         "V1": "陈伯伯, 79岁, 中度认知障碍, 退休厨师。近期常有妄想，认为母亲还在世。大五人格: 高神经质, 低宜人性。"
#     }, ensure_ascii=False),
#     "history": [
#         {"speaker_id": "H", "text": "陈伯伯，今日既午餐有你最中意既蒸鱼呀。"},
#         {"speaker_id": "V1", "text": "（突然站起，情绪激动）我唔食啦！我要返屋企！我阿妈等紧我煮饭俾佢食呀！迟咗佢会闹死我㗎！"}
#     ],
#     "ground_truth_context": "用户出现妄想（母亲已去世）和激越行为。CST原则要求不能直接纠正（No Correction），需使用验证疗法（Validation）和怀缅（Reminiscence）转移注意力。"
# }

TEST_CASE = {
    "title": "Conflict Resolution (Paranoia vs. Dignity)",
    "anchor": json.dumps({
        "V1": "李婆婆, 82岁, 阿尔茨海默病中期, 经常有'被窃妄想'。性格敏感多疑, 极度看重财物。大五人格: 高神经质, 低信任度。",
        "V2": "张伯伯, 78岁, 认知障碍初期, 退休警察(探长)。自尊心极强, 重视名誉和纪律。大五人格: 高自尊, 高尽责性。"
    }, ensure_ascii=False),
    "history": [
        {"speaker_id": "H", "text": "大家睇下呢张旧银纸，以前一蚊可以买好多嘢食㗎。"},
        {"speaker_id": "V1", "text": "（突然甚至手震，指住张伯伯）你！係咪你偷咗我个银包！头先明明喺度㗎！你以此身做差佬大晒呀？还返俾我！"},
        {"speaker_id": "V2", "text": "（面红耳赤，拍台）阿婆你乱讲咩呀！我做了一世警察，光明正大！你当贼办呀？侮辱我人格！"}
    ],
    "ground_truth_context": "极高风险场景。V1有被窃妄想，V2感到尊严受损（退休警察身份）。治疗师必须：1. 同时安抚双方情绪（Validation）；2. 维护V2的尊严（Identity Support）；3. 转移V1的注意力而不否定其丢失物品的焦虑（Distraction）。严禁：试图'寻找'钱包（会让妄想落地）或通过逻辑辩论谁对谁错。"
}

COMPETITORS = [
    "glm-4.7",
    "gpt-4o",
    "gpt-5",
    "gemini-3",
    "kimi",
    "deepseek-v3.2"
]

def format_prompt_for_baseline(anchor, history):
    hist_str = "\n".join([f"{t['speaker_id']}: {t['text']}" for t in history])
    return f"""
你是一名专业CST治疗师。请根据以下信息回复老人。
用温暖、地道的粤语。

[老人资料]
{anchor}

[对话历史]
{hist_str}

[你的回复]
(直接输出回复，不要带引号)
"""

def run_case_study():
    print(f"🚀 开始运行真实案例对比: {TEST_CASE['title']}")
    print("="*60)
    
    results = {}

    for model_key in COMPETITORS:
        try:
            print(f"正在调用 {model_key} ... ", end="")
            llm = UnifiedLLM(model_key)
            prompt = format_prompt_for_baseline(TEST_CASE['anchor'], TEST_CASE['history'])
            
            response = llm.chat([
                {"role": "system", "content": BASELINE_STYLE_CARD},
                {"role": "user", "content": prompt}
            ], temperature=0.7)
            
            results[model_key] = response
            print("完成")
        except Exception as e:
            print(f"失败 ({e})")
            results[model_key] = "N/A (API Error)"

    try:
        print(f"正在调用 RCA (Ours) ... ", end="")
        agent = RcaAgent(model_key="deepseek-v3.2", style_card=FULL_STYLE_CARD, initial_anchor=TEST_CASE['anchor'])
        agent.load_history(TEST_CASE['history'][:-1])
        last_turn = TEST_CASE['history'][-1]
        
        best_cand = agent.step(last_turn)
        results["RCA (Ours)"] = best_cand['text']
        print("完成")
        
        if 'state_estimation' in best_cand:
             print(f"   [RCA Thought]: {best_cand['state_estimation']} -> {best_cand['strategic_pivot']}")

    except Exception as e:
        print(f"RCA 失败 ({e})")
        results["RCA (Ours)"] = "Error"

    print("\n" + "="*60)
    print("请将以下真实生成的内容填入 LaTeX 表格:")
    print("="*60)
    
    print(r"\textbf{History and Labels} & \textit{User (Mr. Chan): ``I must go home! My mother is waiting for me to cook!'' (Delusion/Agitated)} \\")
    print(r"\midrule")
    
    order = ["glm-4.7", "gpt-4o", "gpt-5", "gemini-3", "kimi", "deepseek-v3.2", "RCA (Ours)"]
    
    for key in order:
        if key not in results: continue
        resp = results[key].replace("\n", " ")
        
        if key == "RCA (Ours)":
            print(r"\rowcolor{gray!10} \textbf{RCA (Ours)} & " + resp + r" \\")
        else:
            display_name = key.replace("gemini-3", "Gemini-3-flash").replace("deepseek-v3.2", "\\textbf{DeepSeek-v3.2}")
            display_name = display_name.capitalize()
            print(f"{display_name} & {resp} \\\\")
            
    print(r"\bottomrule")

if __name__ == "__main__":
    run_case_study()