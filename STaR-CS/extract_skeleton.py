import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import multiprocessing

import time

load_dotenv()
KEY = os.getenv("XXX")
BASE_URL = "XXX"
MODEL_NAME = "deepseek-ai/deepseek-v3"

NUM_PROCESSES = 1

INPUT_DIR = Path("output/2_desensitized_final_test/") 
OUTPUT_DIR = Path("output/3_dialogue_skeletons")

client = None
progress_queue = None

def initialize_worker(q, api_key, base_url):
    global client, progress_queue
    progress_queue = q
    client = OpenAI(api_key=api_key, base_url=base_url)

def get_skeleton_extraction_prompt(dialogue_turns: list[dict]) -> str:
    dialogue_str = "\n".join([f"{turn['speaker_id']}: {turn['text_cantonese']}" for turn in dialogue_turns])
    return f"""
# 角色
你是一位专业的对话分析师和剧本结构专家，擅长将复杂的对话分解为核心的互动模式。

# 任务
请分析以下CST小组对话的完整文本，并将其转换为一个结构化的“对话骨架 (Dialogue Skeleton)”。
这个骨架应该捕捉对话的主要流程和关键的互动转向。

# 对话文本
--- START OF DIALOGUE ---
{dialogue_str}
--- END OF DIALOGUE ---

# 互动行为分类 (Action Categories)
在分析时，请将每一轮的发言归纳为以下几种核心行为之一：
- **开启话题 (Initiate_Topic)**: 治疗师或参与者开启一个全新的话题或活动。
- **提问 (Question)**: 提出一个开放式或封闭式的问题。
- **回答 (Answer)**: 直接回应一个问题。
- **肯定/鼓励 (Affirm/Encourage)**: 对他人的发言表示赞同、表扬或鼓励。
- **补充/阐述 (Elaborate)**: 在已有话题上提供更多信息、细节或个人想法。
- **质疑/挑战 (Challenge)**: 对他人的观点提出不同看法或质疑。
- **表达情绪 (Express_Emotion)**: 直接表达个人感受（如高兴、困惑、难过）。
- **引导/转向 (Facilitate/Redirect)**: 治疗师主动管理对话流程，点名某人发言或转换话题焦点。
- **社交回应 (Social_Response)**: 简单的附和、感谢、问候等。

# 输出格式 (必须严格遵守JSON格式)
请返回一个JSON对象，其中包含一个名为 "skeleton" 的列表。列表中的每个元素都代表一个互动步骤，包含以下字段：
- "speaker": 发言者的ID (e.g., "H", "V1")。
- "action": 从上述“互动行为分类”中选择的最贴切的一个。
- "target": 互动的目标对象。如果是面向所有人，请使用 "ALL"；如果是针对特定某人，请使用其ID (e.g., "V2")。
- "summary": 用一句话简要概括该轮发言的核心内容。

# JSON输出格式示例
{{
  "skeleton": [
    {{ "speaker": "H", "action": "Initiate_Topic", "target": "ALL", "summary": "欢迎大家并引入今天的主题'物件分类'" }},
    {{ "speaker": "V2", "action": "Social_Response", "target": "H", "summary": "热情地表示期待" }}
  ]
}}
"""

def call_extraction_api(prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        progress_queue.put({'level': 'error', 'message': f"API 调用失败: {e}"})
        return {"skeleton": [{"error": str(e)}]}

def process_file_task(file_path_tuple):
    input_path, output_path = file_path_tuple

    progress_queue.put({'level': 'info', 'message': f"开始处理: {input_path.name}"})

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dialogue = data.get('dialogue')
        if not dialogue:
            progress_queue.put({'level': 'warning', 'message': f"文件 {input_path.name} 中没有对话内容，跳过。"})
            return None

        prompt = get_skeleton_extraction_prompt(dialogue)
        skeleton_data = call_extraction_api(prompt)

        output_content = {
            "session_id": data.get("session_id"),
            "source_files": data.get("source_files"),
            "speaker_map": data.get("speaker_map"),
            "privacy_map": data.get("privacy_map"),
            "dialogue_skeleton": skeleton_data.get("skeleton", [])
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_content, f, ensure_ascii=False, indent=4)

        return str(output_path.name)
    except Exception as e:
        progress_queue.put({'level': 'error', 'message': f"处理文件 {input_path.name} 时发生严重错误: {e}"})
        return None

def main():
    if not INPUT_DIR.exists():
        print(f"[错误] 输入目录不存在: {INPUT_DIR}")
        return

    json_files = sorted(list(INPUT_DIR.rglob('*.json')))
    if not json_files:
        print(f"[警告] 在目录 '{INPUT_DIR}' 中未找到任何 .json 文件。")
        return

    file_path_tuples = []
    for input_json_path in json_files:
        relative_path = input_json_path.relative_to(INPUT_DIR)
        output_json_path = OUTPUT_DIR / f"{relative_path}"
        file_path_tuples.append((input_json_path, output_json_path))

    with multiprocessing.Manager() as manager:
        progress_q = manager.Queue()
        pool_initargs = (progress_q, KEY, BASE_URL)
        
        print(f"启动 {NUM_PROCESSES} 个工作进程...")
        with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=initialize_worker, initargs=pool_initargs) as pool:
            results = [pool.apply_async(process_file_task, args=(f_tuple,)) for f_tuple in file_path_tuples]
            
            with tqdm(total=len(results), desc="总体文件进度") as pbar:
                for res in results:
                    res.get()
                    pbar.update(1)

                    while not progress_q.empty():
                        msg = progress_q.get()
                        if msg['level'] != 'progress':
                            tqdm.write(f"[{msg.get('level', 'INFO').upper()}] {msg.get('message', '')}")
            
    print(f"\n--- 对话骨架提取完成 ---")
    print(f"共提取了 {len(json_files)} 个会话的骨架。")
    print(f"结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()