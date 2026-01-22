import os
import json
import time
from pathlib import Path
from openai import OpenAI
from typing import Dict, Optional, List
from dotenv import load_dotenv
from tqdm import tqdm
import re
import multiprocessing

load_dotenv()
KEY = os.getenv("XXX")
BASE_URL="XXX"
MODEL_NAME = "deepseek-ai/deepseek-v3"
NUM_PROCESSES = 5
BASE_INPUT_DIR = Path("output/1_structured_data7.30")
BASE_OUTPUT_DIR = Path("output/2_desensitized_final7.30")

client = None
worker_id = None
progress_queue = None

def initialize_worker(q, api_key, base_url):
    global client, worker_id, progress_queue
    worker_id = multiprocessing.current_process()._identity[0]
    progress_queue = q
    client = OpenAI(api_key=api_key, base_url=base_url)

def get_final_prompt(speaker_map: Dict, dialogue_history: List[Dict], current_turn: Dict, entity_map: Dict, entity_counters: Dict) -> str:
    speaker_map_str = json.dumps(speaker_map, ensure_ascii=False, indent=2)
    history_str = "\n".join([f"- {d.get('speaker_id', '')}: {d.get('text_cantonese', '')}" for d in dialogue_history[-5:]])
    entity_map_str = json.dumps(entity_map, ensure_ascii=False, indent=2)
    counters_str = json.dumps(entity_counters, ensure_ascii=False)
    return f"""
# 角色
你是一个顶级的、具有上下文记忆和推理能力、极其严谨、精准的数据隐私保护专家，严格遵循指令，区分真正的个人身份信息和通用词汇。
# 任务
你的核心任务是处理`current_turn`，仅识别并脱敏**真正的、具体的、能够直接或间接定位到个人的敏感信息**，并精确地更新实体地图和计数器。
# 已知上下文信息
1.  **会话角色地图 (speaker_map)**:
    {speaker_map_str}
2.  **最近对话历史 (dialogue_history)**:
    {history_str}
3.  **当前已发现的实体地图 (entity_map)**:
    {entity_map_str}
4.  **当前各类实体的最大ID计数器 (entity_counters)**:
    {counters_str}
# 处理流程 (你必须严格按以下顺序执行)
### 步骤 1: 识别
在 `current_turn` 的 `speaker_role` 和 `text_cantonese` 字段中，识别出所有潜在的敏感实体 (人名, 地点, 日期等)。
【极其重要】敏感实体的严格定义
- **PERSON**: 必须是明确的人名（全名、昵称、英文名），或者能明确指向某个特定个人的带称谓的词（如“邓婆婆”、“何生”）。
- **LOCATION**: 必须是具体的、非公开的地点（如街道地址、小区名称）。
- **DATE**: 必须是具体的、与个人事件强相关的日期（如生日、具体预约时间）。
- **INSTITUTE**: 必须是具体的机构名称（如具体公司名、学校名、医院名）。
【极其重要】排除规则 (Exclusion Rules)
你**绝对不能**将以下类型的通用词汇识别为敏感实体：
- **通用时间词**: 例如 `今日`, `下次`, `呢排`, `而家`, `以前`, `夜晚`, `早晨`, `星期日` 等所有表示相对时间或通用时间段的词。
- **通用地点词**: 例如 `香港`, `中國`, `日本`, `公园`, `海洋公园` 等众所周知的、非私人的地点。
- **通用人物指代**: 例如 `皇帝`, `姐姐`, `美人魚`, `小狐狸` 等角色、亲属关系或比喻性称呼。
- **通用数字**: 除非它们是电话号码、地址或身份证号的一部分。
- **产品/品牌名**, **食物/物品**, **说话人ID或拟声词**。

### 步骤 2: 查重与分配 (核心逻辑)
对于你在步骤1中识别出的每一个实体，执行以下判断：
- **检查**: (大小写不敏感地) 检查该实体是否已经作为 **键** 存在于 `entity_map` 中。
- **如果存在**: 在后续步骤中，必须使用 `entity_map` 中已有的占位符。
- **如果不存在**:
    a. 判断该实体的类别 (PERSON, LOCATION, DATE 等)。
    b. 从 `entity_counters` 中获取该类别的当前最大ID。
    c. **分配新的占位符**: 新的占位符必须是 `[CATEGORY_{{最大ID}}]`。
    d. 这个新发现的实体及其占位符，将是你输出中 `new_entities` 的一部分。

### 步骤 3: 排除
- **不要**将 `speaker_map` 中的纯职业/描述（如“穿紫色衣服的醫生”）识别为实体。只识别其中包含的核心名字（如“鄧婆婆”）。
- **不要**识别说话人ID（如 `V1`, `H`）或拟声词。
# 当前待处理轮次 (current_turn)
{{
    "speaker_role": "{current_turn['speaker_role']}",
    "text_cantonese": "{current_turn['text_cantonese']}"
}}
# 输出格式 (必须严格遵守JSON格式)
{{
  "modified_speaker_role": "处理后的speaker_role",
  "modified_text_cantonese": "处理后的text_cantonese",
  "log": [
    {{"entity": "原始实体1", "placeholder": "占位符1"}}
  ],
  "new_entities": [
    {{"entity": "新发现实体A", "category": "PERSON", "placeholder": "[PERSON_7]"}}
  ]
}}
"""

def call_api(prompt: str) -> Optional[Dict]:
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
                temperature=0, response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            progress_queue.put({'worker_id': worker_id, 'level': 'error', 'message': f"API 调用失败: {e}"})
            time.sleep(5)
    return None

def process_session_file_final(file_path_tuple):
    input_path, output_path = file_path_tuple
    
    progress_queue.put({'worker_id': worker_id, 'level': 'info', 'message': f"开始处理: {input_path.name}"})
    start_time = time.time()

    with open(input_path, 'r', encoding='utf-8') as f:
        session_data = json.load(f)

    entity_map, entity_map_lower, entity_counters = {}, {}, \
        {"PERSON": 1, "LOCATION": 1, "DATE": 1, "INSTITUTE": 1, "CONTACT": 1, "MEDIA": 1, "MISC": 1}
    
    dialogue_list = session_data.get('dialogue', [])
    total_turns = len(dialogue_list)
    
    for i, turn in enumerate(dialogue_list):
        progress_queue.put({
            'worker_id': worker_id, 'level': 'progress', 'total': total_turns,
            'current': i + 1, 'filename': input_path.name
        })

        dialogue_history = dialogue_list[:i]
        
        prompt = get_final_prompt(session_data['speaker_map'], dialogue_history, turn, entity_map, entity_counters)
        api_result = call_api(prompt)
        
        if api_result:
            turn['speaker_role'] = api_result.get("modified_speaker_role", turn['speaker_role'])
            turn['text_cantonese'] = api_result.get("modified_text_cantonese", turn['text_cantonese'])
            turn['desensitization_log'] = api_result.get("log", [])
            
            new_entities = api_result.get("new_entities", [])
            for item in new_entities:
                entity = item.get('entity')
                category = item.get('category', 'MISC').upper()
                placeholder = item.get('placeholder')
                
                if entity and placeholder:
                    key = entity.lower()
                    if key not in entity_map_lower:
                        entity_map[entity] = placeholder
                        entity_map_lower[key] = placeholder
                        match = re.match(r'\[([A-Z]+)_(\d+)\]', placeholder)
                        if match:
                            cat, num = match.groups()
                            entity_counters[cat] = max(entity_counters.get(cat, 0), int(num))


    privacy_map = {cat.lower(): [] for cat in entity_counters if entity_counters[cat] > 1}
    for entity, placeholder in entity_map.items():
        if placeholder:
            match = re.match(r'\[([A-Z]+)_\d+\]', placeholder)
            if match:
                cat_key = match.group(1).lower()
                if cat_key in privacy_map:
                    privacy_map[cat_key].append({"entity": entity, "placeholder": placeholder})

    ordered_session_data = {
        "session_id": session_data["session_id"], "source_files": session_data["source_files"],
        "speaker_map": session_data["speaker_map"], "privacy_map": privacy_map, "dialogue": session_data["dialogue"]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ordered_session_data, f, ensure_ascii=False, indent=4)
        
    end_time = time.time()
    progress_queue.put({
        'worker_id': worker_id, 'level': 'info', 
        'message': f"完成处理: {input_path.name}, 耗时: {end_time - start_time:.2f} 秒"
    })
    
    return str(output_path.name)

def main():
    if not KEY:
        raise ValueError("错误：请在项目根目录创建 .env 文件并添加 KEY='你的密钥'。")
    if not BASE_INPUT_DIR.exists():
        print(f"[错误] 输入目录不存在: {BASE_INPUT_DIR}")
        return
        
    print(f"--- 开始执行脱敏流程 ---")
    
    file_path_tuples = []
    for input_json_path in sorted(list(BASE_INPUT_DIR.rglob('*.json'))):
        relative_path = input_json_path.relative_to(BASE_INPUT_DIR)
        output_json_path = BASE_OUTPUT_DIR / relative_path
        file_path_tuples.append((input_json_path, output_json_path))

    with multiprocessing.Manager() as manager:
        progress_q = manager.Queue()
        pool_initargs = (progress_q, KEY, BASE_URL)
        
        with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=initialize_worker, initargs=pool_initargs) as pool:
            pool_results = [pool.apply_async(process_session_file_final, args=(f_tuple,)) for f_tuple in file_path_tuples]
            
            worker_progress = {} 
            main_pbar = tqdm(total=len(file_path_tuples), desc="总体文件进度", position=0)
            
            finished_tasks = 0
            while finished_tasks < len(file_path_tuples):
                try:
                    msg = progress_q.get(timeout=60) 
                    w_id = msg['worker_id']
                    
                    if msg['level'] == 'progress':
                        if w_id not in worker_progress:
                            worker_progress[w_id] = tqdm(total=msg['total'], desc=f"Worker-{w_id}", position=w_id, leave=True)
                        pbar = worker_progress[w_id]
                        pbar.set_description(f"Worker-{w_id} ({msg['filename']:.25s}...)")
                        pbar.update(1) 

                    elif msg['level'] == 'info':
                        tqdm.write(f"[Worker-{w_id}] {msg['message']}")
                        if "完成处理" in msg['message']:
                            if w_id in worker_progress:
                                worker_progress[w_id].close(); del worker_progress[w_id]
                            main_pbar.update(1)
                            finished_tasks += 1
                            
                    elif msg['level'] == 'error':
                        tqdm.write(f"[错误][Worker-{w_id}] {msg['message']}")

                except (KeyboardInterrupt, SystemExit): break
                except Exception: pass
            
            main_pbar.close()

    print(f"\n--- 所有脱敏处理完成 ---")
    print(f"最终结果已保存至: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()