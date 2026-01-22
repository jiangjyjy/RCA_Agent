import os
import json
import time
import random
from pathlib import Path
from openai import OpenAI, Timeout
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict, List, Any
import multiprocessing
import re

load_dotenv()
KEY = os.getenv("XXX")
BASE_URL = "XXX"

PATIENT_GEN_MODEL = "deepseek-ai/deepseek-v3"
MAIN_GEN_MODEL = "Qwen/Qwen3-30B"

NUM_PROCESSES = 12
ANALYSIS_REPORT_PATH = Path("output/Therapist_Style_and_Personality_Analysis.md")
SKELETON_DIR = Path("output/3_dialogue_skeletons")
OUTPUT_DIR = Path("output/4_synthetic_data")

NUM_DIALOGUES_TO_GENERATE = 20000
TASK_RETRIES = 3
API_TIMEOUT = 300.0
SKELETON_MAX_LENGTH = 40
SKELETON_TRUNCATE_TO = 30

therapist_style_card = None
cst_scenes_list = None
all_skeletons = None
client = None

def initialize_worker(api_key, base_url, style_card, scenes, skeletons):
    global client, therapist_style_card, cst_scenes_list, all_skeletons
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=API_TIMEOUT)
    therapist_style_card = style_card
    cst_scenes_list = scenes
    all_skeletons = skeletons

def load_style_card() -> str:
    if not ANALYSIS_REPORT_PATH.exists():
        raise FileNotFoundError(f"错误：找不到风格卡片文件: {ANALYSIS_REPORT_PATH}")
    with open(ANALYSIS_REPORT_PATH, 'r', encoding='utf-8') as f:
        return f.read()

def get_cst_scenes() -> Dict[str, str]:
    return {
        '體能活動': '通过轻松的抛球游戏或模仿动作，进行身体活动，并分享与身体、运动相关的感受。',
        '聲音': '播放不同类别的声音（如动物、乐器），让组员进行识别和联想，或播放歌曲猜歌名。',
        '童年往事': '引导组员回忆并分享童年时期的游戏、食物、朋友或趣事。',
        '食物': '讨论有纪念价值或特别意义的食物，分享烹饪经验。',
        '時事': '讨论一些轻松有趣的报章杂志或电视新闻，分享个人看法。',
        '人面景物': '展示不同的人物或风景照片，引导组员讨论其异同、表达个人喜好。',
        '文字聯想': '进行如成语接龙、填字等游戏，促进语言表达能力。',
        '創意': '进行简单的创意手工或烹饪活动。',
        '物件分類': '提供多种物品或图片，邀请组员按不同标准进行分类。',
        '導向': '使用地图、日历或照片，讨论当前位置、日期、季节。',
        '金錢運用': '使用新旧钱币或道具，讨论物价变化或进行模拟购物。',
        '數字遊戲': '进行如“排七”、猜大小等简单的数字或纸牌游戏。',
        '文字遊戲': '进行如“有口難言”、接龙等文字或猜谜游戏。',
        '小組比賽': '将组员分为两队进行简单的问答或游戏比赛。'
    }

def load_all_skeletons(skeleton_dir: Path) -> List[Dict[str, Any]]:
    skeletons = []
    if not skeleton_dir.exists():
        raise FileNotFoundError(f"骨架目录不存在: {skeleton_dir}")
    for file_path in skeleton_dir.rglob("*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            skeletons.append(json.load(f))
    return skeletons

def _call_api_with_retry(prompt: str, model: str, temp: float, max_tokens: int, retries=3, delay=5) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temp, response_format={"type": "json_object"},
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Timeout:
            print(f"\n[警告] [{model}] API调用超时 (尝试 {attempt+1}/{retries})")
        except Exception as e:
            print(f"\n[警告] [{model}] API调用失败 (尝试 {attempt+1}/{retries}): {e}")
        
        if attempt < retries - 1:
            time.sleep(delay)
    return None

def generate_virtual_patients(num_patients: int = 5) -> Dict[str, str]:
    prompt = f"""
# 角色
你是一位资深的CST治疗小组剧本作家。
# 任务
请为一场CST小组活动创造 {num_patients} 位虚拟的老年参与者。每一位参与者都需要有独特且具体的背景、健康状况和鲜明的大五人格特质。
# 输出格式 (必须严格遵守JSON格式，只返回JSON对象)
{{ "V1": "角色描述...", "V2": "角色描述...", "V3": "角色描述..." }}
# 示例
{{ "V1": "李婆婆, 82岁, 轻度认知障碍, 健谈但记忆力差。大五人格: 高外向性, 低责任心。", "V2": "陈伯伯, 78岁, 曾是教师, 比较严谨。大五人格: 低宜人性, 高责任心。" }}
"""
    response_str = _call_api_with_retry(prompt, model=PATIENT_GEN_MODEL, temp=1.0, max_tokens=2048)
    if response_str:
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            print(f"\n[错误] [{PATIENT_GEN_MODEL}] 生成虚拟患者时返回了无效JSON")
    return None

def get_main_generation_prompt(style_card: str, scene: Dict, patients: Dict, skeleton: List[Dict]) -> str:
    patient_str = "\n".join([f"- {pid}: {desc}" for pid, desc in patients.items()])
    skeleton_str = json.dumps(skeleton, ensure_ascii=False, indent=2)
    dialogue_len = random.randint(25, 35)

    return f"""
# 总指令
你是一个顶级的CST多人对话剧本改编与扩展专家，具备深刻的心理学洞察力。你的任务是根据一个真实的“对话骨架”作为流程模板，结合全新的“治疗师风格”、“治疗场景”和“虚拟参与者”信息，创作一段全新的、完整的、带有多维度标签的CST多人对话。

# --- 1. 核心输入模块 ---
## 1.1. 目标治疗师风格卡片 (必须严格遵守的核心指令)
{style_card}

## 1.2. 本次治疗场景
- **场景名称**: {scene['name']}
- **核心活动**: {scene['description']}

## 1.3. 新的虚拟参与者 (本次对话的主角)
{patient_str}

## 1.4. 对话骨架/流程模板 (作为流程参考，但长度不是限制)
{skeleton_str}

# --- 2. 任务与改编要求 ---
1.  **长度优先**: 你的首要目标是生成一个包含 **{dialogue_len}** 轮左右对话的完整剧本，轮次不能过短（低于20轮）。你必须在遵循骨架流程的基础上进行**充分的扩展和内容填充**来达到此长度，剧本应有完整性，哪怕超出 **{dialogue_len}** 轮，也不能戛然而止，除非超过太多（40轮）。
2.  **遵循骨架**: 新对话的整体流程、事件顺序和互动模式必须严格参考“对话骨架”。
3.  **风格迁移与角色扮演**: 治疗师(H)的发言必须符合其“风格卡片”，参与者(V1, V2...)的发言必须符合其人格设定。
4.  **内容创新**: 在遵循骨架流程的基础上，将“核心活动”和新角色的性格融入对话。
5.  **冲突消解**: 如果参与者(V1, V2...)之间发生言语或意见冲突，治疗师(H)需要运用原则和策略解决。
6.  **多维度标签生成**: 为每一句发言精确生成`emotion`, `strategy`, `principles`, `cognitive_state` 标签。
    - **情感标签 (`emotion`)**: JSON对象, 包含`type` (从`Joy`, `Sadness`, `Anger`, `Fear`, `Disgust`, `Surprise`, `Trust`, `Anticipation`中选) 和 `intensity` (0.0-1.0浮点数)。
    - **策略标签 (`strategy`)**:
      - **治疗师(H)**: JSON对象, 包含`type` (从`Question`, `Restatement or Paraphrasing`, `Reflection of Feelings`, `Self-disclosure`, `Affirmation and Reassurance`, `Providing Suggestions`, `Information`, `Others`中选) 和 `confidence` (0.0-1.0浮点数)。
      - **参与者**: `null`。
    - **原则标签 (`principles`)**:
      - **治疗师(H)**: JSON对象列表 `[]`, 每个对象包含`type` (`P-XX`格式) 和 `confidence` (0.0-1.0浮点数)。
      - **参与者**: `null`。
    - **认知状态 (`cognitive_state`)**:
      - **治疗师(H)**: `null`。
      - **参与者**: 一句话简述。

# --- 3. JSON输出格式 ---
{{
    "dialogue": [
        {{ 
            "turn_id": 1, "speaker_id": "H", "text": "...", 
            "emotion": {{"type": "Anticipation", "intensity": 0.7}}, 
            "strategy": {{"type": "Question", "confidence": 0.95}}, 
            "principles": [{{"type": "P-7", "confidence": 0.9}}, {{"type": "P-13", "confidence": 0.8}}],
            "cognitive_state": null
        }}
    ]
}}
"""

def call_generation_api(prompt: str) -> str:
    return _call_api_with_retry(prompt, model=MAIN_GEN_MODEL, temp=0.8, max_tokens=8192)

def generate_one_dialogue_robust(dialogue_id: int) -> bool:
    for attempt in range(TASK_RETRIES):
        try:
            skeleton_data = random.choice(all_skeletons)
            dialogue_skeleton = skeleton_data.get("dialogue_skeleton", [])
            
            if len(dialogue_skeleton) > SKELETON_MAX_LENGTH:
                start_index = random.randint(0, len(dialogue_skeleton) - SKELETON_TRUNCATE_TO)
                dialogue_skeleton = dialogue_skeleton[start_index : start_index + SKELETON_TRUNCATE_TO]

            num_participants = len(skeleton_data.get("speaker_map", {})) - 1
            if num_participants <= 0: continue

            patients = generate_virtual_patients(num_participants)
            if not isinstance(patients, dict):
                raise ValueError(f"生成虚拟患者时返回了错误的类型: {type(patients)}")

            scene_name, scene_desc = random.choice(cst_scenes_list)
            current_scene = {'name': scene_name, 'description': scene_desc}

            main_prompt = get_main_generation_prompt(therapist_style_card, current_scene, patients, dialogue_skeleton)
            
            response_str = call_generation_api(main_prompt)
            if response_str is None:
                raise ConnectionError("主生成API调用失败或超时，返回了None")

            try:
                response_json = json.loads(response_str)
            except json.JSONDecodeError as json_err:
                raise ValueError(f"主生成API返回了无效的JSON。错误: {json_err}. 原始文本: {response_str[:200]}...")

            dialogue_data = response_json.get("dialogue", [])
            if not isinstance(dialogue_data, list):
                raise TypeError("返回的JSON中'dialogue'字段不是一个列表")
            
            output_data = {
                "synthetic_id": dialogue_id,
                "source_skeleton": skeleton_data.get("session_id"),
                "cst_scene": current_scene,
                "virtual_participants": patients,
                "dialogue": dialogue_data
            }
            
            output_path = OUTPUT_DIR / f"synthetic_dialogue_{dialogue_id}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            
            return True

        except Exception as e:
            print(f"\n[任务失败] ID {dialogue_id} 在尝试 {attempt+1}/{TASK_RETRIES} 时失败: {e}")
            if attempt < TASK_RETRIES - 1:
                time.sleep((attempt + 1) * 5)
            else:
                print(f"[任务放弃] ID {dialogue_id} 在重试 {TASK_RETRIES} 次后最终失败。")
    
    return False

def main():
    style_card = load_style_card()
    scenes = list(get_cst_scenes().items())
    skeletons = load_all_skeletons(SKELETON_DIR)
    
    if not skeletons:
        print("[错误] 未能加载任何对话骨架，程序终止。")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_task_ids = set(range(1, int(NUM_DIALOGUES_TO_GENERATE) + 1))

    existing_ids = set()
    for f in OUTPUT_DIR.glob("synthetic_dialogue_*.json"):
        match = re.search(r'_(\d+)\.json$', f.name)
        if match:
            existing_ids.add(int(match.group(1)))

    tasks_to_run = sorted(list(all_task_ids - existing_ids))
    
    print(f"总目标: {int(NUM_DIALOGUES_TO_GENERATE)} 份, 已存在: {len(existing_ids)} 份, 本次需生成: {len(tasks_to_run)}")
    if not tasks_to_run:
        print("所有文件均已生成，无需额外操作。程序退出。")
        return

    pool_initargs = (KEY, BASE_URL, style_card, scenes, skeletons)
    
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=initialize_worker, initargs=pool_initargs) as pool:
        results = list(tqdm(pool.imap_unordered(generate_one_dialogue_robust, tasks_to_run), total=len(tasks_to_run), desc="生成对话文件"))

    success_count = sum(1 for r in results if r)
    print(f"本次成功生成: {success_count} 份")
    print(f"当前总计: {len(existing_ids) + success_count} / {NUM_DIALOGUES_TO_GENERATE} 份")
    print(f"结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()