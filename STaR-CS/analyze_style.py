import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken

load_dotenv()
KEY = os.getenv("XXX")
BASE_URL="XXX"
MODEL_NAME = "deepseek-ai/deepseek-v3"

INPUT_DIR = Path("output/2_desensitized_final_test/") 
OUTPUT_FILE = Path("output/Therapist_Style_and_Personality_Analysis.md")

CHUNK_TARGET_SIZE_TOKENS = 120000 

if not KEY:
    raise ValueError("错误：请在项目根目录创建 .env 文件并添加 KEY='你的密钥'。")

client = OpenAI(api_key=KEY, base_url=BASE_URL)

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tokenizer = tiktoken.encoding_for_model("gpt-4")

def aggregate_therapist_speech(input_dir: Path) -> str:
    print(f"正在从 '{input_dir}' 目录中聚合治疗师(H)的发言...")
    therapist_lines = []
    json_files = list(input_dir.rglob('*.json'))
    
    if not json_files:
        print(f"[警告] 在目录 '{input_dir}' 中未找到任何 .json 文件。")
        return ""

    for file_path in tqdm(json_files, desc="读取文件"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for turn in data.get('dialogue', []):
                    if turn.get('speaker_id') == 'H':
                        text = turn.get('text_cantonese', '').strip()
                        if text:
                            therapist_lines.append(text)
        except json.JSONDecodeError:
            print(f"[警告] 文件格式错误，跳过: {file_path}")
        except Exception as e:
            print(f"[错误] 读取文件时出错 {file_path}: {e}")
            
    print(f"聚合完成！共找到 {len(therapist_lines)} 句治疗师发言。")
    return "\n".join(therapist_lines)

def split_corpus_into_chunks(corpus: str, chunk_size: int) -> list[str]:
    print("正在将语料库切分为多个处理块...")
    lines = corpus.split('\n')
    chunks = []
    current_chunk_lines = []
    current_chunk_tokens = 0

    for line in lines:
        line_tokens = len(tokenizer.encode(line))
        
        if current_chunk_tokens + line_tokens > chunk_size and current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_chunk_tokens = 0
        
        current_chunk_lines.append(line)
        current_chunk_tokens += line_tokens
    
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))
        
    print(f"切分完成！共得到 {len(chunks)} 个处理块。")
    return chunks

def get_chunk_analysis_prompt(chunk: str) -> str:
    return f"""
# 角色
你是一位专业的心理学和语言学分析师。

# 任务
请对以下CST治疗师的对话片段进行初步的、要点式的风格分析。请重点关注并列出：
1.  **语言风格**: 语调、常用词、句式特点。
2.  **核心治疗原则**: 在这个片段中，最明显的体现了CST的哪些原则？
3.  **人格特质线索**: 这段对话揭示了治疗师大五人格的哪些线索？

# 待分析的对话片段
--- START OF CHUNK ---
{chunk}
--- END OF CHUNK ---

# 输出要求
请以简洁的要点形式返回你的分析，这将作为后续汇总的材料。
"""

def get_final_summary_prompt(preliminary_analyses: list[str]) -> str:
    analyses_str = "\n\n---\n\n".join(preliminary_analyses)
    
    return f"""
# 角色
你是一位顶级的心理学和语言学专家，擅长从多个分散的分析报告中进行归纳、总结和升华，形成一份全面、深入、结构化的最终报告。

# 任务
你已经收到了对同一位CST治疗师全部对话内容分块进行的多份初步分析报告。现在，你的任务是**综合所有这些初步分析**，撰写一份唯一的、最终的、完整的《CST治疗师H风格与人格分析报告》。

# 所有初步分析报告汇总
{analyses_str}

# 最终报告结构 (请严格按照此Markdown结构输出)

## 1. 总体语言风格 (Linguistic Style)
- **语调与语气**: 整体给人的感觉是怎样的？(例如：温暖、鼓励、中立、幽默、引导性强)
- **常用句式**: 他/她最喜欢用什么样的句子结构？(例如：多用开放式提问如“大家觉得呢？”，还是多用确认性短句如“係啦。”)
- **口头禅/常用词**: 有没有一些高频词汇或短语？

## 2. CST原则应用分析 (Analysis of CST Principles Application)
请从CST的18项核心原则中，找出这位治疗师 **最常体现、最为擅长的3-5个原则**。对于每一个原则：
- **原则名称**: [例如：原则4 - 求意见，不求对错]
- **行为体现**: 具体分析他/她是 **如何** 通过语言和互动来实现这个原则的。
- **典型例句**: 从语料库中摘录1-2句最能代表该原则应用的典型发言。

## 3. 场景引导能力 (Scene Facilitation Skills)
分析他/她是如何在不同治疗场景中引导对话的。是否有独特的过渡技巧或引导模式？

## 4. 大五人格画像 (Big Five Personality Profile)
根据其言行，评估其在以下五个维度上的倾向（高/中/低），并为每个评估提供**详细的分析依据和例证**。
- **开放性 (Openness)**:
- **责任心 (Conscientiousness)**:
- **外向性 (Extraversion)**:
- **宜人性 (Agreeableness)**:
- **神经质 (Neuroticism)**:

## 5. 综合总结 (Overall Summary)
用一小段话总结这位治疗师的核心风格与特点。
"""

def call_api(prompt: str, is_final_summary: bool = False) -> str:
    try:
        max_tokens = 4096 if is_final_summary else 2048
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, 
            max_tokens=max_tokens
        )
        return response.choices[0].message.content or ""

    except Exception as e:
        print(f"\n[严重错误] 调用API时失败: {e}")
        return f"错误：API调用失败。\n{e}"
    
def main():
    if not INPUT_DIR.exists():
        print(f"[错误] 输入目录不存在: {INPUT_DIR}")
        return

    therapist_corpus = aggregate_therapist_speech(INPUT_DIR)
    if not therapist_corpus:
        print("未能聚合任何治疗师发言，程序终止。")
        return
        
    chunks = split_corpus_into_chunks(therapist_corpus, CHUNK_TARGET_SIZE_TOKENS)

    preliminary_analyses = []
    print("\n--- 开始对每个语料块进行初步分析 ---")
    for i, chunk in enumerate(tqdm(chunks, desc="分析语料块")):
        chunk_prompt = get_chunk_analysis_prompt(chunk)
        analysis = call_api(chunk_prompt)
        preliminary_analyses.append(f"## 初步分析报告 - {i+1}/{len(chunks)}\n\n{analysis}")
    
    print("\n所有语料块的初步分析已完成。")

    print("\n--- 正在汇总所有分析，生成最终报告 (这可能需要一些时间)... ---")
    final_prompt = get_final_summary_prompt(preliminary_analyses)
    final_report = call_api(final_prompt, is_final_summary=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_report)
        
    print(f"\n--- 分析完成 ---")
    print(f"最终分析报告已成功保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()