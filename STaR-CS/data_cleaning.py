import os
import re
import json
import pysrt
from pathlib import Path

def parse_speaker_map_final(txt_path: Path) -> dict:
    speaker_map = {}
    try:
        with open(txt_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'^[HV][\d]*', line)
                
                if match:
                    speaker_id = match.group(0)
                    role_description = line[len(speaker_id):].lstrip(' \t:：;；').strip()
                    
                    if role_description:
                        speaker_map[speaker_id] = role_description
    except FileNotFoundError:
        print(f"[警告] 未找到对应的TXT文件: {txt_path}")
    except Exception as e:
        print(f"[错误] 解析TXT文件 '{txt_path.name}' 时出错: {e}")
    return speaker_map

def parse_and_structure_session(srt_path: Path, txt_path: Path, output_path: Path, input_dir: Path):
    speaker_map = parse_speaker_map_final(txt_path)
    if not speaker_map:
        print(f"[跳过] 因为无法从 '{txt_path.name}' 解析出角色信息，跳过文件: {srt_path.name}")
        return

    dialogue_turns = []
    try:
        subs = pysrt.open(str(srt_path), encoding='utf-8')
        
        turn_counter = 1
        for sub in subs:
            lines = sub.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.upper() == '[S]':
                    continue
                
                match = re.match(r'^([HV][\d]*)\s*[:：]\s*(.*)', line)
                if match:
                    speaker_id = match.group(1).strip()
                    text_cantonese = match.group(2).strip()
                    
                    dialogue_turns.append({
                        "turn_id": turn_counter,
                        "timestamp": f"{sub.start} --> {sub.end}",
                        "speaker_id": speaker_id,
                        "speaker_role": speaker_map.get(speaker_id, f"Unknown ID: {speaker_id}"),
                        "text_cantonese": text_cantonese
                    })
                    turn_counter += 1
                else:
                    if dialogue_turns:
                        dialogue_turns[-1]["text_cantonese"] += f" {line}"
                    else:
                        print(f"[警告] 在文件 {srt_path.name} 中找到无归属的行: {line}")
    except Exception as e:
        print(f"[错误] 处理SRT文件 '{srt_path.name}' 时出错: {e}")
        return
        
    session_data = {
        "session_id": srt_path.stem,
        "source_files": {
            "srt": str(srt_path.relative_to(input_dir)),
            "txt": str(txt_path.relative_to(input_dir))
        },
        "speaker_map": speaker_map,
        "dialogue": dialogue_turns
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False, indent=4)
    print(f"已成功处理并保存: {output_path.name}")

def main():
    input_dir = Path("Data_Set_25")
    output_dir = Path("output/1_structured_data")
    
    if not input_dir.exists():
        print(f"[错误] 输入目录不存在: {input_dir}")
        return
    
    srt_files = list(input_dir.rglob('*.srt'))
    
    if not srt_files:
        print("[警告] 在输入目录中未找到任何 .srt 文件。")
        return

    processed_count = 0
    total_files = len(srt_files)
    for srt_path in srt_files:
        txt_path = srt_path.with_suffix('.txt')
        relative_path = srt_path.relative_to(input_dir)
        output_json_path = (output_dir / relative_path).with_suffix('.json')
        
        if txt_path.exists():
            parse_and_structure_session(srt_path, txt_path, output_json_path, input_dir)
            processed_count += 1
        else:
            print(f"[跳过] 找不到与 {srt_path.name} 对应的 .txt 文件。")

    print(f"\n--- 第一阶段处理完成 ---")
    print(f"总共找到 {total_files} 个会话，成功处理了 {processed_count} 个。")
    print(f"结构化JSON文件已保存至: {output_dir}")

if __name__ == "__main__":
    main()