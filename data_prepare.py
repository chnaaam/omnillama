from datasets import load_dataset
import os
import json
from tqdm import tqdm
import soundfile as sf
import numpy as np

# 데이터셋 로드
dataset = load_dataset("Junhoee/STT_Korean_Dataset_80000", split="train")

# 디버깅을 위해 첫 번째 항목의 구조 확인
print("데이터 구조 확인:")
first_item = dataset[0]
print(first_item.keys())
print("오디오 데이터 타입:", type(first_item["audio"]))
print("오디오 데이터 키:", first_item["audio"].keys() if isinstance(first_item["audio"], dict) else "딕셔너리 아님")

# 다운로드 및 저장 디렉토리 생성
base_dir = "korean_stt_dataset"
audio_dir = os.path.join(base_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)

# 데이터 정보를 저장할 리스트
data_info = []

# 데이터셋 순회하며 오디오 데이터 추출 및 정보 수집
for idx, item in enumerate(tqdm(dataset, desc="데이터셋 처리 중")):
    # 고유한 파일명 생성
    wav_filename = f"audio_{idx}.wav"
    local_wav_path = os.path.join(audio_dir, wav_filename)
    
    # 오디오 데이터 저장
    try:
        # 데이터셋에서 오디오 데이터 추출
        audio_data = item["audio"]["array"]

        # 오디오 데이터 저장
        sf.write(local_wav_path, audio_data, samplerate=16000)

        # JSON에 저장할 데이터 정보
        data_entry = {
            "audio_path": local_wav_path,
            "text": item["transcripts"],
        }
        data_info.append(data_entry)
    except Exception as e:
        print(f"오디오 파일 처리 중 오류 발생: {e}")
        continue
    
    # 테스트를 위해 처음 몇 개만 처리하려면 주석 해제
    # if idx >= 10:
    #     break

# JSON Lines 파일로 저장
json_path = os.path.join(base_dir, "data_info.jsonl")  # 확장자를 .jsonl로 변경
with open(json_path, "w", encoding="utf-8") as f:
    for item in data_info:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"처리 완료: {len(data_info)}개 데이터")
print(f"오디오 파일 저장 위치: {audio_dir}")
print(f"JSON Lines 파일 저장 위치: {json_path}")

