#AImers 8기 최종코드

import os, shutil, random, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from datasets import load_dataset

BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# 저장 디렉토리
OUT_DIR = "./result"

SEED = 42
NUM_SAMPLES = 128
MAX_SEQ = 1024
MAP_BATCH_SIZE = 4
#128 4 유지

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError()

config = AutoConfig.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    config=config,
    torch_dtype=torch.float16,
    device_map={"": 0},  # 로컬 CUDA 사용
    trust_remote_code=True
)

ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train", trust_remote_code=True)

total_rows = len(ds)
step = max(total_rows // NUM_SAMPLES, 1)
indices = [random.randint(i * step, min((i + 1) * step - 1, total_rows - 1)) for i in range(NUM_SAMPLES)]

ds_calib = ds.select(indices)

del ds
gc.collect() #메모리즉시반환

def preprocess(examples):
    texts = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True  #대회 평가 환경과 동일하게 답변 생성 토큰 활성화
        )
        for conv in examples["conversations"]
    ]
    return {"text": texts}

ds_calib = ds_calib.map(
    preprocess,
    batched=True,
    batch_size=MAP_BATCH_SIZE,
    remove_columns=ds_calib.column_names
)

gc.collect()

#앞뒤로 2개씩 총 4개의 핵심 레이어를 보호하여 성능 하락 철통 방어
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8",
    ignore=[
        "embed_tokens", "lm_head",
        "model.layers.0", "model.layers.1",
        "model.layers.22", "model.layers.23"
    ]
)

oneshot(
    model=model,
    dataset=ds_calib,
    recipe=recipe,
    max_seq_length=MAX_SEQ,
    num_calibration_samples=NUM_SAMPLES
)

#저장 부분을 두 번째 코드처럼 변경 (zip/랩퍼 제거 + safetensors 체크/로그)
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUT_DIR)

st_path = os.path.join(OUT_DIR, "model.safetensors")
if not os.path.exists(st_path):
    raise FileNotFoundError(f"model.safetensors가 생성되지 않았습니다: {st_path}")

print("[DONE] Saved to:", OUT_DIR)
print("safetensors:", st_path)