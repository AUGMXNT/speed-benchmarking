import logging
from   loguru import logger
import time
import torch

from   transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from   datasets import load_dataset

from mytimer import TimeIt


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

for i in range(5):
    sample = dataset[0]["audio"]
    with TimeIt('SRT'):
        result = pipe(sample)
    print(result["text"])
    time.sleep(1)

logger.info(f"SRT average: {TimeIt.get_mean('SRT'):.1f} ms")
logger.info(f"SRT median: {TimeIt.get_median('SRT'):.1f} ms")
