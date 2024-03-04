from   datasets import load_dataset
import logging
from   loguru import logger
import os
from   pprint import pprint
import soundfile as sf
import tempfile
import time
import whisperx

from mytimer import TimeIt

batch_size = 16
model_size = "large-v2"
model = whisperx.load_model(model_size, device="cuda", compute_type="float16")

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
    sf.write(tmpfile.name, sample['array'], sample['sampling_rate'])
    tmpfile_path = tmpfile.name

audio = whisperx.load_audio(tmpfile_path)
print(tmpfile_path)

for i in range(5):
    with TimeIt('SRT'):
        result = model.transcribe(audio, batch_size=batch_size)
    pprint(result)
    time.sleep(1)

os.remove(tmpfile_path)

logger.info(f"SRT average: {TimeIt.get_mean('SRT'):.1f} ms")
logger.info(f"SRT median: {TimeIt.get_median('SRT'):.1f} ms")
