
from   datasets import load_dataset
import logging
from   loguru import logger
import os
from   pprint import pprint
import soundfile as sf
import tempfile

from   faster_whisper import WhisperModel

from mytimer import TimeIt

model_size = "large-v2"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
    sf.write(tmpfile.name, sample['array'], sample['sampling_rate'])
    tmpfile_path = tmpfile.name

with TimeIt('SRT'):
    segments, info = model.transcribe(tmpfile_path, beam_size=5)

for segment in segments:
  print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

os.remove(tmpfile_path)
