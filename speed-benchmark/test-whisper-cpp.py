from   datasets import load_dataset
import logging
from   loguru import logger
import os
from   pprint import pprint
import requests
import soundfile as sf
import tempfile
import time

from mytimer import TimeIt

# Whisper.cpp API endpoint
API_URL = "http://127.0.0.1:8001/inference"

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
    sf.write(tmpfile.name, sample['array'], sample['sampling_rate'])
    tmpfile_path = tmpfile.name

print(tmpfile_path)

for i in range(5):
    with TimeIt('SRT'):
        with open(tmpfile_path, 'rb') as audio_file:
            files = {'file': audio_file}
            response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            result = response.json()
        else:
            result = f"Error: {response.status_code}\n{response.text}"
    pprint(result)
    time.sleep(1)

os.remove(tmpfile_path)

logger.info(f"SRT average: {TimeIt.get_mean('SRT'):.1f} ms")
logger.info(f"SRT median: {TimeIt.get_median('SRT'):.1f} ms")
