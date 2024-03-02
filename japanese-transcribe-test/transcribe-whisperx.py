from contextlib import contextmanager
from pprint import pprint

import time
import whisperx


### Timing Tools 

@contextmanager
def timing_context(label="Code block"):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{label} took {elapsed_time:.5f} seconds to run.")
    print()


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.5f} seconds to run.")
        print()
        return result
    return wrapper


@timing_decorator
def transcribe(input_path):
    audio = whisperx.load_audio(input_path)
    result = model.transcribe(audio)
    pprint(result)


### Load Model
with timing_context("Loading Model"):
    model_size = "large-v2"
    model = whisperx.load_model(model_size, device="cuda", compute_type="float16", language="ja", task="transcribe")


# Transcription
transcribe("japanese-phrases/hai.wav")
transcribe("japanese-phrases/konichiwa.wav")
transcribe("japanese-phrases/konichiwa.ogenkidesuka.wav")
transcribe("japanese-phrases/kumbawa.wav")
transcribe("japanese-phrases/sosososo.wav")
transcribe("japanese-phrases/sumimasen.wav")
transcribe("japanese-phrases/prompt.mp3")
