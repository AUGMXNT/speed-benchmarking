from contextlib import contextmanager
from faster_whisper import WhisperModel
import time


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
    segments, info = model.transcribe(input_path, language="ja", beam_size=5)

    # We force it to 1.0 ja
    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

### Load Model
with timing_context("Loading Model"):
    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Transcription
transcribe("japanese-phrases/hai.wav")
transcribe("japanese-phrases/konichiwa.wav")
transcribe("japanese-phrases/konichiwa.ogenkidesuka.wav")
transcribe("japanese-phrases/kumbawa.wav")
transcribe("japanese-phrases/sosososo.wav")
transcribe("japanese-phrases/sumimasen.wav")
transcribe("japanese-phrases/prompt.mp3")
