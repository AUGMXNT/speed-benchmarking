from contextlib import contextmanager
from pydub import AudioSegment
from seamless_communication.models.inference import Translator

import numpy as np
import torch
import torchaudio
import os
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


### Transcribe

@timing_decorator
def transcribe(input_path):
    # Load WAV - use pydub for other filetypes
    input_extension = os.path.splitext(input_path)[-1].lower()
    if input_extension != '.wav':
        # Convert to WAV using pydub (required for torchaudio)
        audio = AudioSegment.from_file(input_path)
        sample_rate = audio.frame_rate
        # convert to numpy array
        samples = np.array(audio.get_array_of_samples())

        # Convert to PyTorch tensor and reshape
        waveform = torch.tensor(samples).float()
        if audio.channels == 2:
            waveform = waveform.view(-1, 2)
        else:
            waveform = waveform.view(1, -1)
    else:
        waveform, sample_rate = torchaudio.load(input_path)

    # Perform resampling - needs to be 16K https://github.com/facebookresearch/seamless_communication/tree/main/scripts/m4t/predict#quick-start
    resample_rate = 16000
    resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    temp_path = "temp.wav"
    torchaudio.save(temp_path, resampled_waveform, resample_rate)

    # ASR - This is equivalent to S2TT with `<tgt_lang>=<src_lang>`.
    with timing_context(f"Transcribing {input_path}"):
        transcribed_text, _, _ = translator.predict(temp_path, "asr", "jpn")
    print(transcribed_text)

    # Delete the temporary WAV file
    os.remove(temp_path)


### Load Translator
with timing_context("Loading Model"):
    translator = Translator("seamlessM4T_large", "vocoder_36langs", torch.device("cuda:0"), torch.float16)

# Transcription
transcribe("japanese-phrases/hai.wav")
transcribe("japanese-phrases/konichiwa.wav")
transcribe("japanese-phrases/konichiwa.ogenkidesuka.wav")
transcribe("japanese-phrases/kumbawa.wav")
transcribe("japanese-phrases/sosososo.wav")
transcribe("japanese-phrases/sumimasen.wav")
transcribe("japanese-phrases/prompt.mp3")
