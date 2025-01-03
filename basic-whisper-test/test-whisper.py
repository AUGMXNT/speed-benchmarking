import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import soundfile as sf
import time
from contextlib import contextmanager
import numpy as np

@contextmanager
def timing_context(label="Code block"):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{label} took {elapsed_time:.5f} seconds to run.")
    print()

def calculate_rtf(audio_duration, processing_time):
    return processing_time / audio_duration

def process_sample(pipe, sample, sample_id=None):

    # {'audio': {'path': '0d38672e0bbdbdc460af55b8bb84a15b2730db2819f2af64f9c777d4d586f2de', 'array': array([0.00238037, 0.0020752 , 0.00198364, ..., 0.00024414, 0.00048828, 0.0005188 ]), 'sampling_rate': 16000}}

    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    
    # Calculate audio duration
    audio_duration = len(audio_array) / sampling_rate
    print(audio_duration)
    
    # Construct the input directly from the array
    audio_input = {
        "array": audio_array,
        "sampling_rate": sampling_rate
    }

    # Time the inference
    start_time = time.time()
    result = pipe(audio_input, return_timestamps=True)
    inference_time = time.time() - start_time
    
    # Calculate RTF
    rtf = calculate_rtf(audio_duration, inference_time)
    
    # Print results
    print(f"Sample {sample_id if sample_id else ''}")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Audio size: {audio_array.shape}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Real-time factor: {rtf:.2f}x")
    print(f"Real-time X: {1/rtf:.2f}x")
    print(f"Transcription: {result['text']}")
    print("-" * 80)
    print()
    
    return {
        "duration": audio_duration,
        "inference_time": inference_time,
        "rtf": rtf,
        "text": result["text"]
    }

# Setup device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.xpu.is_available():
    device = "xpu" 
torch_dtype = torch.float16 if device != "cpu" else torch.float32

# Load model with timing
with timing_context("Loading Model"):
    model_id = "openai/whisper-large-v3-turbo"
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
        torch_dtype=torch_dtype,
        device=device,
    )

# Load dataset
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# dataset = load_dataset("Isma/librispeech_tiny", split="train[:10]")
# dataset = load_dataset("olympusmons/librispeech_asr_test_clean_word_timestamp", split="train[:10]")
# dataset = load_dataset("librispeech_asr", "clean", split="test[:10]")
# dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:10]")  # Let's start with 10 samples

# Process samples and collect metrics
results = []
for i, sample in enumerate(dataset):
    result = process_sample(pipe, sample, i+1)
    results.append(result)

# Print summary statistics
durations = [r["duration"] for r in results]
inference_times = [r["inference_time"] for r in results]
rtfs = [r["rtf"] for r in results]

print("Summary Statistics:")
print(f"Average audio duration: {np.mean(durations):.2f} seconds")
print(f"Average inference time: {np.mean(inference_times):.2f} seconds")
print(f"Average RTF: {np.mean(rtfs):.2f}x")
print(f"Min RTF: {min(rtfs):.2f}x")
print(f"Max RTF: {max(rtfs):.2f}x")
