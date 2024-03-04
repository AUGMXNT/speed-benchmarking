export LD_LIBRARY_PATH=/workspace/miniforge3/lib/python3.10/site-packages/torch/lib

for i in {1..5}; do python test-hf-pipe.py | grep DEBUG; sleep 1; done
for i in {1..5}; do python test-whisperx.py | grep DEBUG; sleep 1; done
for i in {1..5}; do python test-fasterwhisper.py | grep DEBUG; sleep 1; done
