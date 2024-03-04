export LD_LIBRARY_PATH=/workspace/miniforge3/lib/python3.10/site-packages/torch/lib:/workspace/miniforge3/lib/python3.10/site-packages/nvidia/cudnn/lib

echo 'test-hf-pipe.py'
echo '---'
for i in {1..5}; do python test-hf-pipe.py | grep DEBUG; sleep 1; done
echo
echo 'test-whisperx.py'
echo '---'
for i in {1..5}; do python test-whisperx.py | grep DEBUG; sleep 1; done
echo 'test-fasterwhisper.py'
echo '---'
for i in {1..5}; do python test-fasterwhisper.py | grep DEBUG; sleep 1; done
