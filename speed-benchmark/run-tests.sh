export LD_LIBRARY_PATH=/workspace/miniforge3/lib/python3.10/site-packages/torch/lib:/workspace/miniforge3/lib/python3.10/site-packages/nvidia/cudnn/lib

echo 'test-hf-pipe.py'
echo '---'
python test-hf-pipe.py | grep INFO
echo
echo 'test-whisperx.py'
echo '---'
python test-whisperx.py | grep INFO
echo 'test-fasterwhisper.py'
echo '---'
python test-fasterwhisper.py | grep INFO
