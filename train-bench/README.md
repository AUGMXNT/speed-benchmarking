# 2024-06-15 Testing of unsloth vs axolotl vs torchtune

Started trying to get [torchtune](https://github.com/pytorch/torchtune) working on ROCm (its own writeup), then used its test lora config (llama3-8b, [yahma/alpaca_cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset, rs=8 alpha=16) and tried to match it for [unsloth](https://github.com/unslothai/unsloth) and [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl).

Logs of runs here: https://wandb.ai/augmxnt/train-bench

More legible spreadsheet summary of results: https://docs.google.com/spreadsheets/d/115Utf5SuQOEOCQpYCeg_zB-nowej7joTkya6gYla5Hc/edit?usp=sharing

* On this particular test, for my hardware (3090, 4090) unsloth is 5-6X is faster than the 2X claimed speedups? https://github.com/unslothai/unsloth/tree/main#-performance-benchmarking - it is also able to fit batch_size=2 in 24GB while torchtune and axolotl OOM

* each trainer reports metrics differently into wandb and I couldn't figure out a way to normalize or group them easily, but most of the raw data is there

* torch.compile() gives about a 15% throughput benefit (and about a 10% benefit w/ the extra compilation time) - this is on a small 50K sample set and 1 epoch, so presumably you'll get closer and closer to getting the full benefit if you us torch.compile() (on axolotl theis setting is `torch_compile`)


General conclusions:

* If you have a supported Nvidia GPU and model and are tuning on a single GPU, it's definitely worth trying to use unsloth first

* I've been gravitating towards axolotl lately for its features, although it continues to be a bugfest and has a *lot* of dependencies. When updating my prior working version to the latest for example I had to fix a fastchat problem *and* it still barfed w/ another new error after the training run. Before that I had to delete an empty safetensors file for some reason (left over after it was split). torchtune has less features but may be more reliable. It is really lacking features though, you need to install the nightly build/from source if you want say sample packing. You'll probably end up having to write a lot more of your own trainer code w/ torchtune.
