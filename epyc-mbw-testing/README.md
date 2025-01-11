I have a recently built EPYC 9274F (24C 48T) w/ 12 x DDR5-4800 ECC memory.

This has a theoretical 460.8 GB/s of MBW, but real world performance was much slower.

See discussion: https://www.reddit.com/r/LocalLLaMA/comments/1hwthrq/comment/m69c0o0/

Chatting w/ Claude about this: https://claude.ai/chat/0beeb3b3-6ca6-4a0f-8fb6-f99dea416efc

TODO:
```
likwid-bench -t load -i 128 -w M0:8GB -w M1:8GB -w M2:8GB -w M3:8GB -w M4:8GB -w M5:8GB -w M6:8GB -w M7:8GB
```
