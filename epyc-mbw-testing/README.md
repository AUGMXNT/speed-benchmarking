I have a recently built EPYC 9274F (24C 48T) w/ 12 x DDR5-4800 ECC memory.

This has a theoretical 460.8 GB/s of MBW, but real world performance was much slower.

See discussion: https://www.reddit.com/r/LocalLLaMA/comments/1hwthrq/comment/m69c0o0/

Chatting w/ Claude about this: https://claude.ai/chat/0beeb3b3-6ca6-4a0f-8fb6-f99dea416efc

TODO:
```
likwid-bench -t load -i 128 -w M0:8GB -w M1:8GB -w M2:8GB -w M3:8GB -w M4:8GB -w M5:8GB -w M6:8GB -w M7:8GB
```

Tuning reccomendations
- https://lenovopress.lenovo.com/lp1977-tuning-uefi-settings-4th-gen-amd-epyc-processor-servers
- https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/58002_amd-epyc-9004-tg-hpc.pdf
- https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/58011-epyc-9004-tg-bios-and-workload.pdf
- https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/other/56301-memory-population-guidelines-epyc-processors.pdf
- https://www.servethehome.com/how-to-populate-amd-epyc-9004-genoa-memory-channels/
- https://documentation.suse.com/sbp/tuning-performance/html/SBP-AMD-EPYC-4-SLES15SP4/index.html
- https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/white-papers/AMD-Optimizes-EPYC-Memory-With-NUMA.pdf


System Info and analysis...
- https://claude.ai/chat/b9aed79d-b11a-4aed-a81d-f3a6d478b9cd
- https://chatgpt.com/c/67866228-da20-8012-905d-b6f1fa5afc55
- https://aistudio.google.com/prompts/1qKDaY1oZ2j0dGVhSiumpd6t3DFBicBNc
