# Limitations

Read `WEAKNESS_AUDIT.md` for the full adversarial analysis. The highest-impact limitations are:

- no modern external pretrained language model was integrated;
- the archived checkpoint's original tokenizer mapping is missing;
- official long-context and agent benchmarks were not run;
- CUDA/Triton and physical HBM measurements were unavailable;
- the public runtime keeps an O(N) exact cold archive;
- the CPU implementation is slower than exact attention;
- split-conformal certificates lose nominal interpretation under query shift;
- global layer/head/device budget optimization is not implemented;
- repeated compaction evidence is only 128 tokens;
- semantic ledger tests structure, not reasoning ability;
- verified decode is greedy-only and the reference backend is unoptimized;
- autonomous Codex execute mode was not run in this container.

The package is suitable for mechanism research, extension, falsification, and systems integration work. It is not ready for production serving or capability claims.
