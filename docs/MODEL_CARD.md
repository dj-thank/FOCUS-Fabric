# Model Card: FOCUS-Native Archived Mechanism Checkpoints

## Summary

The local research run used two small symbolic Transformer checkpoints archived from the same FOCUS-Native research series and restored for deterministic loading. Their configuration and evidence metadata are retained here, but the weight binaries are not redistributed in this public repository:

- `focus-native-small` — approximately one million parameters;
- `focus-native-memory-code` — the same base family with a memory-code path.

They exist to exercise Q/K/V tracing, exact-cache parity, page compilation, and end-to-end token-ID plumbing. They are not modern language models.

## Intended use

- cache mechanism unit/integration tests;
- learned Q/K/V trace collection;
- numerical exact-vs-approximate comparison;
- examples for FOCUS-Native loss and runtime adapters.

## Out-of-scope use

- natural-language assistant deployment;
- factual QA or safety evaluation;
- human-facing text generation claims;
- comparison with production-scale LLMs;
- benchmark leaderboard submission.

## Provenance

Weights were produced during the archived FOCUS-Native research series. The exported weights survived locally, but the original symbolic tokenizer mapping, complete training corpus/configuration, and a sufficient redistribution provenance record did not. For that reason, the public repository excludes `model.safetensors` while preserving architecture configuration, validation records, expected SHA-256 digests, loader code, and derived evidence.

A deterministic compatibility byte tokenizer is supplied to keep APIs complete. It is **not** semantically equivalent to the lost tokenizer, so text-level reproduction is impossible from these artifacts alone.

## Architecture

- vocabulary: 1,058;
- hidden width: 128;
- layers: 4;
- heads: 4;
- head dimension: 32;
- feed-forward width: 384;
- maximum configured sequence length: 2,048;
- FOCUS-Native patches/rank: 4/8.

## Current measured use

On committed random token-ID traces, the new Fabric path achieved 1.0000 argmax agreement across 64 teacher-forced positions and 1.0000 agreement across 8 greedy generated tokens. These are mechanism measurements, not language quality.

## Risks

Even small symbolic checkpoints can memorize training artifacts. Their training data is incompletely documented. Do not expose them as trustworthy text generators. Do not infer license compatibility of unknown original data from the Apache-2.0 code license.
