# Archived checkpoint artifacts

The two historical FOCUS-Native weight binaries used for the committed mechanism evidence are intentionally **not redistributed** in this public GitHub repository. The original symbolic tokenizer, complete training-data record, and redistribution provenance were not preserved well enough for an unconditional public weight release.

The configuration, metadata, validation summaries, loader implementation, and derived benchmark evidence remain public. Authorized holders of the original local artifacts can verify them against the following identifiers:

| Local path | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/focus-native-small/model.safetensors` | 4,032,904 | `348e4d7699060add3a155b961e2998bcbf5ff071b14272ce8699e21507a0631a` |
| `checkpoints/focus-native-memory-code/model.safetensors` | 4,033,564 | `db8c5532eba3a6c23dbf7148bd262befcb8eadffc109da900b582e67f2486e3a` |

Place an authorized copy at the corresponding path to enable the two optional checkpoint-specific integration tests. All source-only algebraic, controller, training, semantic-memory, monitoring, and trace-capture tests run without these binaries.

The H001 autonomous evaluator also uses `focus-native-small` as a read-only local input. Its preflight and experiment contract require the exact SHA-256 above, pass the directory to candidate and trusted benchmark processes explicitly, and do not copy the binary into isolated Git worktrees or public evidence.

Do not upload replacement weights without documenting model origin, training-data governance, license compatibility, tokenizer revision, and cryptographic hashes.
