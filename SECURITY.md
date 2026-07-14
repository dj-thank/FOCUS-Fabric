# Security Policy

This is a research prototype and has no production security warranty.

Report vulnerabilities privately to the eventual repository maintainer rather than publishing exploit details immediately. Before a public repository URL is assigned, no monitored disclosure address exists; the publisher must replace this section.

High-risk areas include exact-archive confidentiality, tool-output injection into semantic memory, Codex candidate code execution, malformed benchmark backends, Triton bounds/numerics, artifact tampering, and tenant isolation.

See `docs/THREAT_MODEL.md`. Never run autonomous execute mode with valuable credentials or unrestricted host access. Use a disposable VM/container, read-only evaluators, network restrictions, and separate release-signing keys.
