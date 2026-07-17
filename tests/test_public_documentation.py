from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TERMINOLOGY_DOCS = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "CHANGELOG.md",
    ROOT / "references" / "literature_2026-07.json",
    ROOT / "checkpoints" / "focus-native-small" / "metadata.json",
    ROOT / "checkpoints" / "focus-native-memory-code" / "metadata.json",
    ROOT / "src" / "focus_native" / "model.py",
    ROOT / "src" / "focus_fabric" / "training.py",
    *sorted((ROOT / "docs").glob("*.md")),
)
DEPRECATED_FOCUS_TERMS = (
    "旧FOCUS",
    "旧実装",
    "FOCUS作用素",
    "legacy FOCUS",
    "legacy operator",
    "legacy mechanism",
    "legacy-mechanism",
    "legacy-memory-code",
    "earlier FOCUS-Native prototype export",
    "earlier FOCUS-Native export",
    "FOCUS operators",
    "FOCUS-native",
)


def test_public_docs_use_the_canonical_focus_operator_name() -> None:
    documents = {
        path: path.read_text(encoding="utf-8") for path in PUBLIC_TERMINOLOGY_DOCS
    }
    for path, text in documents.items():
        for term in DEPRECATED_FOCUS_TERMS:
            assert term not in text, (path.relative_to(ROOT), term)

    readme = documents[ROOT / "README.md"]
    assert "FOCUS-Native由来の解析的局所attention応答作用素" in readme
    assert "FOCUS-Native作用素" in readme
