from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TERMINOLOGY_DOCS = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "docs" / "FOCUS_LINEAGE.md",
    ROOT / "docs" / "EVALUATION.md",
    ROOT / "docs" / "MODEL_CARD.md",
    ROOT / "docs" / "PAPER_DRAFT.md",
    ROOT / "docs" / "PUBLICATION_STATUS.md",
)
DEPRECATED_FOCUS_TERMS = (
    "旧FOCUS",
    "旧実装",
    "FOCUS作用素",
    "legacy FOCUS",
    "legacy operator",
    "legacy mechanism",
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
