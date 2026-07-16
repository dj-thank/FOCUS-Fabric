from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TERMINOLOGY_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "FOCUS_LINEAGE.md",
)


def test_public_docs_use_the_canonical_focus_operator_name() -> None:
    for path in PUBLIC_TERMINOLOGY_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "旧FOCUS" not in text, path.relative_to(ROOT)
        assert "旧実装" not in text, path.relative_to(ROOT)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "FOCUS-Native由来の解析的局所attention応答作用素" in readme
    assert "FOCUS作用素" in readme
