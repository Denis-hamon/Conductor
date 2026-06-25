"""Tests for RAGVerifier — citation verification."""

import pytest

from runtime.internal.verifier.rag.rag import RAGVerifier, RAGResult


@pytest.fixture
def verifier():
    return RAGVerifier()


class TestRAGVerifier:
    def test_no_sources_returns_na(self, verifier):
        result = verifier.verify("Some response", sources=[])
        assert result.score == 0.0
        assert result.na is True

    def test_no_citations_returns_na(self, verifier):
        result = verifier.verify("Response without citations", sources=["doc-1"])
        assert result.score == 0.0
        assert result.na is True

    def test_all_citations_valid(self, verifier):
        response = "According to [source: doc-1] the policy applies."
        result = verifier.verify(response, sources=["doc-1"])
        assert result.score == 1.0
        assert result.verified == 1
        assert result.total == 1

    def test_mixed_valid_and_invalid_citations(self, verifier):
        response = "Claim A [source: doc-1] and claim B [source: doc-fake]."
        result = verifier.verify(response, sources=["doc-1"])
        assert result.score == 0.5
        assert result.verified == 1
        assert result.total == 2
        assert len(result.unverified) == 1

    def test_hallucinated_source_detected(self, verifier):
        response = "Claim [source: doc-nonexistent]."
        result = verifier.verify(response, sources=["doc-1", "doc-2"])
        assert result.score == 0.0
        assert result.verified == 0
        assert len(result.unverified) == 1
        assert "doc-nonexistent" in result.unverified[0]

    def test_citation_with_passage(self, verifier):
        response = 'Claim [source: doc-1, passage: "Section 4.2"]'
        result = verifier.verify(response, sources=["doc-1"])
        assert result.score == 1.0
        assert result.verified == 1

    def test_multiple_citations_to_same_source(self, verifier):
        response = "A [source: doc-1]. B [source: doc-1]. C [source: doc-1]."
        result = verifier.verify(response, sources=["doc-1"])
        assert result.score == 1.0
        assert result.verified == 3

    def test_rag_result_to_dict(self):
        result = RAGResult(score=0.5, verified=1, total=2, unverified=["fake-source"], na=False)
        d = result.to_dict()
        assert d["score"] == 0.5
        assert d["verified"] == 1
        assert d["total"] == 2
        assert d["na"] is False
