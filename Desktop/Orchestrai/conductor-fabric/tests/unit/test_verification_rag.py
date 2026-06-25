"""ATDD: RAG citation verification tests (V-05, V-06, V-07).

FR-15: RAG verification avec citations
"""

import pytest


def check_citation(response: str, corpus: dict) -> dict:
    """Mock RAG citation checker. Real impl calls the RAG verifier service."""
    import re
    pattern = r'\[source:\s*([^,\]]+),\s*passage:\s*"([^"]+)"\]'
    matches = re.findall(pattern, response)
    results = {"verified": 0, "hallucinated": 0, "mismatch": 0, "unverified": 0}

    for source_id, passage in matches:
        if source_id not in corpus:
            results["hallucinated"] += 1
        elif passage not in corpus[source_id]:
            results["mismatch"] += 1
        else:
            results["verified"] += 1

    # Check for unverifiable claims
    claims = response.split(".")
    for claim in claims:
        if any(kw in claim.lower() for kw in ["is", "are", "was", "were", "has", "have"]):
            if "[source:" not in claim and len(claim.strip()) > 20:
                results["unverified"] += 1

    return results


class TestRAGCitationVerification:
    """V-05 / V-06: RAG citation accuracy."""

    @pytest.mark.atdd
    def test_correct_citation_passes(self):
        """Correct citation → score 1.0."""
        corpus = {"doc-123": ["Paris is the capital of France"]}
        response = 'The capital of France is Paris [source: doc-123, passage: "Paris is the capital of France"]'
        result = check_citation(response, corpus)
        assert result["verified"] >= 1
        assert result["hallucinated"] == 0

    @pytest.mark.atdd
    def test_hallucinated_source_detected(self):
        """Non-existent source → hallucinated."""
        corpus = {"doc-123": ["Paris is the capital of France"]}
        response = 'The population is 2M [source: doc-999, passage: "Paris has 2M inhabitants"]'
        result = check_citation(response, corpus)
        assert result["hallucinated"] >= 1

    @pytest.mark.atdd
    def test_wrong_passage_detected(self):
        """Correct source, wrong passage → mismatch."""
        corpus = {"doc-123": ["Paris is the capital of France"]}
        response = 'Paris has 2M inhabitants [source: doc-123, passage: "Paris has 2M inhabitants"]'
        result = check_citation(response, corpus)
        assert result["mismatch"] >= 1

    @pytest.mark.atdd
    def test_verifiable_claim_without_citation(self):
        """Claim without citation → unverified."""
        corpus = {}
        response = "The Eiffel Tower is 330 meters tall."
        result = check_citation(response, corpus)
        assert result["unverified"] >= 1

    @pytest.mark.atdd
    def test_mixed_citations(self):
        """Mixed correct + hallucinated → partial score."""
        corpus = {"doc-123": ["Paris is the capital"]}
        response = (
            'Paris is capital [source: doc-123, passage: "Paris is the capital"]. '
            'Population 2M [source: doc-999, passage: "fake"]'
        )
        result = check_citation(response, corpus)
        assert result["verified"] >= 1
        assert result["hallucinated"] >= 1
        total = result["verified"] + result["hallucinated"]
        score = result["verified"] / total if total > 0 else 0.0
        assert score == 0.5
