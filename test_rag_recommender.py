import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.recommender import recommend_action, verify_recommendations_with_rag


def test_healthy_status_skips_verification():
    """Healthy status should return high confidence without RAG query."""
    actions = recommend_action([], "Healthy")
    result = verify_recommendations_with_rag(
        baseline_actions=actions, shap_factors=[], status="Healthy"
    )
    assert result['confidence_score'] == 95.0, f"Expected 95.0, got {result['confidence_score']}"
    assert result['confidence_level'] == 'High'
    assert result['verification_method'] == 'baseline_healthy'
    print("[PASS] Healthy status returns high confidence without RAG.")


def test_no_rag_service_returns_low_confidence():
    """Without RAG service, should return low confidence with baseline actions."""
    actions = ["Investigate log/traffic volume spike"]
    result = verify_recommendations_with_rag(
        baseline_actions=actions,
        shap_factors=[("log_feature 82", 0.45)],
        status="Critical",
        rag_service=None
    )
    assert result['confidence_score'] == 25.0
    assert result['confidence_level'] == 'Low'
    assert result['verification_method'] == 'no_rag'
    assert result['verified_steps'] == actions
    print("[PASS] No RAG service returns low confidence with baseline actions.")


def test_local_similarity_verification_structure():
    """Verify the full verification pipeline returns correct structure with RAG."""
    from src.rag_service import RAGService
    rag = RAGService()
    
    baseline = recommend_action(
        [("log_feature 82", 0.45), ("resource_resource_type 8", 0.32)],
        "Critical"
    )
    
    result = verify_recommendations_with_rag(
        baseline_actions=baseline,
        shap_factors=[("log_feature 82", 0.45), ("resource_resource_type 8", 0.32)],
        status="Critical",
        location="location 684",
        active_resources=["resource_type 8"],
        active_events=["event_type 15"],
        rag_service=rag,
        llm_service=None  # Force local fallback
    )
    
    # Verify structure
    assert isinstance(result, dict), "Result must be a dict"
    required_keys = ['confidence_score', 'confidence_level', 'verified_steps', 
                     'citations', 'historical_precedents', 'verification_method', 'rag_query']
    for k in required_keys:
        assert k in result, f"Missing key: {k}"
    
    # Verify types
    assert isinstance(result['confidence_score'], float), "confidence_score must be float"
    assert 0 <= result['confidence_score'] <= 100, f"confidence_score out of range: {result['confidence_score']}"
    assert result['confidence_level'] in ('High', 'Moderate', 'Low', 'Insufficient')
    assert isinstance(result['verified_steps'], list)
    assert len(result['verified_steps']) > 0, "verified_steps must not be empty"
    assert isinstance(result['citations'], list)
    assert isinstance(result['historical_precedents'], list)
    assert result['verification_method'] == 'local_similarity'
    assert len(result['rag_query']) > 0, "rag_query must not be empty"
    
    print(f"[PASS] Local verification returned: confidence={result['confidence_score']}%, "
          f"level={result['confidence_level']}, steps={len(result['verified_steps'])}, "
          f"citations={len(result['citations'])}, precedents={len(result['historical_precedents'])}")


def test_citations_contain_sop_fields():
    """Verify citations contain correct SOP metadata fields."""
    from src.rag_service import RAGService
    rag = RAGService()
    
    result = verify_recommendations_with_rag(
        baseline_actions=["Investigate log volume spike"],
        shap_factors=[("log_feature 203", 0.5)],
        status="Warning",
        location="location 100",
        active_resources=["resource_type 2"],
        active_events=["event_type 11"],
        rag_service=rag,
        llm_service=None
    )
    
    assert len(result['citations']) > 0, "Should have at least one citation"
    for c in result['citations']:
        assert 'id' in c, "Citation missing 'id'"
        assert 'title' in c, "Citation missing 'title'"
        assert 'citation' in c, "Citation missing 'citation' reference"
        assert 'steps' in c, "Citation missing 'steps'"
        assert isinstance(c['steps'], list), "'steps' must be a list"
    
    print(f"[PASS] Citations contain SOP fields: {[c['id'] for c in result['citations']]}")


def test_verified_steps_include_sop_procedures():
    """Verify that local similarity merges SOP procedure steps into verified_steps."""
    from src.rag_service import RAGService
    rag = RAGService()
    
    baseline = ["Investigate log/traffic volume spike"]
    result = verify_recommendations_with_rag(
        baseline_actions=baseline,
        shap_factors=[("log_feature 82", 0.45)],
        status="Critical",
        location="location 684",
        active_resources=["resource_type 8"],
        active_events=["event_type 15"],
        rag_service=rag,
        llm_service=None
    )
    
    # Should have more steps than baseline (SOP steps merged)
    assert len(result['verified_steps']) > len(baseline), \
        f"Expected more steps than baseline ({len(baseline)}), got {len(result['verified_steps'])}"
    
    # Baseline actions should still be present
    for a in baseline:
        assert a in result['verified_steps'], f"Baseline action missing: {a}"
    
    print(f"[PASS] Verified steps ({len(result['verified_steps'])}) include merged SOP procedures.")


def test_confidence_score_bounds():
    """Verify confidence score is always bounded 0-100."""
    from src.rag_service import RAGService
    rag = RAGService()
    
    # Critical with matching resources should give decent score
    result = verify_recommendations_with_rag(
        baseline_actions=["Schedule immediate maintenance"],
        shap_factors=[("resource_resource_type 8", 0.6)],
        status="Critical",
        location="location 684",
        active_resources=["resource_type 8", "resource_type 2"],
        active_events=["event_type 15", "event_type 42"],
        rag_service=rag,
        llm_service=None
    )
    
    assert 0 <= result['confidence_score'] <= 100, f"Score out of bounds: {result['confidence_score']}"
    print(f"[PASS] Confidence score within bounds: {result['confidence_score']}%")


if __name__ == "__main__":
    tests = [
        test_healthy_status_skips_verification,
        test_no_rag_service_returns_low_confidence,
        test_local_similarity_verification_structure,
        test_citations_contain_sop_fields,
        test_verified_steps_include_sop_procedures,
        test_confidence_score_bounds,
    ]
    
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    if failed == 0:
        print(f"[PASS] ALL {len(tests)} TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"[FAIL] {failed}/{len(tests)} tests failed.")
        sys.exit(1)
