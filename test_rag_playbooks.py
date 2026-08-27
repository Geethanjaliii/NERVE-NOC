import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_faiss_indexes_exist():
    """Verify all required FAISS indexes and pickle mappings exist on disk."""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    required = [
        "faiss_playbooks.index",
        "playbook_summaries.pkl",
        "faiss_devices.index",
        "device_summaries.pkl",
        "faiss_index.index",
        "location_summaries.pkl",
    ]
    for f in required:
        path = os.path.join(models_dir, f)
        assert os.path.exists(path), f"Missing index file: {path}"
        assert os.path.getsize(path) > 0, f"Empty index file: {path}"
    print("[PASS] All FAISS index files exist and are non-empty.")


def test_rag_service_loads():
    """Verify RAGService initializes and loads all three corpora."""
    from src.rag_service import RAGService
    rag = RAGService()
    assert rag.index is not None, "Location FAISS index not loaded"
    assert rag.device_index is not None, "Device FAISS index not loaded"
    assert rag.playbook_index is not None, "Playbook FAISS index not loaded"
    assert len(rag.playbook_summaries) > 0, "Playbook summaries list is empty"
    print(f"[PASS] RAGService loaded: {len(rag.playbook_summaries)} playbooks indexed.")


def test_retrieve_playbooks_returns_results():
    """Verify playbook retrieval returns relevant SOP documents for a resource query."""
    from src.rag_service import RAGService
    rag = RAGService()
    
    results = rag.retrieve_playbooks("fiber transceiver optical signal loss resource_type 8", k=3)
    assert isinstance(results, list), "retrieve_playbooks must return a list"
    assert len(results) > 0, "retrieve_playbooks returned 0 results"
    assert len(results) <= 3, f"retrieve_playbooks returned {len(results)} results, expected <= 3"
    
    # Each result should be a dict with standard fields
    for r in results:
        assert isinstance(r, dict), f"Each playbook result should be a dict, got {type(r)}"
        assert 'title' in r or 'summary' in r, f"Playbook result missing 'title' or 'summary': {list(r.keys())}"
    
    print(f"[PASS] retrieve_playbooks returned {len(results)} relevant SOPs.")


def test_retrieve_playbooks_empty_index_fallback():
    """Verify retrieve_playbooks returns empty list gracefully when no index."""
    from src.rag_service import RAGService
    rag = RAGService()
    # Temporarily disable playbook index
    original_index = rag.playbook_index
    rag.playbook_index = None
    
    results = rag.retrieve_playbooks("anything", k=3)
    assert results == [], "Should return empty list when playbook_index is None"
    
    rag.playbook_index = original_index
    print("[PASS] retrieve_playbooks gracefully returns [] when index is None.")


def test_cross_reference_fault_integration():
    """Verify cross_reference_fault combines device and playbook retrieval."""
    from src.rag_service import RAGService
    rag = RAGService()
    
    result = rag.cross_reference_fault(
        location="location 684",
        active_resources=["resource_type 8"],
        active_events=["event_type 15"],
        shap_factors=[("log_feature 82", 0.45), ("resource_resource_type 8", 0.32)],
        k_devices=3,
        k_playbooks=3
    )
    
    assert isinstance(result, dict), "cross_reference_fault must return a dict"
    assert 'historical_devices' in result, "Missing 'historical_devices' key"
    assert 'matching_playbooks' in result, "Missing 'matching_playbooks' key"
    assert 'query_constructed' in result, "Missing 'query_constructed' key"
    assert isinstance(result['historical_devices'], list), "'historical_devices' must be a list"
    assert isinstance(result['matching_playbooks'], list), "'matching_playbooks' must be a list"
    assert len(result['historical_devices']) > 0, "No historical devices retrieved"
    assert len(result['matching_playbooks']) > 0, "No matching playbooks retrieved"
    
    print(f"[PASS] cross_reference_fault returned {len(result['historical_devices'])} devices, {len(result['matching_playbooks'])} playbooks.")
    print(f"       Query: {result['query_constructed']}")


def test_retrieval_latency():
    """Verify playbook retrieval completes in under 500ms."""
    from src.rag_service import RAGService
    rag = RAGService()
    
    t0 = time.time()
    _ = rag.retrieve_playbooks("high volume log surge resource_type 2 event_type 11", k=3)
    elapsed_ms = (time.time() - t0) * 1000
    
    assert elapsed_ms < 500, f"Playbook retrieval took {elapsed_ms:.1f}ms, expected < 500ms"
    print(f"[PASS] Playbook retrieval completed in {elapsed_ms:.1f}ms.")


if __name__ == "__main__":
    tests = [
        test_faiss_indexes_exist,
        test_rag_service_loads,
        test_retrieve_playbooks_returns_results,
        test_retrieve_playbooks_empty_index_fallback,
        test_cross_reference_fault_integration,
        test_retrieval_latency,
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
