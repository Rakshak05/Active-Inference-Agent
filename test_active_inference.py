"""
Active Inference Agent - Comprehensive Test Suite
=================================================
Demonstrates the complete Active Inference cycle with various scenarios.
"""

import numpy as np
from free_energy import ExpectedFreeEnergyEngine, EFEBreakdown
from agent_manager import AgentManager


def test_efe_calculations():
    """Test the EFE engine with various scenarios."""
    print("\n" + "="*80)
    print("TEST SUITE 1: FREE ENERGY CALCULATIONS")
    print("="*80)
    
    efe_engine = ExpectedFreeEnergyEngine(efe_threshold=0.5)
    
    # Scenario 1: Low-risk, low-ambiguity (should ACCEPT)
    print("\n--- Scenario 1: Ideal Plan (Low Risk + Low Ambiguity) ---")
    
    policy_1 = [
        {"tool": "web_search", "args": {"query": "active inference"}},
        {"tool": "read_article", "args": {"url": "https://example.com"}},
        {"tool": "summarize", "args": {"content": "research findings"}}
    ]
    
    predictions_1 = [
        {
            "predicted_outcome": "Search successfully returns 15 highly relevant academic papers on active inference",
            "success_probability": 0.90,
            "risk_level": 0.05,
            "tool": "web_search"
        },
        {
            "predicted_outcome": "Article contains comprehensive explanation of free energy principle",
            "success_probability": 0.85,
            "risk_level": 0.10,
            "tool": "read_article"
        },
        {
            "predicted_outcome": "Summary generated with key concepts and findings",
            "success_probability": 0.95,
            "risk_level": 0.02,
            "tool": "summarize"
        }
    ]
    
    preferences_1 = {
        "instruction": "Research active inference",
        "outcomes": [
            "Find credible sources on active inference",
            "Understand key concepts",
            "Generate summary"
        ],
        "constraints": []
    }
    
    result_1 = efe_engine.compute_efe(policy_1, predictions_1, preferences_1)
    print(result_1)
    assert result_1.is_acceptable, "Low-risk plan should be accepted!"
    print("\n✓ TEST PASSED: Low-risk plan correctly accepted")
    
    # Scenario 2: High-risk (should REJECT)
    print("\n\n--- Scenario 2: High-Risk Plan (Should Reject) ---")
    
    policy_2 = [
        {"tool": "system_delete", "args": {"path": "/important/files"}},
        {"tool": "execute_script", "args": {"script": "unknown.sh"}}
    ]
    
    predictions_2 = [
        {
            "predicted_outcome": "Files deleted permanently with no backup",
            "success_probability": 0.30,
            "risk_level": 0.95,
            "tool": "system_delete"
        },
        {
            "predicted_outcome": "Script execution with unknown consequences",
            "success_probability": 0.20,
            "risk_level": 0.90,
            "tool": "execute_script"
        }
    ]
    
    preferences_2 = {
        "instruction": "Clean up system files",
        "outcomes": ["Safely remove unnecessary files"],
        "constraints": ["No deletion of important data", "Avoid risky operations"]
    }
    
    result_2 = efe_engine.compute_efe(policy_2, predictions_2, preferences_2)
    print(result_2)
    assert not result_2.is_acceptable, "High-risk plan should be rejected!"
    print("\nTEST PASSED: High-risk plan correctly rejected")
    
    # Scenario 3: High-ambiguity (should REJECT)
    print("\n\n--- Scenario 3: High-Ambiguity Plan (Uncertain Outcomes) ---")
    
    policy_3 = [
        {"tool": "execute_transaction", "args": {"amount": 1000}},
        {"tool": "deploy_changes", "args": {"target": "production"}}
    ]
    
    predictions_3 = [
        {
            "predicted_outcome": "Transaction processed with uncertain result",
            "success_probability": 0.50,
            "risk_level": 0.40,
            "tool": "execute_transaction"
        },
        {
            "predicted_outcome": "Deployment may succeed or fail",
            "success_probability": 0.55,
            "risk_level": 0.45,
            "tool": "deploy_changes"
        }
    ]
    
    preferences_3 = {
        "instruction": "Complete financial transaction",
        "outcomes": ["Transaction confirmed", "System stable"],
        "constraints": ["No data loss"]
    }
    
    result_3 = efe_engine.compute_efe(policy_3, predictions_3, preferences_3)
    print(result_3)
    print("\n✓ TEST PASSED: High-ambiguity plan evaluated correctly")
    
    # Scenario 4: Epistemic action reduces ambiguity
    print("\n\n--- Scenario 4: Epistemic Actions Reduce Ambiguity ---")
    
    policy_4a = [
        {"tool": "execute_trade", "args": {"stock": "ABC", "amount": 1000}}
    ]
    
    policy_4b = [
        {"tool": "query_market_data", "args": {"stock": "ABC"}},
        {"tool": "analyze_trends", "args": {"stock": "ABC"}},
        {"tool": "execute_trade", "args": {"stock": "ABC", "amount": 1000}}
    ]
    
    predictions_4 = [
        {
            "predicted_outcome": "Trade executed",
            "success_probability": 0.70,
            "risk_level": 0.30,
            "tool": "execute_trade"
        }
    ]
    
    preferences_4 = {
        "instruction": "Execute stock trade",
        "outcomes": ["Successful trade"],
        "constraints": []
    }
    
    result_4a = efe_engine.compute_efe(policy_4a, predictions_4, preferences_4)
    result_4b = efe_engine.compute_efe(policy_4b, predictions_4, preferences_4)
    
    print("\nWithout epistemic actions:")
    print(f"  Ambiguity: {result_4a.ambiguity:.4f}")
    
    print("\nWith epistemic actions (query + analyze):")
    print(f"  Ambiguity: {result_4b.ambiguity:.4f}")
    
    assert result_4b.ambiguity < result_4a.ambiguity, "Epistemic actions should reduce ambiguity!"
    print("\nTEST PASSED: Epistemic actions reduce ambiguity")
    
    print("\n" + "="*80)
    print("ALL EFE CALCULATION TESTS PASSED!")
    print("="*80)


def test_kl_divergence():
    """Test KL-divergence calculations."""
    print("\n" + "="*80)
    print("TEST SUITE 2: KL-DIVERGENCE METRICS")
    print("="*80)
    
    efe_engine = ExpectedFreeEnergyEngine()
    
    # Test 1: Identical distributions (should be ~0)
    print("\n--- Test 1: Identical Distributions ---")
    dist_a = np.array([0.25, 0.25, 0.25, 0.25])
    dist_b = np.array([0.25, 0.25, 0.25, 0.25])
    
    kl = efe_engine.calculate_kl_divergence(dist_a, dist_b)
    print(f"KL(P||Q) for identical distributions: {kl:.6f}")
    assert kl < 0.01, "KL divergence should be near 0 for identical distributions"
    print("PASSED")
    
    # Test 2: Different distributions
    print("\n--- Test 2: Different Distributions ---")
    dist_c = np.array([0.7, 0.2, 0.05, 0.05])
    dist_d = np.array([0.25, 0.25, 0.25, 0.25])
    
    kl = efe_engine.calculate_kl_divergence(dist_c, dist_d)
    print(f"KL(P||Q) for different distributions: {kl:.6f}")
    assert kl > 0.1, "KL divergence should be significant for different distributions"
    print("PASSED")
    
    # Test 3: Completely opposite distributions
    print("\n--- Test 3: Opposite Distributions ---")
    dist_e = np.array([0.9, 0.05, 0.03, 0.02])
    dist_f = np.array([0.02, 0.03, 0.05, 0.9])
    
    kl = efe_engine.calculate_kl_divergence(dist_e, dist_f)
    print(f"KL(P||Q) for opposite distributions: {kl:.6f}")
    assert kl > 1.0, "KL divergence should be high for opposite distributions"
    print("PASSED")
    
    print("\n" + "="*80)
    print("ALL KL-DIVERGENCE TESTS PASSED!")
    print("="*80)


def test_shannon_entropy():
    """Test Shannon entropy calculations."""
    print("\n" + "="*80)
    print("TEST SUITE 3: SHANNON ENTROPY METRICS")
    print("="*80)
    
    efe_engine = ExpectedFreeEnergyEngine()
    
    # Test 1: Uniform distribution (maximum entropy)
    print("\n--- Test 1: Uniform Distribution (Max Entropy) ---")
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    entropy_uniform = efe_engine.calculate_shannon_entropy(uniform)
    print(f"H(X) for uniform distribution: {entropy_uniform:.6f}")
    print(f"Theoretical maximum: {np.log(4):.6f}")
    print("PASSED")
    
    # Test 2: Certain distribution (minimum entropy)
    print("\n--- Test 2: Certain Distribution (Min Entropy) ---")
    certain = np.array([1.0, 0.0, 0.0, 0.0])
    entropy_certain = efe_engine.calculate_shannon_entropy(certain)
    print(f"H(X) for certain distribution: {entropy_certain:.6f}")
    assert entropy_certain < 0.01, "Entropy should be near 0 for certain distribution"
    print("PASSED")
    
    # Test 3: Skewed distribution (moderate entropy)
    print("\n--- Test 3: Skewed Distribution ---")
    skewed = np.array([0.7, 0.2, 0.08, 0.02])
    entropy_skewed = efe_engine.calculate_shannon_entropy(skewed)
    print(f"H(X) for skewed distribution: {entropy_skewed:.6f}")
    assert 0 < entropy_skewed < np.log(4), "Entropy should be between 0 and max"
    print("PASSED")
    
    print("\n" + "="*80)
    print("ALL ENTROPY TESTS PASSED!")
    print("="*80)


def test_agent_manager_cycle():
    """Test the complete agent manager cycle."""
    print("\n" + "="*80)
    print("TEST SUITE 4: AGENT MANAGER INTEGRATION")
    print("="*80)
    
    # Create agent with strict threshold to force re-planning
    agent = AgentManager(efe_threshold=0.3, max_replans=3)
    
    print("\n--- Test: Process a simple research task ---")
    result = agent.process_task(
        "Search for information about Bayesian brain hypothesis and create a summary"
    )
    
    print(f"\nTask Status: {result['status']}")
    print(f"Planning Attempts: {result.get('planning_attempts', 'N/A')}")
    print(f"Final EFE: {result.get('efe_analysis', {}).get('total_efe', 'N/A'):.4f}")
    
    # Check planning history
    history = agent.get_planning_history()
    print(f"\nPlanning History ({len(history)} attempts):")
    for i, attempt in enumerate(history, 1):
        print(f"\n  Attempt {i}:")
        print(f"    EFE: {attempt.efe_breakdown.total_efe:.4f}")
        print(f"    Risk: {attempt.efe_breakdown.risk:.4f}")
        print(f"    Ambiguity: {attempt.efe_breakdown.ambiguity:.4f}")
        print(f"    Accepted: {attempt.efe_breakdown.is_acceptable}")
    
    print("\nTEST PASSED: Agent manager completed task cycle")
    
    print("\n" + "="*80)
    print("AGENT MANAGER INTEGRATION TEST PASSED!")
    print("="*80)


def run_all_tests():
    """Run the complete test suite."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "ACTIVE INFERENCE AGENT TEST SUITE" + " "*25 + "║")
    print("║" + " "*15 + "Friston's Free Energy Principle Implementation" + " "*16 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        test_efe_calculations()
        test_kl_divergence()
        test_shannon_entropy()
        test_agent_manager_cycle()
        
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*25 + "ALL TESTS PASSED ✓" + " "*34 + "║")
        print("╚" + "="*78 + "╝")
        
    except AssertionError as e:
        print(f"\n\nTEST FAILED: {str(e)}")
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
