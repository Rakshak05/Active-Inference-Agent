"""
Standalone Active Inference Framework Demonstration
===================================================
Complete implementation demonstrating Free Energy minimization.
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import the enhanced free energy engine
from free_energy import ExpectedFreeEnergyEngine, EFEBreakdown


def demo_free_energy_scenarios():
    """Demonstrate different Free Energy scenarios."""
    print("\n" + "="*80)
    print("ACTIVE INFERENCE FRAMEWORK - FREE ENERGY DEMONSTRATIONS")
    print("="*80)
    
    efe_engine = ExpectedFreeEnergyEngine(efe_threshold=0.5)
    
    # ========================================================================
    # SCENARIO 1: Safe Research Task (Should Accept)
    # ========================================================================
    print("\n\n" + "─"*80)
    print("SCENARIO 1: Safe Information Gathering Task")
    print("─"*80)
    print("\nTask: 'Research active inference and write a summary'")
    
    policy_1 = [
        {"tool": "web_search", "args": {"query": "active inference"}},
        {"tool": "read_document", "args": {"doc_id": "paper_123"}},
        {"tool": "write_summary", "args": {"format": "markdown"}}
    ]
    
    predictions_1 = [
        {
            "predicted_outcome": "Search returns 10 relevant academic papers on active inference and free energy principle",
            "success_probability": 0.90,
            "risk_level": 0.05,
            "tool": "web_search"
        },
        {
            "predicted_outcome": "Document successfully read, containing detailed mathematical formulation of EFE",
            "success_probability": 0.85,
            "risk_level": 0.10,
            "tool": "read_document"
        },
        {
            "predicted_outcome": "Summary written with key concepts: variational inference, prediction errors, action selection",
            "success_probability": 0.95,
            "risk_level": 0.02,
            "tool": "write_summary"
        }
    ]
    
    preferences_1 = {
        "instruction": "Research active inference and write a summary",
        "outcomes": [
            "Find credible academic sources",
            "Understand mathematical foundations",
            "Create comprehensive summary"
        ],
        "constraints": [
            "Use only open-access sources",
            "Ensure accuracy"
        ]
    }
    
    result_1 = efe_engine.compute_efe(policy_1, predictions_1, preferences_1)
    
    print("\nEFE Analysis:")
    print(result_1)
    
    print("\nDecision:")
    if result_1.is_acceptable:
        print("   EXECUTE - Plan meets safety and goal alignment criteria")
        print("   → Proceeding to execution phase")
    else:
        print("   REJECT - Re-planning required")
    
    # ========================================================================
    # SCENARIO 2: High-Risk Destructive Action (Should Reject)
    # ========================================================================
    print("\n\n" + "─"*80)
    print("SCENARIO 2: High-Risk Destructive Action")
    print("─"*80)
    print("\nTask: 'Clean up old system files'")
    
    policy_2 = [
        {"tool": "system_delete", "args": {"path": "/system/critical"}},
        {"tool": "purge_cache", "args": {"force": True}}
    ]
    
    predictions_2 = [
        {
            "predicted_outcome": "Critical system files deleted without backup - potential system failure",
            "success_probability": 0.25,
            "risk_level": 0.95,
            "tool": "system_delete"
        },
        {
            "predicted_outcome": "Cache purged with uncertain impact on running processes",
            "success_probability": 0.40,
            "risk_level": 0.80,
            "tool": "purge_cache"
        }
    ]
    
    preferences_2 = {
        "instruction": "Clean up old system files",
        "outcomes": [
            "Safely remove unnecessary files",
            "Free up disk space"
        ],
        "constraints": [
            "No deletion of critical files",
            "Maintain system stability",
            "Avoid data loss"
        ]
    }
    
    result_2 = efe_engine.compute_efe(policy_2, predictions_2, preferences_2)
    
    print("\nEFE Analysis:")
    print(result_2)
    
    print("\nDecision:")
    if result_2.is_acceptable:
        print("   EXECUTE")
    else:
        print("   REJECT - High Expected Free Energy detected!")
        print("   → Primary issue: High Risk")
        print("   → Recommended action: Re-plan with safer approach")
        print("   → Suggestion: Add verification steps and use safer tools")
    
    # ========================================================================
    # SCENARIO 3: Ambiguous Plan (Lacks Information Gathering)
    # ========================================================================
    print("\n\n" + "─"*80)
    print("SCENARIO 3: Ambiguous Plan - Insufficient Information")
    print("─"*80)
    print("\nTask: 'Execute a financial transaction'")
    
    policy_3a = [
        {"tool": "execute_transaction", "args": {"amount": 5000, "recipient": "unknown"}}
    ]
    
    predictions_3a = [
        {
            "predicted_outcome": "Transaction executed with uncertain recipient verification",
            "success_probability": 0.50,
            "risk_level": 0.60,
            "tool": "execute_transaction"
        }
    ]
    
    preferences_3 = {
        "instruction": "Execute a financial transaction",
        "outcomes": [
            "Transaction completed successfully",
            "Funds transferred securely"
        ],
        "constraints": [
            "Verify recipient identity",
            "Ensure sufficient funds"
        ]
    }
    
    result_3a = efe_engine.compute_efe(policy_3a, predictions_3a, preferences_3)
    
    print("\nEFE Analysis (Without Epistemic Actions):")
    print(result_3a)
    
    # Now with epistemic actions
    policy_3b = [
        {"tool": "query_recipient", "args": {"recipient_id": "unknown"}},
        {"tool": "check_balance", "args": {"account": "primary"}},
        {"tool": "verify_transaction", "args": {"amount": 5000}},
        {"tool": "execute_transaction", "args": {"amount": 5000, "recipient": "verified"}}
    ]
    
    predictions_3b = [
        {
            "predicted_outcome": "Recipient verified as legitimate account",
            "success_probability": 0.85,
            "risk_level": 0.10,
            "tool": "query_recipient"
        },
        {
            "predicted_outcome": "Sufficient balance confirmed",
            "success_probability": 0.90,
            "risk_level": 0.05,
            "tool": "check_balance"
        },
        {
            "predicted_outcome": "Transaction validated against fraud patterns",
            "success_probability": 0.88,
            "risk_level": 0.08,
            "tool": "verify_transaction"
        },
        {
            "predicted_outcome": "Transaction executed successfully with confirmation",
            "success_probability": 0.92,
            "risk_level": 0.05,
            "tool": "execute_transaction"
        }
    ]
    
    result_3b = efe_engine.compute_efe(policy_3b, predictions_3b, preferences_3)
    
    print("\nEFE Analysis (With Epistemic Actions):")
    print(result_3b)
    
    print("\nComparison:")
    print(f"   Without epistemic actions:")
    print(f"     - Total EFE: {result_3a.total_efe:.4f}")
    print(f"     - Ambiguity: {result_3a.ambiguity:.4f} HIGH")
    print(f"     - Status: {'ACCEPT' if result_3a.is_acceptable else 'REJECT'}")
    
    print(f"\n   With epistemic actions (query, check, verify):")
    print(f"     - Total EFE: {result_3b.total_efe:.4f}")
    print(f"     - Ambiguity: {result_3b.ambiguity:.4f} ✓ REDUCED")
    print(f"     - Status: {'ACCEPT' if result_3b.is_acceptable else 'REJECT'}")
    
    print(f"\n   Ambiguity Reduction: {(result_3a.ambiguity - result_3b.ambiguity):.4f}")
    print("   Key Insight: Epistemic actions reduce uncertainty!")
    
    # ========================================================================
    # SCENARIO 4: Iterative Re-planning Simulation
    # ========================================================================
    print("\n\n" + "─"*80)
    print("SCENARIO 4: Simulating the Re-planning Loop")
    print("─"*80)
    print("\nTask: 'Deploy software update to production'")
    
    # Attempt 1: Naive direct deployment (will fail)
    print("\n--- Attempt 1: Direct Deployment ---")
    
    policy_4_attempt1 = [
        {"tool": "deploy", "args": {"target": "production", "force": True}}
    ]
    
    predictions_4_attempt1 = [
        {
            "predicted_outcome": "Deployment initiated without testing or validation",
            "success_probability": 0.40,
            "risk_level": 0.85,
            "tool": "deploy"
        }
    ]
    
    preferences_4 = {
        "instruction": "Deploy software update to production",
        "outcomes": [
            "Update deployed successfully",
            "No downtime",
            "All systems operational"
        ],
        "constraints": [
            "Test before deployment",
            "Have rollback plan",
            "Monitor system health"
        ]
    }
    
    result_4_attempt1 = efe_engine.compute_efe(
        policy_4_attempt1,
        predictions_4_attempt1,
        preferences_4
    )
    
    print(f"\nEFE Score: {result_4_attempt1.total_efe:.4f} (Threshold: {result_4_attempt1.threshold:.4f})")
    print(f"Status: {'ACCEPT ✓' if result_4_attempt1.is_acceptable else 'REJECT ✗'}")
    print(f"Issue: High risk + constraint violations detected")
    print(f"→ Agent triggers re-planning...")
    
    # Attempt 2: Add some testing (still not enough)
    print("\n--- Attempt 2: With Basic Testing ---")
    
    policy_4_attempt2 = [
        {"tool": "run_tests", "args": {"suite": "unit"}},
        {"tool": "deploy", "args": {"target": "production"}}
    ]
    
    predictions_4_attempt2 = [
        {
            "predicted_outcome": "Unit tests pass successfully",
            "success_probability": 0.80,
            "risk_level": 0.15,
            "tool": "run_tests"
        },
        {
            "predicted_outcome": "Deployment to production without staging verification",
            "success_probability": 0.60,
            "risk_level": 0.50,
            "tool": "deploy"
        }
    ]
    
    result_4_attempt2 = efe_engine.compute_efe(
        policy_4_attempt2,
        predictions_4_attempt2,
        preferences_4
    )
    
    print(f"\nEFE Score: {result_4_attempt2.total_efe:.4f} (Threshold: {result_4_attempt2.threshold:.4f})")
    print(f"Status: {'ACCEPT ✓' if result_4_attempt2.is_acceptable else 'REJECT ✗'}")
    print(f"Improvement: EFE reduced by {result_4_attempt1.total_efe - result_4_attempt2.total_efe:.4f}")
    print(f"Issue: Still lacks proper validation pipeline")
    print(f"→ Agent triggers re-planning again...")
    
    # Attempt 3: Comprehensive deployment pipeline (should succeed)
    print("\n--- Attempt 3: Full Deployment Pipeline ---")
    
    policy_4_attempt3 = [
        {"tool": "run_tests", "args": {"suite": "all"}},
        {"tool": "deploy_staging", "args": {"target": "staging"}},
        {"tool": "validate_staging", "args": {"checks": "comprehensive"}},
        {"tool": "create_rollback", "args": {"snapshot": True}},
        {"tool": "deploy_production", "args": {"strategy": "blue-green"}},
        {"tool": "monitor_health", "args": {"duration": "1h"}}
    ]
    
    predictions_4_attempt3 = [
        {
            "predicted_outcome": "All tests pass: unit, integration, e2e",
            "success_probability": 0.88,
            "risk_level": 0.08,
            "tool": "run_tests"
        },
        {
            "predicted_outcome": "Staging deployment successful",
            "success_probability": 0.90,
            "risk_level": 0.05,
            "tool": "deploy_staging"
        },
        {
            "predicted_outcome": "Staging validation confirms system stability",
            "success_probability": 0.85,
            "risk_level": 0.10,
            "tool": "validate_staging"
        },
        {
            "predicted_outcome": "Rollback snapshot created successfully",
            "success_probability": 0.95,
            "risk_level": 0.02,
            "tool": "create_rollback"
        },
        {
            "predicted_outcome": "Blue-green deployment minimizes downtime",
            "success_probability": 0.92,
            "risk_level": 0.05,
            "tool": "deploy_production"
        },
        {
            "predicted_outcome": "Health monitoring shows all systems operational",
            "success_probability": 0.88,
            "risk_level": 0.08,
            "tool": "monitor_health"
        }
    ]
    
    result_4_attempt3 = efe_engine.compute_efe(
        policy_4_attempt3,
        predictions_4_attempt3,
        preferences_4
    )
    
    print(f"\nEFE Score: {result_4_attempt3.total_efe:.4f} (Threshold: {result_4_attempt3.threshold:.4f})")
    print(f"Status: {'ACCEPT ✓' if result_4_attempt3.is_acceptable else 'REJECT ✗'}")
    print(f"Total Improvement: EFE reduced by {result_4_attempt1.total_efe - result_4_attempt3.total_efe:.4f}")
    
    print("\nRe-planning Summary:")
    print(f"   Attempt 1: EFE = {result_4_attempt1.total_efe:.4f} (Rejected - Too risky)")
    print(f"   Attempt 2: EFE = {result_4_attempt2.total_efe:.4f} (Rejected - Still risky)")
    print(f"   Attempt 3: EFE = {result_4_attempt3.total_efe:.4f} {'(Accepted!)' if result_4_attempt3.is_acceptable else '✗ (Rejected)'}")
    
    if result_4_attempt3.is_acceptable:
        print("\n   Plan validated - Proceeding to execution")
        print("   Agent successfully minimized Expected Free Energy")
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n\n" + "="*80)
    print("DEMONSTRATION COMPLETE - KEY INSIGHTS")
    print("="*80)
    
    print("\nMathematical Framework:")
    print("   G(π) = Risk + Ambiguity")
    print("   Risk = D_KL[Predicted || Preferred]  (Goal alignment)")
    print("   Ambiguity = H[Observations]          (Uncertainty)")
    
    print("\nActive Inference Principles Demonstrated:")
    print("   1. Plans with high EFE are rejected automatically")
    print("   2. Epistemic actions (search, query, verify) reduce ambiguity")
    print("   3. Constraint violations increase risk score")
    print("   4. Iterative re-planning minimizes Free Energy")
    print("   5. Only validated low-EFE plans proceed to execution")
    
    print("\nKey Advantages Over Reactive Agents:")
    print("   Proactive risk assessment before execution")
    print("   Automatic refinement when plans are unsafe")
    print("   Explicit uncertainty quantification")
    print("   Goal-directed behavior with safety constraints")
    print("   Mathematically grounded decision-making")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demo_free_energy_scenarios()
