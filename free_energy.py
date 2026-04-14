"""
Enhanced Expected Free Energy (EFE) Engine
==========================================
Implements Friston's Active Inference framework for autonomous agents.

Mathematical Foundation:
------------------------
G(π) = E_Q(o|π)[D_KL[Q(s|o,π) || P(s|C)] - ln P(o|C)]
     = Risk (pragmatic value) + Ambiguity (epistemic value)

Where:
- G(π): Expected Free Energy of policy π
- Q(s|o,π): Posterior beliefs about states given observations under policy π
- P(s|C): Preferred states (goal/prior preferences)
- P(o|C): Preferred observations
- D_KL: Kullback-Leibler divergence
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EFEBreakdown:
    """Structured breakdown of EFE components."""
    total_efe: float
    risk: float
    ambiguity: float
    is_acceptable: bool
    risk_components: Dict[str, float]
    ambiguity_components: Dict[str, float]
    threshold: float
    
    def __str__(self) -> str:
        return (
            f"EFE Analysis:\n"
            f"  Total EFE: {self.total_efe:.4f} (Threshold: {self.threshold:.4f})\n"
            f"  Risk: {self.risk:.4f}\n"
            f"    - Goal Divergence: {self.risk_components.get('goal_divergence', 0):.4f}\n"
            f"    - Constraint Violation: {self.risk_components.get('constraint_violation', 0):.4f}\n"
            f"  Ambiguity: {self.ambiguity:.4f}\n"
            f"    - State Uncertainty: {self.ambiguity_components.get('state_uncertainty', 0):.4f}\n"
            f"    - Observation Entropy: {self.ambiguity_components.get('observation_entropy', 0):.4f}\n"
            f"  Status: {'ACCEPTABLE' if self.is_acceptable else 'REJECTED - REPLAN REQUIRED'}"
        )


class ExpectedFreeEnergyEngine:
    """
    Computes Expected Free Energy (EFE) using information-theoretic metrics.
    
    This implementation uses:
    1. KL-Divergence for measuring distribution mismatch (Risk)
    2. Shannon Entropy for uncertainty quantification (Ambiguity)
    3. Cosine similarity for semantic alignment checks
    """
    
    def __init__(self, efe_threshold: float = 0.5):
        """
        Initialize the EFE engine.
        
        Args:
            efe_threshold: Maximum acceptable EFE score (lower = stricter)
        """
        self.efe_threshold = efe_threshold
        
        # Weighting factors for EFE components
        self.risk_weight = 1.0
        self.ambiguity_weight = 0.7
        
        # Sub-component weights
        self.goal_divergence_weight = 0.7
        self.constraint_violation_weight = 0.3
        self.state_uncertainty_weight = 0.6
        self.observation_entropy_weight = 0.4
        
    def calculate_kl_divergence(self, 
                               predicted_dist: np.ndarray, 
                               preferred_dist: np.ndarray) -> float:
        """
        Calculate KL-Divergence: D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
        
        Measures how much one probability distribution diverges from another.
        Lower values indicate better alignment with preferences.
        
        Args:
            predicted_dist: Predicted outcome distribution
            preferred_dist: Goal/preferred distribution
            
        Returns:
            KL-divergence value (0 = perfect match, higher = more divergence)
        """
        # Ensure valid probability distributions
        predicted_dist = np.asarray(predicted_dist) + 1e-10  # Avoid log(0)
        preferred_dist = np.asarray(preferred_dist) + 1e-10
        
        # Normalize to sum to 1
        predicted_dist = predicted_dist / np.sum(predicted_dist)
        preferred_dist = preferred_dist / np.sum(preferred_dist)
        
        # Calculate KL divergence
        kl_div = entropy(predicted_dist, preferred_dist)
        
        return float(kl_div)
    
    def calculate_shannon_entropy(self, distribution: np.ndarray) -> float:
        """
        Calculate Shannon Entropy: H(X) = -Σ P(x) * log(P(x))
        
        Measures uncertainty in a probability distribution.
        Higher values indicate more uncertainty/ambiguity.
        
        Args:
            distribution: Probability distribution
            
        Returns:
            Entropy value (0 = certain, higher = more uncertain)
        """
        dist = np.asarray(distribution) + 1e-10
        dist = dist / np.sum(dist)
        
        h = entropy(dist)
        return float(h)
    
    def vectorize_observation(self, observation: Dict) -> np.ndarray:
        """
        Convert observation dictionary into a numerical vector.
        
        This is a simplified implementation. In production, you would use:
        - Embeddings from sentence-transformers
        - Feature extraction from structured data
        - Multi-modal encodings
        
        Args:
            observation: Predicted observation dictionary
            
        Returns:
            Numerical vector representation
        """
        # Extract key features
        features = []
        
        # Feature 1: Success probability (if available)
        success_prob = observation.get('success_probability', 0.5)
        features.append(success_prob)
        
        # Feature 2: Risk indicator (if available)
        risk_indicator = observation.get('risk_level', 0.5)
        features.append(1.0 - risk_indicator)  # Invert so high risk = low value
        
        # Feature 3: Tool reliability
        tool = observation.get('tool', 'unknown')
        tool_reliability = self._get_tool_reliability(tool)
        features.append(tool_reliability)
        
        # Feature 4: Outcome sentiment (simplified)
        outcome_text = observation.get('predicted_outcome', '').lower()
        sentiment = self._simple_sentiment(outcome_text)
        features.append(sentiment)
        
        # Normalize to create probability-like distribution
        features = np.array(features)
        features = np.clip(features, 0.01, 0.99)  # Avoid extremes
        
        return features
    
    def _get_tool_reliability(self, tool_name: str) -> float:
        """
        Estimate tool reliability based on tool type.
        In production, this would be learned from execution history.
        """
        reliability_map = {
            'search': 0.85,
            'read': 0.90,
            'write': 0.75,
            'execute': 0.70,
            'query': 0.80,
            'calculate': 0.95,
            'unknown': 0.50
        }
        
        for key, value in reliability_map.items():
            if key in tool_name.lower():
                return value
        
        return 0.50
    
    def _simple_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis (positive words increase score).
        In production, use proper sentiment analysis models.
        """
        positive_words = ['success', 'complete', 'achieve', 'correct', 'valid', 'safe']
        negative_words = ['fail', 'error', 'danger', 'risk', 'invalid', 'unsafe']
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Normalize to 0-1 range
        sentiment = 0.5 + (pos_count - neg_count) * 0.1
        return np.clip(sentiment, 0.0, 1.0)
    
    def calculate_risk(self, 
                      predicted_observations: List[Dict],
                      preferred_distribution: Dict) -> Tuple[float, Dict]:
        """
        Calculate Risk (Pragmatic Value): D_KL[Q(s|o,π) || P(s|C)]
        
        Risk measures how much the predicted outcomes diverge from 
        the agent's preferred goal states.
        
        Args:
            predicted_observations: List of predicted outcome dictionaries
            preferred_distribution: Goal specification with outcomes and constraints
            
        Returns:
            Tuple of (risk_score, component_breakdown)
        """
        if not predicted_observations:
            return 1.0, {'goal_divergence': 1.0, 'constraint_violation': 0.0}
        
        # Extract preferences
        preferred_outcomes = preferred_distribution.get('outcomes', [])
        constraints = preferred_distribution.get('constraints', [])
        
        # Component 1: Goal Divergence using KL-Divergence
        goal_divergence = self._calculate_goal_divergence(
            predicted_observations, 
            preferred_outcomes
        )
        
        # Component 2: Constraint Violation
        constraint_violation = self._calculate_constraint_violation(
            predicted_observations,
            constraints
        )
        
        # Weighted combination
        risk_score = (
            self.goal_divergence_weight * goal_divergence +
            self.constraint_violation_weight * constraint_violation
        )
        
        return risk_score, {
            'goal_divergence': goal_divergence,
            'constraint_violation': constraint_violation
        }
    
    def _calculate_goal_divergence(self, 
                                  predictions: List[Dict],
                                  preferred_outcomes: List[str]) -> float:
        """
        Calculate divergence between predicted and preferred outcomes.
        """
        if not preferred_outcomes:
            return 0.3  # Mild penalty for no explicit goals
        
        # Vectorize predictions
        pred_vectors = [self.vectorize_observation(pred) for pred in predictions]
        
        if not pred_vectors:
            return 1.0
        
        # Create a target preference vector (simplified)
        # In production, use embeddings of preferred_outcomes text
        target_vector = np.array([0.8, 0.8, 0.85, 0.9])  # High success expectation
        
        # Calculate average predicted distribution
        avg_pred_vector = np.mean(pred_vectors, axis=0)
        
        # Compute KL divergence
        kl_div = self.calculate_kl_divergence(avg_pred_vector, target_vector)
        
        # Normalize to 0-1 range (KL can be unbounded)
        normalized_kl = 1.0 - np.exp(-kl_div)
        
        return float(normalized_kl)
    
    def _calculate_constraint_violation(self,
                                       predictions: List[Dict],
                                       constraints: List[str]) -> float:
        """
        Check if predicted actions violate any hard constraints.
        """
        if not constraints:
            return 0.0
        
        violation_count = 0
        
        for pred in predictions:
            outcome_text = pred.get('predicted_outcome', '').lower()
            tool = pred.get('tool', '').lower()
            
            for constraint in constraints:
                constraint_lower = constraint.lower()
                
                # Check for explicit violations
                if 'no' in constraint_lower or 'avoid' in constraint_lower:
                    forbidden_term = constraint_lower.replace('no ', '').replace('avoid ', '').strip()
                    if forbidden_term in outcome_text or forbidden_term in tool:
                        violation_count += 1
        
        # Normalize by number of predictions
        violation_ratio = violation_count / max(len(predictions), 1)
        
        return min(1.0, violation_ratio)
    
    def calculate_ambiguity(self, 
                           policy: List[Dict],
                           predicted_observations: List[Dict],
                           context: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Calculate Ambiguity (Epistemic Value): H[Q(s|π)]
        
        Ambiguity measures uncertainty about states and observations.
        Epistemic actions (search, query, read) reduce ambiguity.
        
        Args:
            policy: List of planned actions
            predicted_observations: Predicted outcomes of the policy
            context: The world model context (to check past epistemic steps)
            
        Returns:
            Tuple of (ambiguity_score, component_breakdown)
        """
        # Component 1: State Uncertainty (inherent uncertainty in the policy)
        state_uncertainty = self._calculate_state_uncertainty(policy, context)
        
        # Component 2: Observation Entropy (uncertainty in predicted outcomes)
        observation_entropy = self._calculate_observation_entropy(predicted_observations)
        
        # Weighted combination
        ambiguity_score = (
            self.state_uncertainty_weight * state_uncertainty +
            self.observation_entropy_weight * observation_entropy
        )
        
        return ambiguity_score, {
            'state_uncertainty': state_uncertainty,
            'observation_entropy': observation_entropy
        }
    
    def _calculate_state_uncertainty(self, policy: List[Dict], context: Optional[Dict] = None) -> float:
        """
        Calculate uncertainty based on action types.
        Epistemic actions (search, query) reduce uncertainty.
        """
        base_uncertainty = 0.5
        
        epistemic_tools = ['search', 'read', 'query', 'lookup', 'analyze', 'inspect', 'check', 'list']
        pragmatic_tools = ['write', 'execute', 'delete', 'send', 'modify', 'create']
        
        epistemic_count = 0
        pragmatic_count = 0
        
        # Consider actions in the proposed policy
        for action in policy:
            tool = action.get('tool', '').lower()
            if any(et in tool for et in epistemic_tools):
                epistemic_count += 1
            elif any(pt in tool for pt in pragmatic_tools):
                pragmatic_count += 1
                
        # Also consider past epistemic actions from context to lower current state uncertainty
        if context and 'completed_steps' in context:
            for step in context['completed_steps']:
                tool = step.get('tool', '').lower()
                if any(et in tool for et in epistemic_tools):
                    epistemic_count += 1
        
        # Epistemic actions reduce uncertainty
        uncertainty_reduction = epistemic_count * 0.15
        
        # Pragmatic actions without prior knowledge increase uncertainty
        if pragmatic_count > 0 and epistemic_count == 0:
            uncertainty_increase = 0.2
        else:
            uncertainty_increase = 0.0
        
        final_uncertainty = base_uncertainty - uncertainty_reduction + uncertainty_increase
        
        return np.clip(final_uncertainty, 0.0, 1.0)
    
    def _calculate_observation_entropy(self, predictions: List[Dict]) -> float:
        """
        Calculate entropy over predicted observations.
        """
        if not predictions:
            return 0.8  # High entropy if no predictions available
        
        # Extract outcome probabilities or create uniform distribution
        pred_vectors = [self.vectorize_observation(pred) for pred in predictions]
        
        if not pred_vectors:
            return 0.8
        
        # Calculate entropy of the average prediction distribution
        avg_vector = np.mean(pred_vectors, axis=0)
        obs_entropy = self.calculate_shannon_entropy(avg_vector)
        
        # Normalize entropy (max entropy for 4 features ≈ log(4) ≈ 1.386)
        normalized_entropy = obs_entropy / np.log(len(avg_vector))
        
        return float(normalized_entropy)
    
    def compute_efe(self,
                   policy: List[Dict],
                   predicted_observations: List[Dict],
                   preferences: Dict,
                   context: Optional[Dict] = None) -> EFEBreakdown:
        """
        Compute the total Expected Free Energy (EFE).
        
        G(π) = Risk + Ambiguity
        
        Args:
            policy: Sequence of planned actions
            predicted_observations: Predicted outcomes under the policy
            preferences: Goal specification from generative model
            context: Additional world model state (e.g. past completed steps)
            
        Returns:
            EFEBreakdown object with complete analysis
        """
        # Calculate Risk (Pragmatic value)
        risk_score, risk_components = self.calculate_risk(
            predicted_observations,
            preferences
        )
        
        # Calculate Ambiguity (Epistemic value)
        ambiguity_score, ambiguity_components = self.calculate_ambiguity(
            policy,
            predicted_observations,
            context
        )
        
        # Total EFE with weighting
        total_efe = (
            self.risk_weight * risk_score +
            self.ambiguity_weight * ambiguity_score
        )
        
        # Determine if plan is acceptable
        is_acceptable = total_efe < self.efe_threshold
        
        return EFEBreakdown(
            total_efe=total_efe,
            risk=risk_score,
            ambiguity=ambiguity_score,
            is_acceptable=is_acceptable,
            risk_components=risk_components,
            ambiguity_components=ambiguity_components,
            threshold=self.efe_threshold
        )


# Example usage for testing
if __name__ == "__main__":
    # Initialize EFE engine
    efe_engine = ExpectedFreeEnergyEngine(efe_threshold=0.5)
    
    # Example policy
    policy = [
        {"tool": "web_search", "args": {"query": "active inference"}},
        {"tool": "read_document", "args": {"doc_id": "123"}},
        {"tool": "write_summary", "args": {"content": "summary"}}
    ]
    
    # Example predictions
    predictions = [
        {
            "predicted_outcome": "Search returns 10 relevant articles on active inference",
            "success_probability": 0.85,
            "risk_level": 0.1,
            "tool": "web_search"
        },
        {
            "predicted_outcome": "Document contains detailed explanation",
            "success_probability": 0.75,
            "risk_level": 0.15,
            "tool": "read_document"
        },
        {
            "predicted_outcome": "Summary written successfully",
            "success_probability": 0.90,
            "risk_level": 0.05,
            "tool": "write_summary"
        }
    ]
    
    # Example preferences
    preferences = {
        "instruction": "Research active inference and write summary",
        "outcomes": [
            "Successfully research active inference",
            "Generate comprehensive summary"
        ],
        "constraints": [
            "No access to paid sources",
            "Avoid outdated information"
        ]
    }
    
    # Compute EFE
    result = efe_engine.compute_efe(policy, predictions, preferences)
    
    print(result)
    print(f"\n{'='*60}")
    print(f"Decision: {'EXECUTE POLICY' if result.is_acceptable else 'REPLAN REQUIRED'}")
    print(f"{'='*60}")
