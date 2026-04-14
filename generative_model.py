"""
The Generative Model for Active Inference.
Stores the agent's beliefs, current state, and the Preferred Distribution (goal).
"""
import copy

class AgentState:
    """Represents the current hidden state / beliefs of the agent."""
    def __init__(self, context=None):
        self.context = context or {}
        self.history = []
        self.known_facts = set()

    def update(self, observation):
        """Update beliefs based on new observation."""
        self.history.append(observation)
        # Update known facts or context
        if isinstance(observation, dict):
            for k, v in observation.items():
                self.context[k] = v
                self.known_facts.add(f"{k}: {v}")

class PreferredDistribution:
    """
    Mathematical representation of a successful task completion.
    This encodes the user's intent as a prior preference P(o).
    """
    def __init__(self, user_instruction: str):
        self.user_instruction = user_instruction
        self.expected_outcomes = []
        self.constraints = []
        
    def add_expected_outcome(self, outcome: str):
        self.expected_outcomes.append(outcome)
        
    def get_preferences(self):
        return {
            "instruction": self.user_instruction,
            "outcomes": self.expected_outcomes,
            "constraints": self.constraints
        }

class GenerativeModel:
    """
    Combines the current state (beliefs) with the preferred distribution (goals).
    """
    def __init__(self):
        self.current_state = AgentState()
        self.preferences = None

    def set_preference(self, user_prompt: str):
        """
        Takes the user's prompt and encodes it as the expected outcome (Prior Preference).
        """
        self.preferences = PreferredDistribution(user_prompt)
        # In a real scenario, we might use an LLM here to extract specific measurable outcomes from the prompt
        self.preferences.add_expected_outcome(f"Successfully complete: {user_prompt}")

    def update_beliefs(self, action_result):
        """
        Updates the internal state based on action results (Observations).
        """
        self.current_state.update(action_result)
        
    def get_context(self):
        return {
            "state": self.current_state.context,
            "preferences": self.preferences.get_preferences() if self.preferences else {}
        }
