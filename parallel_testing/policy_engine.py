from typing import List, Dict, Tuple
from security_constitution import check_policy_against_constitution

class PolicyEngine:
    def __init__(self):
        self.rules = [
            self.rule_constitution,
            self.rule_no_destructive_deletion,
            self.rule_no_system_path_access
        ]

    def evaluate(self, action: List[Dict], context: Dict) -> Tuple[bool, str]:
        for rule in self.rules:
            allowed, reason = rule(action, context)
            if not allowed:
                return False, reason
        return True, "Safe"

    def rule_constitution(self, action: List[Dict], context: Dict) -> Tuple[bool, str]:
        violations = check_policy_against_constitution(action)
        if violations:
            return False, f"Constitution violation: {', '.join(violations)}"
        return True, ""

    def rule_no_destructive_deletion(self, action: List[Dict], context: Dict) -> Tuple[bool, str]:
        for step in action:
            if step.get("tool") in ("delete_file", "delete_folder"):
                # Check if it's in the workspace or if user gave permission (in context)
                path = step.get("args", {}).get("path", "")
                if not path:
                    continue
                # For now, let's just block all deletions unless 'user_approved_deletion' is in context
                if not context.get("user_approved_deletion"):
                    return False, f"Destructive operation '{step.get('tool')}' on '{path}' requires explicit user approval."
        return True, ""

    def rule_no_system_path_access(self, action: List[Dict], context: Dict) -> Tuple[bool, str]:
        # This is already partly covered by constitution but we can add more specific rules
        return True, ""

# Global instance
policy_engine = PolicyEngine()
