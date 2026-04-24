"""
LLM Interpreter — Translates natural-language instructions into executable policies.

The system prompt embeds the full tool catalog so the LLM can choose ANY tool
without being hardcoded to specific behaviours.
"""

import json
from llm_gateway import LLMGateway


# ── Tool Registry with Metadata ──────────────────────────────────────────────

TOOL_REGISTRY = {
    "FILESYSTEM": {
        "metadata": {"cost": "free", "latency": "<10ms", "risk": "medium", "description": "Local file operations"},
        "tools": '''
  read_file        args: {path}                          → file contents (string)
  write_file       args: {path, content, mode}           → confirmation (mode: "w"|"a")
  delete_file      args: {path}                          → confirmation
  delete_folder    args: {path}                          → confirmation
  list_directory   args: {path}                          → list of {name,type,size,path}
  create_directory args: {path}                          → confirmation
  move_file        args: {src, dst}                      → confirmation
  copy_file        args: {src, dst}                      → confirmation
  check_path       args: {path}                          → {exists,type,size}
'''
    },
    "DATA": {
        "metadata": {"cost": "free", "latency": "<50ms", "risk": "low", "description": "JSON/CSV data manipulation and persistent semantic memory"},
        "tools": '''
  read_csv         args: {path}                          → list of row-dicts
  write_csv        args: {path, data}                    → confirmation
  read_json        args: {path}                          → parsed object
  write_json       args: {path, data}                    → confirmation
  store_memory     args: {fact, metadata?}               → confirmation
  search_memory    args: {query}                         → list of retrieved facts
  filter_records   args: {data, field, op, value}        → filtered list
  transform_records args: {data, output_field, template} → list with new field
  slice_records    args: {data, start, end}              → sublist
  get_field_values args: {data, field}                   → flat list of values
'''
    },
    "WEB": {
        "metadata": {"cost": "medium (API limits)", "latency": "1-3s", "risk": "low", "description": "HTTP requests & downloading"},
        "tools": '''
  http_get         args: {url, headers?, params?}        → response (auto JSON)
  http_post        args: {url, body, headers?}           → response (auto JSON)
  download_file    args: {url, path}                     → confirmation
'''
    },
    "COMMUNICATION": {
        "metadata": {"cost": "high (SMTP limits)", "latency": "5s+", "risk": "high", "description": "Sending emails & webhooks"},
        "tools": '''
  send_email       args: {to, subject, body}             → confirmation
  send_emails_bulk args: {recipients, subject_template, body_template, to_field?}
  send_webhook     args: {url, payload}                  → response
  log_message      args: {message}                       → echoed message
  print_table      args: {data, fields?}                 → formatted table string
'''
    },
    "CODE": {
        "metadata": {"cost": "free", "latency": "<1s", "risk": "high", "description": "Python code execution"},
        "tools": '''
  execute_python   args: {code, inputs?}                 → {stdout, result}
  evaluate         args: {expression, inputs?}           → computed value
'''
    }
}

VARIABLE_SYSTEM_DOCS = """
=== VARIABLE SYSTEM (TOOL CHAINING) ===
• Add "output_var": "myvar" to any step to store its return value.
• Reference stored values anywhere in later args using "$myvar". This chains outputs securely without LLM loops!
• Add "fallback": { step_object } to gracefully degrade if the main tool fails.

=== CONTROL FLOW ===
  foreach          args: {items: "$listvar", steps: [ ... ]}
  conditional      args: {condition: "$var", if_true: [ ... ]}
"""

class LLMInterpreter:
    """Parses user instructions into structured executable policies (Policy π)."""

    def __init__(self):
        self.gateway = LLMGateway()

    def _route_intent(self, user_instruction: str) -> list:
        """
        Tool Intent Router: Decides WHICH tool categories are relevant 
        to avoid blinding the planner with irrelevant massive tool catalogs.
        """
        categories_map = {k: v["metadata"]["description"] for k, v in TOOL_REGISTRY.items()}
        sys_prompt = (
            "You are a Tool Intent Router. Based on the task, return ONLY a JSON array of required Tool Categories.\n"
            f"Available Categories: {json.dumps(categories_map)}"
        )
        usr_prompt = f"Task: {user_instruction}\nWhich categories are needed? Return JSON array of strings."
        response = self.gateway.generate_completion(sys_prompt, usr_prompt, json_mode=True)
        try:
            cats = json.loads(response)
            if not isinstance(cats, list) or len(cats) == 0:
                return list(TOOL_REGISTRY.keys())
            # Ensure valid categories
            valid_cats = [c for c in cats if c in TOOL_REGISTRY]
            return valid_cats if valid_cats else list(TOOL_REGISTRY.keys())
        except:
            return list(TOOL_REGISTRY.keys())

    def _build_optimized_catalog(self, categories: list) -> str:
        """Constructs a lean tool catalog based on routed categories + metadata."""
        catalog = "=== OPTIMIZED TOOL CATALOG ===\n"
        for cat in categories:
            meta = TOOL_REGISTRY[cat]["metadata"]
            catalog += f"\n[{cat}] | Cost: {meta['cost']} | Latency: {meta['latency']} | Risk: {meta['risk']}\n"
            catalog += TOOL_REGISTRY[cat]["tools"]
        catalog += "\n" + VARIABLE_SYSTEM_DOCS
        
        catalog += """
=== OUTPUT FORMAT ===
Respond with a RAW JSON array only — no markdown fences. Example:
[
  {"tool": "read_csv", "args": {"path": "data.csv"}, "output_var": "rows", "fallback": {"tool": "log", "args": {}}}
]"""
        return catalog

    def generate_policy(self, user_instruction: str, current_context: dict,
                        feedback: str = None) -> list:
        """
        Generate a multi-step policy for the given instruction.
        """
        routed_categories = self._route_intent(user_instruction)
        optimized_catalog = self._build_optimized_catalog(routed_categories)
        
        system_prompt = (
            "You are an AI Planner for an Active Inference agent system.\n"
            "Your job is to break down any user instruction into a concrete, "
            "step-by-step execution policy using the tools listed below.\n"
            "CRITCAL RULE: When saving files, always check and create parent directories if they do not exist.\n\n"
            + optimized_catalog
        )

        if feedback:
            system_prompt += (
                f"\n\nREFINEMENT GUIDANCE (previous plan was rejected by the safety assessor):\n"
                f"{feedback}\n"
                "Adjust the plan to reduce risk or ambiguity. "
                "Still output ONLY the raw JSON array."
            )

        from environment_probe import EnvironmentProbe, RateLimitTracker
        env_probe = EnvironmentProbe()

        user_prompt = (
            f"{env_probe.get_constraint_string()}\n"
            f"{RateLimitTracker.get_usage_string()}\n\n"
            f"Current context: {json.dumps(current_context)}\n\n"
            f"User instruction: {user_instruction}\n\n"
            "Output the JSON policy array now:"
        )

        response = self.gateway.generate_completion(system_prompt, user_prompt, json_mode=True)

        return self._parse_policy(response)

    def generate_dag_plan(self, user_instruction: str, current_context: dict) -> list:
        """
        Planner Module: Introduce a dedicated prompt phase solely for draft-planning
        before taking any action. Generates a DAG of subgoals.
        """
        system_prompt = (
            "You are the Strategic Planner for an AI Agent. "
            "Break down the user's complex task into a Directed Acyclic Graph (DAG) of explicit sub-goals.\n"
            "CRITICAL CAPABILITIES: The agent has native tools for Filesystem, Code Execution, HTTP requests, and a Persistent Semantic Memory VectorDB (search_memory, store_memory).\n"
            "RULE 1: If the user asks to 'Remember' something, the goal is 'Store fact in memory'.\n"
            "RULE 2: If the user asks about personal context, past interactions, or explicit past facts, the first goal MUST be 'Search memory for facts'.\n"
            "RULE 3: If the user asks for general real-world facts, recent news, or the 'latest version' of external software, DO NOT search memory. Instead, use HTTP requests to search the internet.\n"
            "Return ONLY a RAW JSON array of objects with these keys:\n"
            "  'id' (string, e.g., 'step_1')\n"
            "  'description' (string, concrete and actionable)\n"
            "  'dependencies' (array of strings representing prerequisite step ids)\n"
            "Do not include any other text or markdown formatting."
        )
        
        from environment_probe import EnvironmentProbe, RateLimitTracker
        env_probe = EnvironmentProbe()
        
        user_prompt = (
            f"{env_probe.get_constraint_string()}\n"
            f"{RateLimitTracker.get_usage_string()}\n\n"
            f"Context: {json.dumps(current_context)}\n\n"
            f"Task: {user_instruction}\n\n"
            "Output the JSON DAG plan now:"
        )
        response = self.gateway.generate_completion(system_prompt, user_prompt, json_mode=True)
        return self._parse_policy(response)

    def critique_policy(self, policy: list, current_context: dict) -> tuple:
        """
        Pre-Action Critique: Verifies parameters before executing high-risk tools.
        Returns (is_valid: bool, feedback: str).
        """
        system_prompt = (
            "You are a strict Safety and Logic Reviewer for an AI agent.\n"
            "Review the proposed tool execution policy. "
            "Check for parameter correctness, destructive risks, and logic flaws.\n"
            "Respond ONLY with a JSON object: {\"is_valid\": true/false, \"feedback\": \"reasoning...\"}"
        )
        user_prompt = (
            f"Context: {json.dumps(current_context)}\n"
            f"Proposed Policy: {json.dumps(policy)}\n\n"
            "Evaluate:"
        )
        response = self.gateway.generate_completion(system_prompt, user_prompt, json_mode=True)
        try:
            parsed = json.loads(response)
            return parsed.get("is_valid", False), parsed.get("feedback", "No feedback provided.")
        except:
            return False, "Failed to parse critique response."

    def validate_outcome(self, user_instruction: str, expected_goal: str, actual_outcome: str) -> tuple:
        """
        Post-Action Validation: Check tool outputs strictly against expected outcomes.
        Returns (is_successful: bool, feedback: str).
        """
        system_prompt = (
            "You are an Outcome Validator for an AI agent.\n"
            "Assess if the actual outcome successfully satisfies the user's expected goal.\n"
            "Respond ONLY with a JSON object: {\"success\": true/false, \"feedback\": \"reasoning...\"}"
        )
        user_prompt = (
            f"Goal/Subtask: {expected_goal}\n"
            f"General Context: {user_instruction}\n"
            f"Actual Outcome: {actual_outcome}\n\n"
            "Assess:"
        )
        response = self.gateway.generate_completion(system_prompt, user_prompt, json_mode=True)
        try:
            parsed = json.loads(response)
            return parsed.get("success", False), parsed.get("feedback", "No feedback provided.")
        except:
            return False, "Failed to parse validation response."

    # ── parsing ────────────────────────────────────────────────────────────────

    def _parse_policy(self, response: str) -> list:
        """Robustly parse the LLM response into a list of step dicts."""
        # Strip any accidental markdown fences
        text = response.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]
        text = text.strip()

        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return parsed

            if isinstance(parsed, dict):
                # If it's a bare action object, wrap it
                if "tool" in parsed or "action" in parsed:
                    if "action" in parsed and "tool" not in parsed:
                        parsed["tool"] = parsed.pop("action")
                    return [parsed]
                    
                # Wrapped in a dict — find first list value
                for val in parsed.values():
                    if isinstance(val, list):
                        return val
                        
                # Handle objects returned as dictionary mappings instead of lists
                if all(isinstance(v, dict) and ("description" in v or "tool" in v) for v in parsed.values()):
                    converted = []
                    for k, v in parsed.items():
                        if "id" not in v:
                            v["id"] = k
                        converted.append(v)
                    return converted

            print(f"Parsed JSON is structurally invalid: {parsed}")
            return []

        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM Response as JSON: {e}")
            print(f"Raw Text: {text}")
            import re
            
            # Attempt aggressive JSON array extraction
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list):
                        print("✓ Successfully extracted JSON array via regex fallback.")
                        return parsed
                except json.JSONDecodeError:
                    pass
            return []