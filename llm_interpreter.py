"""
LLM Interpreter — Translates natural-language instructions into executable policies.

The system prompt embeds the full tool catalog so the LLM can choose ANY tool
without being hardcoded to specific behaviours.
"""

import json
from llm_gateway import LLMGateway


# ── Tool catalog injected into every planning prompt ──────────────────────────

TOOL_CATALOG = """
=== AVAILABLE TOOLS ===

FILESYSTEM
  read_file        args: {path}                          → file contents (string)
  write_file       args: {path, content, mode}           → confirmation (mode: "w"|"a")
  delete_file      args: {path}                          → confirmation
  delete_folder    args: {path}                          → confirmation
  list_directory   args: {path}                          → list of {name,type,size,path}
  create_directory args: {path}                          → confirmation
  move_file        args: {src, dst}                      → confirmation
  copy_file        args: {src, dst}                      → confirmation
  check_path       args: {path}                          → {exists,type,size}

DATA
  read_csv         args: {path}                          → list of row-dicts
  write_csv        args: {path, data}                    → confirmation
  read_json        args: {path}                          → parsed object
  write_json       args: {path, data}                    → confirmation
  filter_records   args: {data, field, op, value}        → filtered list
                        op: "eq"|"ne"|"gt"|"lt"|"gte"|"lte"|"contains"|"startswith"
  transform_records args: {data, output_field, template} → list with new field
                        template uses {field_name} placeholders
  slice_records    args: {data, start, end}              → sublist
  get_field_values args: {data, field}                   → flat list of values

WEB / HTTP
  http_get         args: {url, headers?, params?}        → response (auto JSON)
  http_post        args: {url, body, headers?}           → response (auto JSON)
  download_file    args: {url, path}                     → confirmation

COMMUNICATION
  send_email       args: {to, subject, body}             → confirmation
  send_emails_bulk args: {recipients, subject_template,  → {summary, results}
                          body_template, to_field?}
                        templates use {field_name} placeholders
  send_webhook     args: {url, payload}                  → response
  log_message      args: {message}                       → echoed message
  print_table      args: {data, fields?}                 → formatted table string

CODE
  execute_python   args: {code, inputs?}                 → {stdout, result}
  evaluate         args: {expression, inputs?}           → computed value

CONTROL FLOW (special — handled by the executor, not real tools)
  foreach          args: {items, steps}
                        items : "$varname" referencing a list
                        steps : list of tool steps; use "$item" for current element,
                                "$item.field" for a field, "$index" for loop index
  conditional      args: {condition, if_true, if_false?}

=== VARIABLE SYSTEM ===
• Add "output_var": "myvar" to any step to store its return value.
• Reference stored values anywhere in later args using "$myvar" or "$myvar.fieldname".
• Inside foreach, "$item" is the current element, "$item.field" accesses a dict field.

=== OUTPUT FORMAT ===
Respond with a RAW JSON array only — no markdown fences, no explanation.
Each element is a tool step object. Example:

[
  {"tool": "read_csv", "args": {"path": "data.csv"}, "output_var": "rows"},
  {"tool": "filter_records", "args": {"data": "$rows", "field": "id", "op": "lte", "value": "100"}, "output_var": "patients"},
  {"tool": "send_emails_bulk", "args": {
      "recipients": "$patients",
      "subject_template": "Reminder for {name}",
      "body_template": "Hello {name}, this is your appointment reminder. Your ID is {id}.",
      "to_field": "email"
  }}
]
"""


class LLMInterpreter:
    """Parses user instructions into structured executable policies (Policy π)."""

    def __init__(self):
        self.gateway = LLMGateway()

    def generate_policy(self, user_instruction: str, current_context: dict,
                        feedback: str = None) -> list:
        """
        Generate a multi-step policy for the given instruction.

        Args:
            user_instruction : Natural language task
            current_context  : World-model context dict
            feedback         : Optional EFE rejection reason for refinement

        Returns:
            List of step dicts (the policy π)
        """
        system_prompt = (
            "You are an AI Planner for an Active Inference agent system.\n"
            "Your job is to break down any user instruction into a concrete, "
            "step-by-step execution policy using the tools listed below.\n\n"
            + TOOL_CATALOG
        )

        if feedback:
            system_prompt += (
                f"\n\nREFINEMENT GUIDANCE (previous plan was rejected by the safety assessor):\n"
                f"{feedback}\n"
                "Adjust the plan to reduce risk or ambiguity. "
                "Still output ONLY the raw JSON array."
            )

        user_prompt = (
            f"Current context: {json.dumps(current_context)}\n\n"
            f"User instruction: {user_instruction}\n\n"
            "Output the JSON policy array now:"
        )

        response = self.gateway.generate_completion(system_prompt, user_prompt, json_mode=True)

        return self._parse_policy(response)

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