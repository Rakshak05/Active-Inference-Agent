"""
Toolgate — Safety valve and execution engine.

Key capabilities over the original:
  1. Variable context  — steps can save outputs with `output_var` and
                         reference them with "$varname" or "$varname.field"
  2. foreach loops     — special tool "foreach" iterates over a list
  3. conditional       — special tool "conditional" for if/else branching
  4. Structured returns — adapters can return dicts/lists; the execution result
                          records a str summary while the raw value goes to context
"""

from security_constitution import check_policy_against_constitution


class Toolgate:
    """
    The Safety Valve and Execution mechanism.
    Only executes plans that have passed Active Inference (EFE < Threshold).
    """

    def __init__(self):
        self.adapters: dict = {}

    def register_adapter(self, name: str, func):
        self.adapters[name] = func

    # ── public entry point ─────────────────────────────────────────────────────

    def execute_policy(self, policy: list) -> list:
        """
        Run the full policy.
        Returns a list of result dicts, one per step (including nested foreach results).
        """
        # Final constitutional safety gate
        violations = check_policy_against_constitution(policy)
        if violations:
            raise Exception(f"CRITICAL SAFETY VIOLATION PRE-EXECUTION: {violations}")

        context = {}   # shared variable store for this execution run
        results = []
        for step in policy:
            result = self._execute_step(step, context)
            results.append(result)
        return results

    # ── internal execution ─────────────────────────────────────────────────────

    def _execute_step(self, step: dict, context: dict) -> dict:
        tool_name = step.get("tool", "")

        # ── special control-flow tools ─────────────────────────────────────────
        if tool_name == "foreach":
            return self._execute_foreach(step, context)

        if tool_name == "conditional":
            return self._execute_conditional(step, context)

        # ── resolve $var references in args ───────────────────────────────────
        resolved_args = self._resolve(step.get("args", {}), context)
        resolved_step = {**step, "args": resolved_args}

        # ── dispatch to adapter ────────────────────────────────────────────────
        if tool_name in self.adapters:
            print(f"[Toolgate] >  Running '{tool_name}' …")
            try:
                raw_result = self.adapters[tool_name](resolved_step)

                # Store structured result in context under output_var
                output_var = step.get("output_var")
                if output_var:
                    context[output_var] = raw_result
                    print(f"[Toolgate]     └─ saved to ${output_var}")

                return {
                    "step":           step,
                    "actual_outcome": self._summarise(raw_result),
                    "raw":            raw_result,
                    "status":         "success",
                }
            except Exception as e:
                print(f"[Toolgate] X '{tool_name}' raised: {e}")
                if "fallback" in step and isinstance(step["fallback"], dict):
                    print(f"[Toolgate] >  Executing Fallback Handler for '{tool_name}'...")
                    return self._execute_step(step["fallback"], context)
                return {"step": step, "error": str(e), "status": "error"}

        else:
            print(f"[Toolgate] !  No adapter for '{tool_name}'. Skipping.")
            return {
                "step":           step,
                "actual_outcome": f"No adapter registered for tool: '{tool_name}'",
                "status":         "skipped",
            }

    # ── foreach ────────────────────────────────────────────────────────────────

    def _execute_foreach(self, step: dict, context: dict) -> dict:
        """
        Iterate step["args"]["items"] and execute step["args"]["steps"] per item.

        In the inner steps:
          $item         → current element
          $item.field   → field of current element (if element is a dict)
          $index        → 0-based loop index
        """
        args     = step.get("args", {})
        items    = self._resolve(args.get("items", []), context)
        substeps = args.get("steps", [])

        if not isinstance(items, list):
            items = [items]

        all_results = []
        total = len(items)
        print(f"[Toolgate] 🔁 foreach: {total} items × {len(substeps)} steps")

        for idx, item in enumerate(items):
            # Shadow context with loop variables
            loop_ctx = {**context, "item": item, "index": idx}

            print(f"[Toolgate]    iteration {idx + 1}/{total}")
            for substep in substeps:
                r = self._execute_step(substep, loop_ctx)
                all_results.append(r)

            # Propagate any output_vars set inside the loop back to parent context
            for key, val in loop_ctx.items():
                if key not in ("item", "index") and key not in context:
                    context[key] = val

        output_var = step.get("output_var")
        if output_var:
            context[output_var] = all_results

        return {
            "step":           step,
            "actual_outcome": f"foreach completed: {total} iterations × {len(substeps)} steps",
            "results":        all_results,
            "status":         "success",
        }

    # ── conditional ───────────────────────────────────────────────────────────

    def _execute_conditional(self, step: dict, context: dict) -> dict:
        """
        if/else branching.
        args:
          condition  : "$varname" or a literal bool/truthy value
          if_true    : list of steps to run when condition is truthy
          if_false   : list of steps to run when condition is falsy (optional)
        """
        args      = step.get("args", {})
        condition = self._resolve(args.get("condition"), context)
        branch    = args.get("if_true", []) if condition else args.get("if_false", [])

        print(f"[Toolgate] 🔀 conditional: condition={bool(condition)}, running {'if_true' if condition else 'if_false'} branch")

        results = []
        for substep in branch:
            results.append(self._execute_step(substep, context))

        return {
            "step":    step,
            "actual_outcome": f"conditional branch executed ({len(results)} steps)",
            "results": results,
            "status":  "success",
        }

    # ── variable resolution ────────────────────────────────────────────────────

    def _resolve(self, value, context: dict):
        """
        Recursively resolve $varname and $varname.field references.
        Works on strings, dicts, and lists.
        """
        if isinstance(value, str):
            return self._resolve_string(value, context)
        elif isinstance(value, dict):
            return {k: self._resolve(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve(v, context) for v in value]
        return value

    def _resolve_string(self, s: str, context: dict):
        """Resolve a single string that may be a $var or $var.field."""
        if not s.startswith("$"):
            return s
        path = s[1:].split(".")       # e.g. "patient.email" → ["patient", "email"]
        val  = context.get(path[0])
        for key in path[1:]:
            if isinstance(val, dict):
                val = val.get(key)
            elif hasattr(val, key):
                val = getattr(val, key)
            else:
                val = None
                break
        return val

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _summarise(raw) -> str:
        """Convert any adapter return value to a concise string for logging."""
        if isinstance(raw, str):
            return raw[:3000]
        if isinstance(raw, (list, dict)):
            import json
            try:
                s = json.dumps(raw, default=str)
                return s[:3000] + ("…" if len(s) > 3000 else "")
            except Exception:
                pass
        return str(raw)[:3000]
