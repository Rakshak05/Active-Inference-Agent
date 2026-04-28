"""
Code Execution Adapters — Run Python code snippets in a restricted namespace.

The sandbox exposes:
  - Standard builtins (print, len, range, enumerate, zip, sorted, sum, etc.)
  - json, re, os.path, datetime, math, csv, io modules
  - An 'inputs' dict passed from the step args

It explicitly blocks:
  - __import__ calls for 'os', 'subprocess', 'shutil', 'sys' (exec cannot escalate further)
  - open() calls to arbitrary paths are allowed but the sandbox user should be aware
"""

import json
import re
import math
import csv
import io
import os
import datetime
import traceback
from typing import Any
import builtins

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    restricted_modules = {"os", "subprocess", "shutil", "sys"}
    base_name = name.split(".")[0]
    if base_name in restricted_modules:
        raise ImportError(f"Importing '{name}' is restricted in the sandbox.")
    return builtins.__import__(name, globals, locals, fromlist, level)

_SAFE_BUILTINS = {
    "print": print, "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "sorted": sorted, "reversed": reversed, "sum": sum,
    "min": min, "max": max, "abs": abs, "round": round, "int": int,
    "float": float, "str": str, "bool": bool, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "type": type, "isinstance": isinstance,
    "hasattr": hasattr, "getattr": getattr, "repr": repr,
    "True": True, "False": False, "None": None,
    "__import__": _safe_import,
}

_SAFE_MODULES = {
    "json": json, "re": re, "math": math, "csv": csv, "io": io,
    "datetime": datetime, "os_path": os.path,
}


def execute_python_adapter(step: dict) -> Any:
    """
    Execute a Python code snippet in a sandboxed namespace.

    args:
      code   : str  — Python source code to execute
      inputs : dict — variables injected into the execution namespace
                      (accessible as `inputs['key']` inside the code snippet)
    
    The code can set a variable named `result` to return a value.
    Any `print()` calls are captured and included in the output.
    """
    args   = step.get("args", {})
    code   = str(args.get("code", ""))
    inputs = args.get("inputs") or {}

    if not code.strip():
        raise ValueError("execute_python: 'code' must be provided.")

    # Capture stdout
    captured_output = io.StringIO()

    # Build namespace
    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "inputs": inputs,
        "result": None,
        **_SAFE_MODULES,
    }

    import builtins
    old_print = builtins.print

    def captured_print(*a, **kw):
        sep = kw.get("sep", " ")
        end = kw.get("end", "\n")
        line = sep.join(str(x) for x in a) + end
        captured_output.write(line)
        old_print(*a, **kw)   # also shows in server console
        
        # Parallel Testing Suite: Emit terminal output event
        try:
            import asyncio
            from parallel_testing.events import emit_event
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(emit_event("TERMINAL_OUTPUT", line=line, source="execute_python"))
            else:
                asyncio.run(emit_event("TERMINAL_OUTPUT", line=line, source="execute_python"))
        except Exception:
            pass

    namespace["print"] = captured_print

    try:
        exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
    except Exception:
        tb = traceback.format_exc()
        print(f"[Code] Execution error:\n{tb}")
        raise RuntimeError(f"Code execution failed:\n{tb}")

    stdout = captured_output.getvalue()
    result = namespace.get("result")

    output = {
        "stdout": stdout,
        "result": result,
    }

    print(f"[Code] Python snippet executed. result={repr(result)[:200]}")
    return output


def evaluate_expression_adapter(step: dict) -> Any:
    """
    Safely evaluate a single Python expression and return its value.
    args: {expression: str, inputs: dict}
    """
    args       = step.get("args", {})
    expression = str(args.get("expression", ""))
    inputs     = args.get("inputs") or {}

    namespace = {**_SAFE_BUILTINS, **_SAFE_MODULES, **inputs}
    try:
        result = eval(expression, {"__builtins__": _SAFE_BUILTINS}, namespace)  # noqa: S307
        print(f"[Code] Evaluated: {expression} = {repr(result)[:200]}")
        return result
    except Exception as e:
        raise RuntimeError(f"Expression evaluation failed: {e}")


# ── registration ───────────────────────────────────────────────────────────────

def setup_code_adapters(toolgate):
    toolgate.register_adapter("execute_python",   execute_python_adapter)
    toolgate.register_adapter("run_code",         execute_python_adapter)   # alias
    toolgate.register_adapter("evaluate",         evaluate_expression_adapter)
    print("[Adapters] Code execution adapters registered.")
