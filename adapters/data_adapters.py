"""
Data Adapters — CSV, JSON, Excel reading/writing and in-memory record processing.
These adapters return structured Python objects (lists/dicts) so they can be
stored in the execution context and referenced by subsequent steps.
"""

import os
import csv
import json
import io
import re


# ── helpers ────────────────────────────────────────────────────────────────────

def _path(step: dict) -> str:
    args = step.get("args", {})
    raw = args.get("path") or args.get("file") or args.get("filename") or ""
    return os.path.normpath(str(raw).strip())


def _resolve_data(value, context: dict):
    """If value is a $var reference, pull from context."""
    if isinstance(value, str) and value.startswith("$"):
        return context.get(value[1:], value)
    return value


# ── CSV ────────────────────────────────────────────────────────────────────────

def read_csv_adapter(step: dict):
    """
    Read a CSV file and return a list of dicts (one per row).
    args: {path: str, delimiter: str (default ',')}
    """
    path = _path(step)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    delimiter = step.get("args", {}).get("delimiter", ",")
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            records.append(dict(row))
    print(f"[Data] Read CSV: {path} ({len(records)} rows)")
    return records


def write_csv_adapter(step: dict):
    """
    Write a list of dicts to a CSV file.
    args: {path: str, data: list|"$varname", fieldnames: list (optional)}
    """
    args = step.get("args", {})
    path = os.path.normpath(str(args.get("path", "")))
    data = args.get("data", [])
    if not isinstance(data, list) or not data:
        raise ValueError("write_csv: 'data' must be a non-empty list of records.")
    fieldnames = args.get("fieldnames") or list(data[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
    print(f"[Data] Wrote CSV: {path} ({len(data)} rows)")
    return f"Wrote {len(data)} rows to {path}"


# ── JSON ───────────────────────────────────────────────────────────────────────

def read_json_adapter(step: dict):
    """Read a JSON file and return the parsed Python object."""
    path = _path(step)
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Data] Read JSON: {path}")
    return data


def write_json_adapter(step: dict):
    """
    Write a Python object to a JSON file.
    args: {path: str, data: any|"$varname", indent: int (default 2)}
    """
    args = step.get("args", {})
    path = os.path.normpath(str(args.get("path", "")))
    data = args.get("data")
    indent = int(args.get("indent", 2))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    print(f"[Data] Wrote JSON: {path}")
    return f"Wrote JSON to {path}"


# ── In-memory record operations ────────────────────────────────────────────────

def filter_records_adapter(step: dict):
    """
    Filter a list of records.
    args:
      data: list|"$varname"
      field: str
      op: "eq"|"ne"|"gt"|"lt"|"gte"|"lte"|"contains"|"startswith"
      value: any
    """
    args  = step.get("args", {})
    data  = args.get("data", [])
    field = str(args.get("field", ""))
    op    = str(args.get("op", "eq")).lower()
    value = args.get("value")

    if not isinstance(data, list):
        raise TypeError("filter_records: 'data' must be a list.")

    ops = {
        "eq":         lambda a, b: str(a) == str(b),
        "ne":         lambda a, b: str(a) != str(b),
        "gt":         lambda a, b: float(a) > float(b),
        "lt":         lambda a, b: float(a) < float(b),
        "gte":        lambda a, b: float(a) >= float(b),
        "lte":        lambda a, b: float(a) <= float(b),
        "contains":   lambda a, b: str(b).lower() in str(a).lower(),
        "startswith": lambda a, b: str(a).lower().startswith(str(b).lower()),
    }

    fn = ops.get(op, ops["eq"])
    filtered = [r for r in data if fn(r.get(field, ""), value)]
    print(f"[Data] filter_records: {len(data)} → {len(filtered)} records (field={field}, op={op}, value={value})")
    return filtered


def transform_records_adapter(step: dict):
    """
    Add or overwrite a field on each record using a template string.
    Template supports {field_name} placeholders.
    args:
      data: list|"$varname"
      output_field: str  (name of the new/overwritten field)
      template: str      (e.g. "Hello {name}, your ID is {id}")
    """
    args         = step.get("args", {})
    data         = args.get("data", [])
    output_field = str(args.get("output_field", "transformed"))
    template     = str(args.get("template", ""))

    if not isinstance(data, list):
        raise TypeError("transform_records: 'data' must be a list.")

    result = []
    for record in data:
        new_record = dict(record)
        try:
            new_record[output_field] = template.format_map(new_record)
        except KeyError as e:
            new_record[output_field] = f"[template error: missing {e}]"
        result.append(new_record)

    print(f"[Data] transform_records: added/updated field '{output_field}' on {len(result)} records")
    return result


def slice_records_adapter(step: dict):
    """
    Return a slice of a list.
    args: {data: list|"$varname", start: int, end: int}
    """
    args  = step.get("args", {})
    data  = args.get("data", [])
    start = int(args.get("start", 0))
    end   = args.get("end")
    sliced = data[start:end] if end is not None else data[start:]
    print(f"[Data] slice_records: {len(data)} → {len(sliced)} records")
    return sliced


def get_field_values_adapter(step: dict):
    """
    Extract a single field's values from all records as a flat list.
    args: {data: list|"$varname", field: str}
    """
    args  = step.get("args", {})
    data  = args.get("data", [])
    field = str(args.get("field", ""))
    values = [r.get(field) for r in data if isinstance(r, dict)]
    return values


# ── registration ───────────────────────────────────────────────────────────────

def setup_data_adapters(toolgate):
    toolgate.register_adapter("read_csv",            read_csv_adapter)
    toolgate.register_adapter("write_csv",           write_csv_adapter)
    toolgate.register_adapter("read_json",           read_json_adapter)
    toolgate.register_adapter("write_json",          write_json_adapter)
    toolgate.register_adapter("filter_records",      filter_records_adapter)
    toolgate.register_adapter("transform_records",   transform_records_adapter)
    toolgate.register_adapter("slice_records",       slice_records_adapter)
    toolgate.register_adapter("get_field_values",    get_field_values_adapter)
    print("[Adapters] Data adapters registered.")
