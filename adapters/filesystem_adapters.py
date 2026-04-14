"""
Filesystem Adapters — Real OS-level file and directory operations.
All path arguments are read from step["args"] and support common key aliases.
"""

import os
import shutil


# ── helpers ────────────────────────────────────────────────────────────────────

def _path(step: dict) -> str:
    args = step.get("args", {})
    raw = (args.get("path") or args.get("folder") or
           args.get("directory") or args.get("dir") or args.get("file") or "")
    return os.path.normpath(str(raw).strip())


def _require_path(step: dict):
    p = _path(step)
    if not p or p == ".":
        raise ValueError("No path provided.")
    return p


# ── adapters ───────────────────────────────────────────────────────────────────

def read_file_adapter(step: dict):
    """Read a text file and return its contents."""
    path = _require_path(step)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.path.isfile(path):
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    print(f"[Filesystem] Read file: {path} ({len(content)} chars)")
    return content


def write_file_adapter(step: dict):
    """Write or append text content to a file."""
    path = _require_path(step)
    args  = step.get("args", {})
    content = str(args.get("content", ""))
    mode    = args.get("mode", "w")           # "w" overwrite, "a" append
    if mode not in ("w", "a"):
        mode = "w"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        f.write(content)
    action = "Wrote" if mode == "w" else "Appended to"
    print(f"[Filesystem] {action} file: {path}")
    return f"{action} {len(content)} chars to: {path}"


def delete_file_adapter(step: dict):
    """Delete a single file."""
    path = _require_path(step)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.path.isfile(path):
        raise IsADirectoryError(f"Not a file: {path}")
    os.remove(path)
    print(f"[Filesystem] Deleted file: {path}")
    return f"Deleted file: {path}"


def delete_folder_adapter(step: dict):
    """Recursively delete a directory and all its contents."""
    path = _require_path(step)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Not a directory: {path}")
    shutil.rmtree(path)
    print(f"[Filesystem] Deleted folder: {path}")
    return f"Deleted folder: {path}"


def list_directory_adapter(step: dict):
    """List directory contents. Returns a list of dicts with name/type/size."""
    path = _require_path(step)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    entries = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        entry = {
            "name": name,
            "type": "directory" if os.path.isdir(full) else "file",
            "size": os.path.getsize(full) if os.path.isfile(full) else None,
            "path": full,
        }
        entries.append(entry)
    print(f"[Filesystem] Listed directory: {path} ({len(entries)} entries)")
    return entries


def create_directory_adapter(step: dict):
    """Create a directory (and any missing parents)."""
    path = _require_path(step)
    os.makedirs(path, exist_ok=True)
    print(f"[Filesystem] Created directory: {path}")
    return f"Directory created: {path}"


def move_file_adapter(step: dict):
    """Move or rename a file/directory."""
    args = step.get("args", {})
    src = os.path.normpath(str(args.get("src") or args.get("source") or ""))
    dst = os.path.normpath(str(args.get("dst") or args.get("destination") or args.get("dest") or ""))
    if not src or not dst:
        raise ValueError("Both 'src' and 'dst' args are required for move_file.")
    shutil.move(src, dst)
    print(f"[Filesystem] Moved: {src} → {dst}")
    return f"Moved: {src} → {dst}"


def copy_file_adapter(step: dict):
    """Copy a file or directory tree."""
    args = step.get("args", {})
    src = os.path.normpath(str(args.get("src") or args.get("source") or ""))
    dst = os.path.normpath(str(args.get("dst") or args.get("destination") or args.get("dest") or ""))
    if not src or not dst:
        raise ValueError("Both 'src' and 'dst' args are required for copy_file.")
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f"[Filesystem] Copied: {src} → {dst}")
    return f"Copied: {src} → {dst}"


def check_path_adapter(step: dict):
    """Check whether a path exists and return metadata."""
    path = _require_path(step)
    if not os.path.exists(path):
        return {"exists": False, "path": path}
    is_dir = os.path.isdir(path)
    info = {
        "exists": True,
        "path": path,
        "type": "directory" if is_dir else "file",
        "size": None if is_dir else os.path.getsize(path),
    }
    return info


# ── registration ───────────────────────────────────────────────────────────────

def setup_filesystem_adapters(toolgate):
    toolgate.register_adapter("read_file",         read_file_adapter)
    toolgate.register_adapter("write_file",        write_file_adapter)
    toolgate.register_adapter("delete_file",       delete_file_adapter)
    toolgate.register_adapter("delete_folder",     delete_folder_adapter)
    toolgate.register_adapter("delete_directory",  delete_folder_adapter)
    toolgate.register_adapter("remove_folder",     delete_folder_adapter)
    toolgate.register_adapter("remove_file",       delete_file_adapter)
    toolgate.register_adapter("list_directory",    list_directory_adapter)
    toolgate.register_adapter("list_folder",       list_directory_adapter)
    toolgate.register_adapter("create_directory",  create_directory_adapter)
    toolgate.register_adapter("mkdir",             create_directory_adapter)
    toolgate.register_adapter("move_file",         move_file_adapter)
    toolgate.register_adapter("copy_file",         copy_file_adapter)
    toolgate.register_adapter("check_path",        check_path_adapter)
    toolgate.register_adapter("file_exists",       check_path_adapter)
    print("[Adapters] Filesystem adapters registered.")
