import os
import json
from pathlib import Path

DATA_DIR = "data"
PROFILE_PATH = Path(DATA_DIR) / "user_profile.json"

def is_first_run():
    return not PROFILE_PATH.exists()

def load_user():
    if PROFILE_PATH.exists():
        try:
            with open(PROFILE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_user(name, gmail):
    os.makedirs(DATA_DIR, exist_ok=True)
    profile = {
        "name": name,
        "gmail": gmail
    }
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=4)

def reset_memory():
    if PROFILE_PATH.exists():
        os.remove(PROFILE_PATH)
        return True
    return False
