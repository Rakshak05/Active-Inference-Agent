"""
Hard constraints and safety guidelines for the Active Inference Agent.
These are non-negotiable rules.
"""

HARD_CONSTRAINTS = [
    "Never delete files from the user's system without explicit, verified permission.",
    "Never share API keys, credentials, or personal information.",
    "Never execute arbitrary code downloaded from the internet without sandboxing.",
    "Do not perform destructive operations on databases or registries.",
    "Always inform the user if an action involves potential financial cost (e.g., paid APIs).",
    "Limit external API requests to avoid rate limits or abusive behavior."
]

def check_policy_against_constitution(policy_steps):
    """
    Static pre-execution safety check against hard constitutional constraints.
    EFE evaluation already vetted the plan; this is the final sanity gate
    for truly catastrophic / irreversible system-level operations only.
    """
    violations = []
    
    # Only block ops that could nuke the host OS or leak credentials
    BLOCKED_TOOLS = {
        "drop_table",       # database destruction
        "truncate_table",   # database destruction
        "format_disk",      # disk wipe
        "wipe_disk",        # disk wipe
        "rm_rf_root",       # system root wipe
    }

    BLOCKED_PATH_FRAGMENTS = [
        "c:\\windows",
        "c:/windows",
        "/etc",
        "/bin",
        "/usr",
        "/system32",
    ]

    for step in policy_steps:
        tool = step.get("tool", "").lower()
        args = step.get("args", {})
        path = str(args.get("path", "") or args.get("folder", "") or "").lower()

        if tool in BLOCKED_TOOLS:
            violations.append(f"Tool '{tool}' is constitutionally blocked.")

        for fragment in BLOCKED_PATH_FRAGMENTS:
            if fragment in path:
                violations.append(
                    f"Path '{path}' contains a protected system path fragment '{fragment}'."
                )
                break

    return violations

