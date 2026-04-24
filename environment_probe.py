import platform
import psutil
import socket
import os
import ctypes

class EnvironmentProbe:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnvironmentProbe, cls).__new__(cls)
            cls._instance.boot_profile = cls._instance._probe_system()
        return cls._instance
        
    def _probe_system(self):
        profile = {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "cpu_cores": psutil.cpu_count(logical=True),
        }
        
        # Check internet connection
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            profile["network_status"] = "online"
        except OSError:
            profile["network_status"] = "offline"
            
        # Admin rights check
        is_admin = False
        try:
            if os.name == 'nt':
                is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                is_admin = os.getuid() == 0
        except Exception:
            pass
        profile["is_admin"] = is_admin
        
        profile["cwd"] = os.getcwd()
        return profile
    
    def get_profile(self):
        # Refresh dynamic metrics
        self.boot_profile["ram_available_gb"] = round(psutil.virtual_memory().available / (1024**3), 2)
        return self.boot_profile
        
    def get_constraint_string(self):
        p = self.get_profile()
        constraints = [
            "=== HOST ENVIRONMENT LIMITATIONS ===",
            f"OS Platform: {p['os']} {p['os_release']} ({p['machine']})",
            f"Authorization: {'Admin/Root Privileges' if p['is_admin'] else 'Standard User - Avoid commands requiring elevation'}",
            f"Network Status: {p['network_status'].upper()}",
            f"Available Memory: {p['ram_available_gb']} GB out of {p['ram_total_gb']} GB",
            f"Current Working Dir: {p['cwd']}",
            "===================================="
        ]
        return "\n".join(constraints)

class RateLimitTracker:
    api_calls = 0
    estimated_tokens = 0
    
    @classmethod
    def log_call(cls, estimated_prompt_tokens: int = 500):
        cls.api_calls += 1
        cls.estimated_tokens += estimated_prompt_tokens
        
    @classmethod
    def get_usage_string(cls):
        return f"=== API USAGE LIMITS ===\nTotal Tool/Planner Invokes: {cls.api_calls}\nEstimated Tokens Consumed: {cls.estimated_tokens}\nManage your output efficiently to avoid rate-limiting.\n========================"
