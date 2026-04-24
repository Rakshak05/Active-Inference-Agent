import time
import sys
import itertools
from colorama import init, Fore, Style

init(autoreset=True)

def print_typing(text, delay=0.02, color=Fore.GREEN):
    """Prints text immediately (typing effect removed per user request)."""
    if color:
        print(color + text + Style.RESET_ALL)
    else:
        print(text)

def spinner(msg, delay=0.1, duration=1.5):
    """Spinner animation for terminal UX."""
    end_time = time.time() + duration
    for c in itertools.cycle(['|','/','-','\\']):
        if time.time() > end_time:
            break
        print(f"\r{Fore.CYAN}{msg} {c}{Style.RESET_ALL}", end="")
        time.sleep(delay)
    print(f"\r{Fore.CYAN}{msg} Done!{Style.RESET_ALL}   ")
