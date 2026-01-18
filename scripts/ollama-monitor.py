#!/usr/bin/env python3
"""
Ollama Monitor - Real-time dashboard for Ollama instances
"""

import urllib.request
import json
import subprocess
import time
import os
import sys

# Terminal colors
CLEAR, BOLD, RESET = "\033[2J\033[H", "\033[1m", "\033[0m"
CYAN, GREEN, YELLOW, MAGENTA, RED = "\033[36m", "\033[32m", "\033[33m", "\033[35m", "\033[31m"


def discover_ollama_instances():
    """Auto-discover running Ollama instances."""
    instances = {}
    
    # Try default port
    if check_ollama_available("http://localhost:11434"):
        instances["Ollama (default)"] = "http://localhost:11434"
    
    # Try common alternative ports
    for port in [11435, 11436]:
        url = f"http://localhost:{port}"
        if check_ollama_available(url):
            instances[f"Ollama (port {port})"] = url
    
    return instances


def check_ollama_available(url):
    """Check if an Ollama instance is available at the given URL."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=1) as r:
            return r.status == 200
    except:
        return False


def get_ollama_ps(url):
    """Get running models from Ollama instance."""
    try:
        with urllib.request.urlopen(f"{url}/api/ps", timeout=0.5) as r:
            return json.loads(r.read().decode()).get('models', [])
    except Exception:
        return None


def get_gpu_metrics():
    """Try to get GPU metrics from AMD GPU sysfs."""
    try:
        # Try multiple possible GPU device paths
        device_paths = [
            "/sys/class/drm/card1/device/",
            "/sys/class/drm/card0/device/",
        ]
        
        for base_path in device_paths:
            if not os.path.exists(base_path):
                continue
                
            try:
                with open(base_path + "mem_info_vram_used", "r") as f:
                    used = int(f.read().strip()) / 1024 / 1024
                with open(base_path + "mem_info_vram_total", "r") as f:
                    total = int(f.read().strip()) / 1024 / 1024
                with open(base_path + "gpu_busy_percent", "r") as f:
                    load = int(f.read().strip())
                
                # Sanity check: If VRAM usage is low but load is 99%, it's a driver glitch
                if load == 99 and used < (total * 0.1):
                    load = 0
                    
                return used, total, load
            except:
                continue
                
        return None, None, None
    except:
        return None, None, None


def get_sys_metrics():
    """Get system CPU and RAM metrics."""
    try:
        load_avg = os.getloadavg()[0]
        mem_output = subprocess.check_output("free -m", shell=True).decode().split('\n')[1].split()
        ram_used = int(mem_output[2])
        ram_total = int(mem_output[1])
        return load_avg, ram_used, ram_total
    except Exception:
        return 0.0, 0, 0


def draw(instances):
    """Draw the monitoring dashboard."""
    load_avg, ram_used, ram_total = get_sys_metrics()
    vram_used, vram_total, gpu_load = get_gpu_metrics()

    out = [f"{CLEAR}{BOLD}{CYAN}=== OLLAMA MONITOR ==={RESET}"]
    
    # System metrics
    out.append(f"{BOLD}CPU Load:{RESET} {YELLOW}{load_avg:.2f}{RESET} | "
               f"{BOLD}RAM:{RESET} {MAGENTA}{ram_used}MB/{ram_total}MB{RESET}", )
    
    # GPU metrics (if available)
    if vram_total is not None and gpu_load is not None:
        load_color = GREEN if gpu_load < 80 else RED
        out.append(f"{BOLD}GPU Load:{RESET} {load_color}{gpu_load}%{RESET} | "
                   f"{BOLD}VRAM:{RESET} {CYAN}{vram_used:.0f}MB/{vram_total:.0f}MB{RESET}")
    
    out.append("─" * 70)

    # Ollama instances
    for name, url in instances.items():
        models = get_ollama_ps(url)
        status = f"{GREEN}ONLINE{RESET}" if models is not None else f"{RED}OFFLINE{RESET}"
        out.append(f"\n{BOLD}{name}{RESET} [{status}] - {url}")
        
        if models:
            if len(models) > 0:
                out.append(f"  {'MODEL':<40} {'SIZE':<12} {'UNTIL':<20}")
                for m in models:
                    size_gb = m.get('size', 0) / (1024**3)
                    until = m.get('expires_at', 'N/A')
                    if until != 'N/A' and 'T' in until:
                        # Parse ISO timestamp and show relative time
                        until = until.split('T')[1].split('.')[0]
                    
                    out.append(f"  {m['name'][:39]:<40} {size_gb:>6.1f} GB   {until}")
            else:
                out.append(f"  {YELLOW}IDLE{RESET}")
        elif models is None:
            out.append(f"  {RED}Connection failed{RESET}")

    print("\n".join(out) + f"\n\n{BOLD}{CYAN}Refreshing... (Ctrl+C to quit){RESET}")


def main():
    print("Discovering Ollama instances...")
    instances = discover_ollama_instances()
    
    if not instances:
        print(f"{RED}✗ No Ollama instances found.{RESET}")
        print("  Make sure Ollama is running on the default port (11434)")
        sys.exit(1)
    
    print(f"Found {len(instances)} instance(s). Starting monitor...\n")
    time.sleep(1)
    
    try:
        while True:
            draw(instances)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
