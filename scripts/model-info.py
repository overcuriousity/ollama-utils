#!/usr/bin/env python3
"""
Ollama Model Inventory
- Parses the official 'Capabilities' section from ollama show
- Accurate VRAM estimation
"""

import subprocess
import re
from typing import Dict, List

def get_cmd_output(cmd: List[str]) -> str:
    try:
        # Run command and get stdout
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""

def parse_parameters(param_str: str) -> float:
    """Parses '8.0B' or '307M' into standard Billions (float)"""
    if not param_str or param_str == "N/A": return 0.0
    clean_val = re.sub(r"[^0-9.]", "", param_str)
    try:
        val = float(clean_val)
        if "M" in param_str.upper(): return val / 1000.0
        return val
    except ValueError: return 0.0

def estimate_vram(params_billions: float, quant: str, context: int, context_used: int) -> str:
    """Estimates VRAM usage (Model Weights + Typical KV Cache)."""
    if params_billions == 0.0: return "N/A"

    # 1. Weights Size (bits per parameter)
    q_up = quant.upper()
    if "MXFP4" in q_up or "FP4" in q_up: bpp = 0.55
    elif "Q8" in q_up: bpp = 1.0
    elif "Q6" in q_up: bpp = 0.85
    elif "Q5" in q_up: bpp = 0.75
    elif "Q4" in q_up: bpp = 0.65
    elif "Q3" in q_up: bpp = 0.55
    elif "Q2" in q_up: bpp = 0.45
    elif "IQ" in q_up: bpp = 0.35  # IQ quantization
    elif "F16" in q_up or "BF16" in q_up: bpp = 2.0
    elif "F32" in q_up: bpp = 4.0
    else: bpp = 0.65  # Default Q4_K_M

    weight_gb = params_billions * bpp

    # 2. KV Cache Size
    # More accurate formula: context_tokens * embedding_dim * layers * 2 (K+V) * bytes_per_value / 1e9
    # Simplified: For a typical LLM, ~0.002 GB per 1000 tokens at FP16
    # Use actual context_used if available, otherwise use a reasonable default (8K)
    effective_context = context_used if context_used > 0 else min(context, 8192)
    kv_cache_gb = (effective_context / 1000) * 0.002
    
    # 3. System Overhead (Ollama runtime, etc.)
    overhead_gb = 0.3

    total_gb = weight_gb + kv_cache_gb + overhead_gb
    
    if total_gb < 1: return f"{total_gb * 1024:.0f} MB"
    return f"{total_gb:.1f} GB"

def get_model_info(name: str, disk_size: str) -> Dict:
    try:
        raw_show = get_cmd_output(['ollama', 'show', name])
    except Exception as e:
        return {
            'model': name,
            'disk': disk_size,
            'family': 'ERROR',
            'params_str': 'N/A',
            'quant': 'N/A',
            'context': 0,
            'context_used': 0,
            'caps': [],
            'vram': 'N/A'
        }
    
    info = {
        'model': name,
        'disk': disk_size,
        'family': 'N/A',
        'params_str': 'N/A',
        'quant': 'N/A',
        'context': 0,
        'context_used': 0,  # Actual context from Parameters section
        'caps': []
    }

    # -- State Machine Parsing --
    current_section = None
    
    lines = raw_show.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Detect Sections
        if line in ["Model", "Capabilities", "Parameters", "System", "License"]:
            current_section = line
            continue

        # Parse 'Model' Section
        if current_section == "Model":
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                k, v = parts[0].lower(), parts[1].strip()
                if 'architecture' in k: info['family'] = v
                elif 'parameters' in k: info['params_str'] = v
                elif 'quantization' in k: info['quant'] = v
                elif 'context' in k and 'length' in k:
                    if v.isdigit(): info['context'] = int(v)
            
            # Fallback regex for context
            if 'context' in line.lower() and info['context'] == 0:
                match = re.search(r'context\s+length\s+(\d+)', line, re.IGNORECASE)
                if match: info['context'] = int(match.group(1))

        # Parse 'Parameters' Section (runtime config)
        elif current_section == "Parameters":
            if 'num_ctx' in line.lower():
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip().isdigit():
                    info['context_used'] = int(parts[1].strip())

        # Parse 'Capabilities' Section
        elif current_section == "Capabilities":
            cap = line.lower()
            if cap in ['tools', 'vision', 'thinking', 'insert']:
                info['caps'].append(cap.capitalize())

    # -- VRAM Calc --
    p_val = parse_parameters(info['params_str'])
    info['vram'] = estimate_vram(p_val, info['quant'], info['context'], info['context_used'])

    return info

def main():
    print("Fetching Ollama inventory...")
    list_out = get_cmd_output(['ollama', 'list'])
    
    data = []
    lines = list_out.split('\n')[1:]
    
    for line in lines:
        if not line.strip(): continue
        parts = line.split()
        if len(parts) >= 3:
            name = parts[0]
            disk = parts[2]
            print(f"   Analyzing {name}...", end='\r')
            data.append(get_model_info(name, disk))
            
    print(" " * 60, end='\r')

    # Formatting Table
    w = {'m': 38, 'a': 12, 'p': 8, 'q': 10, 'ctx': 12, 'cp': 18, 'd': 8, 'v': 8}
    
    header = (f"{'MODEL':<{w['m']}} {'ARCH':<{w['a']}} {'PARAMS':<{w['p']}} "
              f"{'QUANT':<{w['q']}} {'CONTEXT':<{w['ctx']}} {'CAPS':<{w['cp']}} "
              f"{'DISK':>{w['d']}} {'VRAM':>{w['v']}}")
    
    print(header)
    print("-" * len(header))

    for r in data:
        caps_str = ", ".join(r['caps']) if r['caps'] else "-"
        # Truncate overly long names
        d_name = (r['model'][:w['m']-2] + '..') if len(r['model']) > w['m'] else r['model']
        
        # Format context: show used/max or just max if used not set
        if r['context_used'] > 0:
            ctx_str = f"{r['context_used']}/{r['context']}"
        else:
            ctx_str = str(r['context'])
        
        print(f"{d_name:<{w['m']}} {r['family']:<{w['a']}} {r['params_str']:<{w['p']}} "
              f"{r['quant']:<{w['q']}} {ctx_str:<{w['ctx']}} {caps_str:<{w['cp']}} "
              f"{r['disk']:>{w['d']}} {r['vram']:>{w['v']}}")

if __name__ == "__main__":
    main()
