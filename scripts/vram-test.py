#!/usr/bin/env python3
"""
Ollama VRAM Test - Evaluate if models fit in VRAM
Tests models with their configured parameters and reports VRAM usage and CPU offloading.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional


def get_ollama_url():
    """Get the Ollama API URL."""
    return "http://localhost:11434"


def get_installed_models() -> List[str]:
    """Get list of installed Ollama models."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        
        models = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if line.strip():
                name = line.split()[0]
                models.append(name)
        
        return models
    except subprocess.CalledProcessError:
        return []


def get_model_info(model_name: str) -> Dict:
    """Get model information from ollama show."""
    try:
        result = subprocess.run(
            ['ollama', 'show', model_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        info = {
            'size': 'N/A',
            'quant': 'N/A',
            'num_ctx': 'N/A',
            'params': 'N/A'
        }
        
        current_section = None
        for line in result.stdout.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line in ["Model", "Parameters"]:
                current_section = line
                continue
            
            if current_section == "Model":
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    k, v = parts[0].lower(), parts[1].strip()
                    if 'quantization' in k:
                        info['quant'] = v
                    elif 'parameters' in k:
                        info['params'] = v
            
            elif current_section == "Parameters":
                if 'num_ctx' in line.lower():
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        info['num_ctx'] = parts[1].strip()
        
        return info
    except subprocess.CalledProcessError:
        return {'size': 'N/A', 'quant': 'N/A', 'num_ctx': 'N/A', 'params': 'N/A'}


def test_model_vram(model_name: str) -> Dict:
    """
    Test a model's VRAM usage by loading it with a minimal prompt.
    Returns dict with model stats and VRAM usage.
    """
    print(f"Testing {model_name}...", end=' ', flush=True)
    
    # Get model info first
    info = get_model_info(model_name)
    
    # Send a minimal test prompt to force model loading
    url = f"{get_ollama_url()}/api/generate"
    prompt_data = {
        "model": model_name,
        "prompt": "Reply with only: OK",
        "stream": False
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(prompt_data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        # Send request and wait for model to load
        with urllib.request.urlopen(req, timeout=30) as response:
            response.read()  # Wait for completion
        
        # Give it a moment to stabilize
        time.sleep(0.5)
        
        # Now check /api/ps for VRAM usage
        ps_url = f"{get_ollama_url()}/api/ps"
        with urllib.request.urlopen(ps_url, timeout=5) as r:
            ps_data = json.loads(r.read().decode())
            models = ps_data.get('models', [])
            
            # Find our model in the running models
            for m in models:
                if m['name'] == model_name or m['name'].startswith(model_name + ':'):
                    size_bytes = m.get('size', 0)
                    size_vram = m.get('size_vram', 0)
                    
                    # Calculate VRAM usage in GB
                    vram_gb = size_vram / (1024**3) if size_vram > 0 else 0
                    total_gb = size_bytes / (1024**3) if size_bytes > 0 else 0
                    
                    # Calculate offload percentage (how much is on CPU)
                    if size_bytes > 0:
                        offload_pct = ((size_bytes - size_vram) / size_bytes) * 100
                    else:
                        offload_pct = 0
                    
                    print("✓")
                    
                    return {
                        'model': model_name,
                        'params': info['params'],
                        'size_gb': total_gb,
                        'quant': info['quant'],
                        'num_ctx': info['num_ctx'],
                        'vram_gb': vram_gb,
                        'offload_pct': offload_pct,
                        'success': True
                    }
        
        # Model not found in ps output
        print("✗ (not in ps)")
        return {
            'model': model_name,
            'params': info['params'],
            'size_gb': 0,
            'quant': info['quant'],
            'num_ctx': info['num_ctx'],
            'vram_gb': 0,
            'offload_pct': 0,
            'success': False
        }
        
    except Exception as e:
        print(f"✗ ({str(e)[:30]})")
        return {
            'model': model_name,
            'params': info['params'],
            'size_gb': 0,
            'quant': info['quant'],
            'num_ctx': info['num_ctx'],
            'vram_gb': 0,
            'offload_pct': 0,
            'success': False
        }


def main():
    parser = argparse.ArgumentParser(
        description='Test Ollama models for VRAM usage and CPU offloading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all installed models
  %(prog)s
  
  # Test a specific model
  %(prog)s ministral-3:3b-instruct-2512-q5_k_m
        """
    )
    
    parser.add_argument(
        'model',
        nargs='?',
        help='Specific model to test (optional, tests all if omitted)'
    )
    
    args = parser.parse_args()
    
    # Check if ollama is available
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Error: 'ollama' command not found. Please install Ollama first.")
        sys.exit(1)
    
    # Determine which models to test
    if args.model:
        models = [args.model]
    else:
        models = get_installed_models()
        if not models:
            print("✗ No models found")
            sys.exit(1)
        print(f"Found {len(models)} installed model(s)\n")
    
    # Test each model
    results = []
    for model in models:
        result = test_model_vram(model)
        results.append(result)
    
    # Display results table
    print("\n" + "="*110)
    print("VRAM USAGE TEST RESULTS")
    print("="*110)
    
    # Column widths
    w = {'m': 38, 'p': 8, 's': 10, 'q': 10, 'ctx': 10, 'v': 10, 'o': 12}
    
    header = (f"{'MODEL':<{w['m']}} {'PARAMS':<{w['p']}} {'SIZE':<{w['s']}} "
              f"{'QUANT':<{w['q']}} {'NUM_CTX':<{w['ctx']}} {'VRAM':>{w['v']}} {'OFFLOAD':>{w['o']}}")
    
    print(header)
    print("-" * 110)
    
    for r in results:
        # Truncate long model names
        name = (r['model'][:w['m']-2] + '..') if len(r['model']) > w['m'] else r['model']
        
        # Format values
        size_str = f"{r['size_gb']:.1f} GB" if r['size_gb'] > 0 else "N/A"
        vram_str = f"{r['vram_gb']:.1f} GB" if r['vram_gb'] > 0 else "N/A"
        
        # Offload status
        if r['success']:
            if r['offload_pct'] > 0:
                offload_str = f"{r['offload_pct']:.1f}% CPU"
            else:
                offload_str = "0% (GPU only)"
        else:
            offload_str = "Failed"
        
        print(f"{name:<{w['m']}} {r['params']:<{w['p']}} {size_str:<{w['s']}} "
              f"{r['quant']:<{w['q']}} {r['num_ctx']:<{w['ctx']}} {vram_str:>{w['v']}} {offload_str:>{w['o']}}")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    with_offload = sum(1 for r in results if r['success'] and r['offload_pct'] > 0)
    
    print("\n" + "="*110)
    print(f"Tested: {len(results)} | Successful: {successful} | CPU Offloading: {with_offload}")
    
    if with_offload > 0:
        print(f"\n⚠  {with_offload} model(s) using CPU offloading - consider reducing num_ctx or using smaller quantization")


if __name__ == '__main__':
    main()
