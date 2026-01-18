#!/usr/bin/env python3
"""
Ollama Context Optimizer - Find optimal num_ctx for models based on VRAM
Iteratively tests different context sizes to recommend the best setting.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from typing import Dict, Optional, Tuple


def get_ollama_url():
    """Get the Ollama API URL."""
    return "http://localhost:11434"


def get_gpu_vram() -> Tuple[Optional[float], Optional[float]]:
    """Get GPU VRAM total and available in GB."""
    import os
    
    device_paths = [
        "/sys/class/drm/card1/device/",
        "/sys/class/drm/card0/device/",
    ]
    
    for base_path in device_paths:
        if not os.path.exists(base_path):
            continue
        
        try:
            with open(base_path + "mem_info_vram_used", "r") as f:
                used = int(f.read().strip()) / 1024 / 1024 / 1024  # GB
            with open(base_path + "mem_info_vram_total", "r") as f:
                total = int(f.read().strip()) / 1024 / 1024 / 1024  # GB
            
            available = total - used
            return total, available
        except:
            continue
    
    return None, None


def get_model_info(model_name: str) -> Dict:
    """Get model information including max context capability."""
    try:
        result = subprocess.run(
            ['ollama', 'show', model_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        info = {
            'max_context': 0,
            'current_num_ctx': 0,
            'params': 'N/A',
            'quant': 'N/A'
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
                    if 'context' in k and 'length' in k:
                        if v.isdigit():
                            info['max_context'] = int(v)
                    elif 'context' in k:
                        # Handle "context length" as two words
                        parts2 = line.split()
                        if len(parts2) >= 3 and parts2[-1].isdigit():
                            info['max_context'] = int(parts2[-1])
                    elif 'quantization' in k:
                        info['quant'] = v
                    elif 'parameters' in k:
                        info['params'] = v
            
            elif current_section == "Parameters":
                if 'num_ctx' in line.lower():
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and parts[1].strip().isdigit():
                        info['current_num_ctx'] = int(parts[1].strip())
        
        return info
    except subprocess.CalledProcessError:
        return {'max_context': 0, 'current_num_ctx': 0, 'params': 'N/A', 'quant': 'N/A'}


def test_context_size(model_name: str, num_ctx: int) -> Optional[Dict]:
    """
    Test a model with a specific context size.
    Returns VRAM usage and offload info, or None if failed.
    """
    url = f"{get_ollama_url()}/api/generate"
    prompt_data = {
        "model": model_name,
        "prompt": "Reply with only: OK",
        "stream": False,
        "options": {
            "num_ctx": num_ctx
        }
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(prompt_data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        # Send request with longer timeout for large contexts
        # Large contexts can take time to allocate
        timeout = 60 if num_ctx > 100000 else 30
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = response.read().decode()
            
            # Check if response contains error
            try:
                resp_json = json.loads(response_data)
                if 'error' in resp_json:
                    error_msg = resp_json['error']
                    # Return special dict to indicate OOM or other errors
                    return {
                        'vram_gb': 0,
                        'total_gb': 0,
                        'offload_pct': 0,
                        'num_ctx': num_ctx,
                        'error': error_msg
                    }
            except:
                pass
        
        # Wait for model to stabilize
        time.sleep(0.5)
        
        # Check /api/ps for VRAM usage
        ps_url = f"{get_ollama_url()}/api/ps"
        with urllib.request.urlopen(ps_url, timeout=5) as r:
            ps_data = json.loads(r.read().decode())
            models = ps_data.get('models', [])
            
            for m in models:
                if m['name'] == model_name or m['name'].startswith(model_name + ':'):
                    size_bytes = m.get('size', 0)
                    size_vram = m.get('size_vram', 0)
                    
                    vram_gb = size_vram / (1024**3)
                    total_gb = size_bytes / (1024**3)
                    
                    offload_pct = 0
                    if size_bytes > 0:
                        offload_pct = ((size_bytes - size_vram) / size_bytes) * 100
                    
                    return {
                        'vram_gb': vram_gb,
                        'total_gb': total_gb,
                        'offload_pct': offload_pct,
                        'num_ctx': num_ctx
                    }
        
        return None
        
    except urllib.error.HTTPError as e:
        # HTTP errors (500, etc.) - often indicates OOM or model loading failure
        try:
            error_body = e.read().decode()
            error_data = json.loads(error_body)
            error_msg = error_data.get('error', str(e))
        except:
            error_msg = f"HTTP {e.code}"
        
        return {
            'vram_gb': 0,
            'total_gb': 0,
            'offload_pct': 0,
            'num_ctx': num_ctx,
            'error': error_msg
        }
    
    except urllib.error.URLError as e:
        # Network/timeout errors
        if 'timed out' in str(e).lower():
            error_msg = "Timeout (loading too slow)"
        else:
            error_msg = f"Connection error: {e}"
        
        return {
            'vram_gb': 0,
            'total_gb': 0,
            'offload_pct': 0,
            'num_ctx': num_ctx,
            'error': error_msg
        }
    
    except Exception as e:
        # Other unexpected errors
        return {
            'vram_gb': 0,
            'total_gb': 0,
            'offload_pct': 0,
            'num_ctx': num_ctx,
            'error': str(e)[:50]
        }


def find_optimal_context(model_name: str, max_turns: Optional[int], overhead_gb: float) -> Dict:
    """
    Find optimal context size through intelligent testing.
    Uses VRAM measurements to extrapolate optimal size.
    
    Args:
        model_name: Name of the Ollama model to test
        max_turns: Maximum iterations (None = optimize until convergence)
        overhead_gb: VRAM to keep free for system overhead
    """
    print(f"Analyzing model: {model_name}")
    print("-" * 70)
    
    # Get model capabilities
    info = get_model_info(model_name)
    max_context = info['max_context']
    current_ctx = info['current_num_ctx']
    
    print(f"Model: {model_name}")
    print(f"Parameters: {info['params']} ({info['quant']})")
    print(f"Max context capability: {max_context:,}")
    print(f"Current num_ctx: {current_ctx:,}")
    
    if max_context == 0:
        print("\n✗ Could not determine model's max context capability")
        return {}
    
    # Get VRAM info
    vram_total, vram_available = get_gpu_vram()
    if vram_total:
        print(f"GPU VRAM: {vram_available:.1f} GB available / {vram_total:.1f} GB total")
        print(f"Overhead reserved: {overhead_gb:.1f} GB")
        # Reserve specified overhead
        target_vram = vram_total - overhead_gb
    else:
        print("⚠ Could not detect GPU VRAM (testing will continue)")
        target_vram = None
    
    if max_turns:
        print(f"Testing with max {max_turns} iterations...")
    else:
        print(f"Testing until convergence (num_ctx must be multiple of 2048)...")
    print()
    
    results = []
    
    # Turn 1: Test current setting to establish baseline
    test_ctx = current_ctx if current_ctx > 0 else 8192
    turn_label = f"Turn 1/{max_turns}" if max_turns else "Turn 1"
    print(f"{turn_label}: Testing num_ctx={test_ctx:,} (baseline)...", end=' ', flush=True)
    result = test_context_size(model_name, test_ctx)
    
    if result and 'error' not in result:
        results.append(result)
        print(f"✓ VRAM: {result['vram_gb']:.2f} GB, Offload: {result['offload_pct']:.1f}% CPU" if result['offload_pct'] > 0 else f"✓ VRAM: {result['vram_gb']:.2f} GB, Offload: GPU only")
        baseline_vram = result['vram_gb']
        baseline_ctx = test_ctx
    else:
        print("✗ Failed")
        return {'model': model_name, 'results': results, 'max_context': max_context, 'current_ctx': current_ctx, 'vram_total': vram_total, 'info': info}
    
    # Turn 2: Test a higher context to calculate VRAM/context ratio
    # Try doubling the context or 32K, whichever is smaller
    test_ctx_2 = min(baseline_ctx * 2, 32768, max_context)
    if test_ctx_2 <= baseline_ctx:
        test_ctx_2 = min(baseline_ctx + 16384, max_context)
    # Round to multiple of 2048
    test_ctx_2 = (test_ctx_2 // 2048) * 2048
    
    turn_label = f"Turn 2/{max_turns}" if max_turns else "Turn 2"
    print(f"{turn_label}: Testing num_ctx={test_ctx_2:,} (calibration)...", end=' ', flush=True)
    result = test_context_size(model_name, test_ctx_2)
    
    if result and 'error' not in result:
        results.append(result)
        print(f"✓ VRAM: {result['vram_gb']:.2f} GB, Offload: {result['offload_pct']:.1f}% CPU" if result['offload_pct'] > 0 else f"✓ VRAM: {result['vram_gb']:.2f} GB, Offload: GPU only")
        
        # Calculate VRAM per 1K context tokens
        vram_diff = result['vram_gb'] - baseline_vram
        ctx_diff = test_ctx_2 - baseline_ctx
        if ctx_diff > 0:
            vram_per_1k_ctx = (vram_diff / ctx_diff) * 1000
            print(f"         → Estimated VRAM usage: {vram_per_1k_ctx:.4f} GB per 1K context")
            
            # Predict optimal context size
            if target_vram and vram_per_1k_ctx > 0:
                available_for_ctx = target_vram - baseline_vram
                estimated_additional_ctx = (available_for_ctx / vram_per_1k_ctx) * 1000
                predicted_optimal = baseline_ctx + int(estimated_additional_ctx)
                # Round to multiple of 2048
                predicted_optimal = (predicted_optimal // 2048) * 2048
                predicted_optimal = max(baseline_ctx, min(predicted_optimal, max_context))
                
                print(f"         → Predicted optimal context: {predicted_optimal:,}")
            else:
                predicted_optimal = None
                vram_per_1k_ctx = None
        else:
            vram_per_1k_ctx = None
            predicted_optimal = None
    else:
        if result and 'error' in result:
            error_msg = result['error']
            if 'memory' in error_msg.lower() or 'oom' in error_msg.lower():
                print(f"✗ OOM (out of memory)")
            else:
                print(f"✗ Error: {error_msg[:30]}")
        else:
            print("✗ Failed")
        vram_per_1k_ctx = None
        predicted_optimal = None
    
    # Remaining turns: Test predicted optimal or use VRAM-based refinement
    min_ctx = baseline_ctx
    max_ctx = max_context
    
    turn = 2
    while True:
        # Check if we should stop
        if max_turns and turn >= max_turns:
            break
        
        if predicted_optimal and turn == 2:
            # Turn 3: Test predicted optimal
            test_ctx = predicted_optimal
            turn_label = f"Turn {turn + 1}/{max_turns}" if max_turns else f"Turn {turn + 1}"
            print(f"{turn_label}: Testing num_ctx={test_ctx:,} (predicted optimal)...", end=' ', flush=True)
        else:
            # Use VRAM-based prediction if we have the data
            if vram_per_1k_ctx and target_vram and len(results) > 0:
                # Find the last successful result (no offload)
                last_good = None
                for r in reversed(results):
                    if r['offload_pct'] == 0:
                        last_good = r
                        break
                
                if last_good and target_vram:
                    # Calculate how much more context we can realistically add
                    available_vram = target_vram - last_good['vram_gb']
                    
                    # Calculate potential additional context
                    additional_ctx = (available_vram / vram_per_1k_ctx) * 1000
                    
                    # If we can only add < 8K context, do small increments
                    if additional_ctx < 8192:
                        # Small increments - round up to next 2048 boundary
                        test_ctx = last_good['num_ctx'] + 2048
                        test_ctx = max(min_ctx + 2048, min(test_ctx, max_ctx))
                    else:
                        # Larger headroom - use 60% of predicted to be conservative
                        test_ctx = last_good['num_ctx'] + int(additional_ctx * 0.6)
                        test_ctx = (test_ctx // 2048) * 2048
                        test_ctx = max(min_ctx + 2048, min(test_ctx, max_ctx))
                else:
                    # No good result yet - binary search
                    test_ctx = (min_ctx + max_ctx) // 2
                    test_ctx = (test_ctx // 2048) * 2048
            else:
                # No VRAM data - fall back to binary search
                test_ctx = (min_ctx + max_ctx) // 2
                test_ctx = (test_ctx // 2048) * 2048
            
            # Avoid retesting same value
            if any(r['num_ctx'] == test_ctx for r in results):
                # Adjust by 2048
                if test_ctx < max_ctx:
                    test_ctx += 2048
                else:
                    test_ctx -= 2048
                    
                if test_ctx <= min_ctx or test_ctx >= max_ctx:
                    print(f"\nConverged after {turn + 1} turns")
                    break
            
            turn_label = f"Turn {turn + 1}/{max_turns}" if max_turns else f"Turn {turn + 1}"
            print(f"{turn_label}: Testing num_ctx={test_ctx:,}...", end=' ', flush=True)
        
        result = test_context_size(model_name, test_ctx)
        
        if result is None:
            print("✗ Failed (model not found)")
            max_ctx = test_ctx
            continue
        
        if 'error' in result:
            error_msg = result['error']
            if 'memory' in error_msg.lower() or 'oom' in error_msg.lower():
                print(f"✗ OOM (out of memory)")
            elif 'timeout' in error_msg.lower():
                print(f"✗ Timeout")
            else:
                print(f"✗ Error: {error_msg[:30]}")
            max_ctx = test_ctx
            continue
        
        results.append(result)
        
        offload_str = f"{result['offload_pct']:.1f}% CPU" if result['offload_pct'] > 0 else "GPU only"
        print(f"✓ VRAM: {result['vram_gb']:.2f} GB, Offload: {offload_str}")
        
        # Adjust search bounds
        if result['offload_pct'] > 0:
            max_ctx = test_ctx
        else:
            min_ctx = test_ctx
        
        # Stop if we're converging (within one step of 2048)
        if max_ctx - min_ctx <= 2048:
            print(f"\nConverged after {turn + 1} turns")
            break
        
        turn += 1
    
    return {
        'model': model_name,
        'results': results,
        'max_context': max_context,
        'current_ctx': current_ctx,
        'vram_total': vram_total,
        'info': info
    }


def print_recommendation(analysis: Dict):
    """Print optimization recommendations."""
    if not analysis or not analysis.get('results'):
        print("\n✗ No results to analyze")
        return
    
    results = analysis['results']
    max_context = analysis['max_context']
    current_ctx = analysis['current_ctx']
    
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATION")
    print("="*70)
    
    # Find best context without offloading
    no_offload = [r for r in results if r['offload_pct'] == 0]
    
    if no_offload:
        # Recommend highest context without offloading
        best = max(no_offload, key=lambda x: x['num_ctx'])
        
        print(f"\n✓ Recommended num_ctx: {best['num_ctx']:,}")
        print(f"  VRAM usage: {best['vram_gb']:.2f} GB")
        print(f"  Status: Fits entirely in GPU memory")
        
        if best['num_ctx'] < max_context:
            print(f"\n⚠ Note: Model supports up to {max_context:,} context")
            print(f"  but VRAM limits optimal usage to {best['num_ctx']:,}")
        
        if current_ctx != best['num_ctx']:
            print(f"\n📝 Suggested Modelfile change:")
            print(f"   Current: PARAMETER num_ctx {current_ctx}")
            print(f"   Optimal: PARAMETER num_ctx {best['num_ctx']}")
    else:
        # All tests had offloading
        print("\n⚠ All tested configurations require CPU offloading")
        
        # Find least offloading
        least_offload = min(results, key=lambda x: x['offload_pct'])
        
        print(f"\n  Least offloading at num_ctx={least_offload['num_ctx']:,}")
        print(f"  CPU offload: {least_offload['offload_pct']:.1f}%")
        print(f"  VRAM usage: {least_offload['vram_gb']:.2f} GB")
        
        print(f"\n💡 Recommendations:")
        print(f"   1. Use lower quantization (Q4 instead of Q5/Q8)")
        print(f"   2. Reduce num_ctx to {least_offload['num_ctx']:,} or lower")
        print(f"   3. Consider a smaller model variant")
    
    # VRAM efficiency
    print(f"\n📊 Tested context sizes:")
    for r in sorted(results, key=lambda x: x['num_ctx']):
        status = "✓" if r['offload_pct'] == 0 else "✗"
        print(f"   {status} {r['num_ctx']:>6,}: {r['vram_gb']:>5.2f} GB VRAM, "
              f"{r['offload_pct']:>4.1f}% CPU offload")


def main():
    parser = argparse.ArgumentParser(
        description='Optimize Ollama model context size for VRAM constraints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize with 5 test iterations (default)
  %(prog)s ministral-3:3b-instruct-2512-q5_k_m
  
  # Use 10 iterations for more precise optimization
  %(prog)s ministral-3:3b-instruct-2512-q5_k_m --turns 10
        """
    )
    
    parser.add_argument(
        'model',
        help='Model name to optimize'
    )
    
    parser.add_argument(
        '--turns',
        type=int,
        default=None,
        help='Maximum number of test iterations (default: optimize until convergence)'
    )
    
    parser.add_argument(
        '--overhead',
        type=float,
        default=1.0,
        help='VRAM overhead to keep free in GB (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    if args.turns is not None and args.turns < 2:
        print("✗ Error: --turns must be at least 2")
        sys.exit(1)
    
    # Check if ollama is available
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Error: 'ollama' command not found. Please install Ollama first.")
        sys.exit(1)
    
    # Run optimization
    analysis = find_optimal_context(args.model, args.turns, args.overhead)
    
    # Print recommendations
    print_recommendation(analysis)


if __name__ == '__main__':
    main()
