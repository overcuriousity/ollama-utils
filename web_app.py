#!/usr/bin/env python3
"""
Ollama-Utils Web Interface
A comprehensive web interface for managing Ollama models and monitoring system resources.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from flask import Flask, render_template, jsonify, request, send_from_directory
from urllib.parse import urlparse

# Import utilities from existing scripts
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import existing CLI tools
import importlib.util

# Load vram-test module
vram_test_spec = importlib.util.spec_from_file_location("vram_test", os.path.join(os.path.dirname(__file__), 'scripts', 'vram-test.py'))
vram_test_module = importlib.util.module_from_spec(vram_test_spec)
vram_test_spec.loader.exec_module(vram_test_module)

# Load context-optimizer module
context_optimizer_spec = importlib.util.spec_from_file_location("context_optimizer", os.path.join(os.path.dirname(__file__), 'scripts', 'context-optimizer.py'))
context_optimizer_module = importlib.util.module_from_spec(context_optimizer_spec)
context_optimizer_spec.loader.exec_module(context_optimizer_module)

# Load model-info module
model_info_spec = importlib.util.spec_from_file_location("model_info", os.path.join(os.path.dirname(__file__), 'scripts', 'model-info.py'))
model_info_module = importlib.util.module_from_spec(model_info_spec)
model_info_spec.loader.exec_module(model_info_module)

# Load hf-llm-install module
hf_install_spec = importlib.util.spec_from_file_location("hf_install", os.path.join(os.path.dirname(__file__), 'scripts', 'hf-llm-install.py'))
hf_install_module = importlib.util.module_from_spec(hf_install_spec)
hf_install_spec.loader.exec_module(hf_install_module)

app = Flask(__name__)
app.config['MODELFILE_REPO'] = os.path.join(os.path.dirname(__file__), 'modelfile-repo')

# Global state for background installations
install_jobs = {}
install_lock = threading.Lock()

# ===== UTILITY FUNCTIONS =====

def get_ollama_url():
    """Get the Ollama API URL."""
    return "http://localhost:11434"


def get_gpu_metrics() -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Get GPU VRAM and load metrics."""
    try:
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
                
                # Sanity check
                if load == 99 and used < (total * 0.1):
                    load = 0
                    
                return used, total, load
            except:
                continue
                
        return None, None, None
    except:
        return None, None, None


def get_sys_metrics() -> Tuple[float, int, int]:
    """Get system CPU and RAM metrics."""
    try:
        load_avg = os.getloadavg()[0]
        mem_output = subprocess.check_output("free -m", shell=True).decode().split('\n')[1].split()
        ram_used = int(mem_output[2])
        ram_total = int(mem_output[1])
        return load_avg, ram_used, ram_total
    except Exception:
        return 0.0, 0, 0


def get_model_info_detailed(model_name: str) -> Dict:
    """Get detailed model information from 'ollama show'. Uses model-info.py logic."""
    # Get basic list info first
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        disk_size = 'N/A'
        for line in result.stdout.strip().split('\n')[1:]:
            if line.strip():
                parts = line.split()
                if len(parts) >= 3 and parts[0] == model_name:
                    disk_size = parts[2]
                    break
    except:
        disk_size = 'N/A'
    
    # Use the existing get_model_info function from model-info.py
    info = model_info_module.get_model_info(model_name, disk_size)
    
    # Convert to expected format (model-info uses slightly different keys)
    return {
        'name': model_name,
        'family': info.get('family', 'N/A'),
        'params': info.get('params_str', 'N/A'),
        'quant': info.get('quant', 'N/A'),
        'max_context': info.get('context', 0),
        'context_used': info.get('context_used', 0),
        'capabilities': [cap for cap in info.get('caps', [])],
        'license': 'N/A',
        'system_prompt': '',
        'vram_estimate': info.get('vram', 'N/A')
    }


def check_modelfile_exists(model_name: str) -> Optional[str]:
    """Check if a Modelfile exists for this model in the modelfile-repo directory."""
    modelfile_dir = app.config['MODELFILE_REPO']
    if not os.path.exists(modelfile_dir):
        return None
    
    # Try exact match first
    modelfile_path = os.path.join(modelfile_dir, f"{model_name}.Modelfile")
    if os.path.exists(modelfile_path):
        return modelfile_path
    
    # Try with colons replaced by dashes (ministral-3:3b -> ministral-3-3b)
    normalized_name = model_name.replace(':', '-')
    modelfile_path = os.path.join(modelfile_dir, f"{normalized_name}.Modelfile")
    if os.path.exists(modelfile_path):
        return modelfile_path
    
    return None


def parse_modelfile_metadata(modelfile_path: str) -> Dict:
    """Parse metadata from a Modelfile using hf-llm-install.py logic."""
    try:
        # Use the existing parse_modelfile function from hf-llm-install.py
        model_info = hf_install_module.parse_modelfile(modelfile_path)
        
        if not model_info:
            return None
        
        # Extract quantization and other params from the modelfile content
        quantization = None
        num_ctx = None
        family = None
        params = None
        
        with open(modelfile_path, 'r') as f:
            content = f.read()
            
            # Extract quantization
            quant_match = re.search(r'#\s*quantization:\s*([a-zA-Z0-9_]+)', content)
            if quant_match:
                quantization = quant_match.group(1).upper()
            else:
                # Extract from filename if not specified
                gguf_filename = model_info.get('gguf_filename', '')
                quant_pattern = re.search(r'[_-](Q[0-9]+_[KLM]+(?:_[LSM])?)\\.gguf', gguf_filename, re.IGNORECASE)
                if quant_pattern:
                    quantization = quant_pattern.group(1).upper()
            
            # Extract num_ctx
            ctx_match = re.search(r'PARAMETER\s+num_ctx\s+(\d+)', content)
            if ctx_match:
                num_ctx = int(ctx_match.group(1))
        
        # Extract params and family from model name
        model_name = model_info['model_name']
        # Pattern: modelbase:Xb-variant  (e.g., "ministral-3:3b-instruct-2512-q5_k_m")
        params_match = re.search(r':(\d+)b', model_name, re.IGNORECASE)
        if params_match:
            params = params_match.group(1) + 'B'
        
        # Extract family from base name
        if ':' in model_name:
            family = model_name.split(':')[0].upper()
        
        # Get capabilities from model_info (parsed by hf_install_module)
        capabilities = model_info.get('capabilities', [])
        
        # Convert to expected format
        return {
            'path': modelfile_path,
            'filename': os.path.basename(modelfile_path),
            'model_name': model_info['model_name'],
            'hf_upstream': model_info.get('hf_url'),
            'quantization': quantization or 'unspecified',
            'sha256': model_info.get('sha256'),
            'num_ctx': num_ctx or 0,
            'family': family or 'Unknown',
            'params': params or 'Unknown',
            'capabilities': capabilities or []
        }
    except Exception as e:
        return None


def get_all_modelfiles() -> List[Dict]:
    """Get all modelfiles from the modelfile-repo directory."""
    modelfile_dir = app.config['MODELFILE_REPO']
    if not os.path.exists(modelfile_dir):
        return []
    
    modelfiles = []
    for filename in os.listdir(modelfile_dir):
        if filename.endswith('.Modelfile'):
            filepath = os.path.join(modelfile_dir, filename)
            metadata = parse_modelfile_metadata(filepath)
            if metadata:
                modelfiles.append(metadata)
    
    return modelfiles


def run_install_job(job_id: str, modelfile_path: str):
    """Run installation in background thread."""
    with install_lock:
        install_jobs[job_id]['status'] = 'running'
        install_jobs[job_id]['progress'] = 'Starting installation...'
    
    # Progress callback
    def update_progress(message):
        with install_lock:
            install_jobs[job_id]['progress'] = message
    
    # Cancellation callback
    def should_cancel():
        with install_lock:
            return install_jobs[job_id].get('cancelled', False)
    
    try:
        success, skipped, model_name = hf_install_module.install_model(
            modelfile_path,
            dry_run=False,
            skip_existing=False,
            existing_models=None,
            should_cancel=should_cancel,
            progress_callback=update_progress
        )
        
        with install_lock:
            if success:
                install_jobs[job_id]['status'] = 'completed'
                install_jobs[job_id]['model_name'] = model_name
                install_jobs[job_id]['progress'] = f'Successfully installed {model_name}'
            else:
                install_jobs[job_id]['status'] = 'failed'
                install_jobs[job_id]['error'] = f'Installation failed for {model_name}'
            
    except InterruptedError as e:
        with install_lock:
            install_jobs[job_id]['status'] = 'cancelled'
            install_jobs[job_id]['error'] = str(e)
    except Exception as e:
        with install_lock:
            # Check if it was actually cancelled before marking as failed
            if install_jobs[job_id].get('cancelled', False):
                install_jobs[job_id]['status'] = 'cancelled'
                install_jobs[job_id]['error'] = 'Installation cancelled by user'
            else:
                install_jobs[job_id]['status'] = 'failed'
                install_jobs[job_id]['error'] = str(e)


def run_huggingface_install_job(job_id: str, model_name: str, modelfile_content: str, file_url: str, gguf_filename: str):
    """Run HuggingFace model installation in background thread."""
    with install_lock:
        install_jobs[job_id]['status'] = 'running'
        install_jobs[job_id]['progress'] = 'Starting download...'
    
    # Progress callback
    def update_progress(message):
        with install_lock:
            install_jobs[job_id]['progress'] = message
    
    # Cancellation callback
    def should_cancel():
        with install_lock:
            return install_jobs[job_id].get('cancelled', False)
    
    temp_gguf = None
    temp_modelfile = None
    
    try:
        # Create temp files
        import tempfile
        temp_gguf = tempfile.NamedTemporaryFile(suffix='.gguf', delete=False)
        temp_gguf.close()
        gguf_path = temp_gguf.name
        
        temp_modelfile = tempfile.NamedTemporaryFile(mode='w', suffix='.Modelfile', delete=False)
        temp_modelfile.write(modelfile_content)
        temp_modelfile.close()
        modelfile_path = temp_modelfile.name
        
        # Use existing download_file function with callbacks
        hf_install_module.download_file(file_url, gguf_path, gguf_filename, should_cancel, update_progress)
        
        # Use existing create_ollama_model function
        hf_install_module.create_ollama_model(modelfile_path, gguf_path, model_name)
        
        # Save Modelfile to repo
        normalized_name = model_name.replace(':', '-')
        final_modelfile_path = os.path.join(app.config['MODELFILE_REPO'], f"{normalized_name}.Modelfile")
        os.makedirs(os.path.dirname(final_modelfile_path), exist_ok=True)
        with open(final_modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        with install_lock:
            install_jobs[job_id]['status'] = 'completed'
            install_jobs[job_id]['model_name'] = model_name
            install_jobs[job_id]['progress'] = f'Successfully created {model_name}'
            
    except InterruptedError as e:
        with install_lock:
            install_jobs[job_id]['status'] = 'cancelled'
            install_jobs[job_id]['error'] = 'Installation cancelled by user'
    except Exception as e:
        with install_lock:
            if install_jobs[job_id].get('cancelled', False):
                install_jobs[job_id]['status'] = 'cancelled'
                install_jobs[job_id]['error'] = 'Installation cancelled by user'
            else:
                install_jobs[job_id]['status'] = 'failed'
                install_jobs[job_id]['error'] = str(e)
    finally:
        # Clean up temp files
        if temp_gguf and os.path.exists(temp_gguf.name):
            os.unlink(temp_gguf.name)
        if temp_modelfile and os.path.exists(temp_modelfile.name):
            os.unlink(temp_modelfile.name)


# ===== WEB ROUTES =====

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Get real-time system status and running models."""
    # Get system metrics
    cpu_load, ram_used, ram_total = get_sys_metrics()
    vram_used, vram_total, gpu_load = get_gpu_metrics()
    
    # Get running models from /api/ps
    running_models = []
    try:
        url = f"{get_ollama_url()}/api/ps"
        with urllib.request.urlopen(url, timeout=2) as response:
            ps_data = json.loads(response.read().decode())
            for model in ps_data.get('models', []):
                size_vram = model.get('size_vram', 0) / (1024**3)  # GB
                size_total = model.get('size', 0) / (1024**3)  # GB
                offload_pct = ((size_total - size_vram) / size_total * 100) if size_total > 0 else 0
                
                running_models.append({
                    'name': model.get('name', 'Unknown'),
                    'size_gb': size_total,
                    'vram_gb': size_vram,
                    'offload_pct': offload_pct,
                    'expires_at': model.get('expires_at', '')
                })
    except Exception as e:
        print(f"Error getting running models: {e}")
    
    return jsonify({
        'cpu_load': round(cpu_load, 2),
        'ram_used_mb': ram_used,
        'ram_total_mb': ram_total,
        'ram_used_pct': round((ram_used / ram_total * 100) if ram_total > 0 else 0, 1),
        'vram_used_mb': round(vram_used) if vram_used is not None else None,
        'vram_total_mb': round(vram_total) if vram_total is not None else None,
        'vram_used_pct': round((vram_used / vram_total * 100) if vram_total and vram_total > 0 else 0, 1),
        'gpu_load': gpu_load,
        'running_models': running_models
    })


@app.route('/api/models')
def api_models():
    """Get list of all installed models and available modelfiles."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        
        installed_models = []
        installed_names = set()
        
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    installed_names.add(name)
                    model_id = parts[1] if len(parts) > 1 else ''
                    size = parts[2] if len(parts) > 2 else 'N/A'
                    
                    # Check if modelfile exists
                    modelfile_path = check_modelfile_exists(name)
                    has_modelfile = modelfile_path is not None
                    
                    # Get detailed info
                    detailed_info = get_model_info_detailed(name)
                    
                    installed_models.append({
                        'name': name,
                        'id': model_id,
                        'size': size,
                        'installed': True,
                        'has_modelfile': has_modelfile,
                        'modelfile_path': modelfile_path,
                        'family': detailed_info['family'],
                        'params': detailed_info['params'],
                        'quant': detailed_info['quant'],
                        'max_context': detailed_info['max_context'],
                        'context_used': detailed_info['context_used'],
                        'capabilities': detailed_info['capabilities'],
                        'vram_estimate': detailed_info['vram_estimate']
                    })
        
        # Get all modelfiles
        all_modelfiles = get_all_modelfiles()
        available_modelfiles = []
        
        for mf in all_modelfiles:
            # Check if this modelfile's model is already installed
            if mf['model_name'] not in installed_names:
                available_modelfiles.append({
                    'name': mf['model_name'],
                    'installed': False,
                    'has_modelfile': True,
                    'modelfile_path': mf['path'],
                    'hf_upstream': mf['hf_upstream'],
                    'quantization': mf['quantization'],
                    'family': mf.get('family', 'Unknown'),
                    'params': mf.get('params', 'Unknown'),
                    'quant': mf['quantization'],
                    'max_context': mf.get('num_ctx', 0),
                    'context_used': 0,
                    'capabilities': mf.get('capabilities', []),
                    'vram_estimate': 'N/A',
                    'size': 'Not installed'
                })
        
        # Combine installed and available
        all_models = installed_models + available_modelfiles
        
        return jsonify({
            'models': all_models,
            'installed_count': len(installed_models),
            'available_count': len(available_modelfiles)
        })
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<path:model_name>')
def api_model_detail(model_name):
    """Get detailed information about a specific model."""
    info = get_model_info_detailed(model_name)
    modelfile_path = check_modelfile_exists(model_name)
    
    return jsonify({
        'info': info,
        'has_modelfile': modelfile_path is not None,
        'modelfile_path': modelfile_path
    })


@app.route('/api/modelfile/<path:model_name>')
def api_get_modelfile(model_name):
    """Get the Modelfile content for a model."""
    modelfile_path = check_modelfile_exists(model_name)
    
    if not modelfile_path or not os.path.exists(modelfile_path):
        return jsonify({'error': 'Modelfile not found'}), 404
    
    try:
        with open(modelfile_path, 'r') as f:
            content = f.read()
        return jsonify({
            'path': modelfile_path,
            'content': content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/modelfile/<path:model_name>', methods=['POST'])
def api_save_modelfile(model_name):
    """Save Modelfile content and optionally recreate the model."""
    data = request.get_json()
    content = data.get('content', '')
    recreate_model = data.get('recreate_model', False)
    
    if not content:
        return jsonify({'error': 'No content provided'}), 400
    
    # Determine the modelfile path
    modelfile_path = check_modelfile_exists(model_name)
    
    if not modelfile_path:
        # Create new Modelfile
        normalized_name = model_name.replace(':', '-')
        modelfile_path = os.path.join(app.config['MODELFILE_REPO'], f"{normalized_name}.Modelfile")
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(modelfile_path), exist_ok=True)
        
        # Save the modelfile
        with open(modelfile_path, 'w') as f:
            f.write(content)
        
        # If there are changes, start background job to recreate the model
        if recreate_model:
            # Create job ID
            job_id = f"recreate_{int(time.time() * 1000)}"
            
            # Initialize job state
            with install_lock:
                install_jobs[job_id] = {
                    'status': 'queued',
                    'progress': 'Queued for recreation',
                    'modelfile_path': modelfile_path,
                    'model_name': model_name,
                    'error': None,
                    'cancelled': False
                }
            
            # Start background thread
            thread = threading.Thread(target=run_install_job, args=(job_id, modelfile_path))
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'path': modelfile_path,
                'job_id': job_id,
                'recreating': True
            })
        
        return jsonify({
            'success': True,
            'path': modelfile_path,
            'recreating': False
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<path:model_name>', methods=['DELETE'])
def api_delete_model(model_name):
    """Delete a model."""
    try:
        subprocess.run(['ollama', 'rm', model_name], check=True, capture_output=True)
        return jsonify({'success': True})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/install/ollama', methods=['POST'])
def api_install_ollama_model():
    """Install a model from Ollama library."""
    data = request.get_json()
    model_name = data.get('model_name', '')
    
    if not model_name:
        return jsonify({'error': 'No model name provided'}), 400
    
    try:
        # Run ollama pull in background
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Read output
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            return jsonify({
                'success': True,
                'message': f'Successfully pulled {model_name}',
                'output': stdout
            })
        else:
            return jsonify({
                'error': f'Failed to pull model: {stderr}'
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/install/huggingface', methods=['POST'])
def api_install_huggingface():
    """Process HuggingFace URL and return Modelfile skeleton or list of GGUF files."""
    data = request.get_json()
    hf_url = data.get('url', '')
    selected_file = data.get('selected_file', None)  # For when user selects from dropdown
    
    if not hf_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        # Parse the URL
        parsed = urlparse(hf_url)
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) < 2:
            return jsonify({'error': 'Invalid HuggingFace URL'}), 400
        
        org = path_parts[0]
        repo = path_parts[1]
        
        # Check if it's a direct GGUF file link
        if hf_url.endswith('.gguf') or '/blob/' in hf_url or '/resolve/' in hf_url:
            # Direct GGUF file URL
            gguf_filename = os.path.basename(parsed.path)
            file_url = hf_url.replace('/blob/', '/resolve/')
            
            return generate_modelfile_response(org, repo, gguf_filename, file_url)
            
        elif selected_file:
            # User selected a file from dropdown
            file_url = f"https://huggingface.co/{org}/{repo}/resolve/main/{selected_file}"
            
            return generate_modelfile_response(org, repo, selected_file, file_url)
            
        else:
            # Repository root - fetch available GGUF files
            api_url = f"https://huggingface.co/api/models/{org}/{repo}"
            
            with urllib.request.urlopen(api_url, timeout=10) as response:
                model_data = json.loads(response.read().decode())
            
            # Extract GGUF files from siblings
            gguf_files = []
            for sibling in model_data.get('siblings', []):
                filename = sibling.get('rfilename', '')
                if filename.lower().endswith('.gguf'):
                    size_bytes = sibling.get('size', 0)
                    size_gb = size_bytes / (1024**3) if size_bytes else 0
                    gguf_files.append({
                        'filename': filename,
                        'size': f"{size_gb:.2f} GB" if size_gb > 0 else "Unknown size"
                    })
            
            if not gguf_files:
                return jsonify({
                    'error': f'No GGUF files found in repository {org}/{repo}'
                }), 404
            
            # Return list of files for user to choose from
            return jsonify({
                'success': True,
                'requires_selection': True,
                'org': org,
                'repo': repo,
                'repo_url': f"https://huggingface.co/{org}/{repo}",
                'gguf_files': gguf_files
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_modelfile_response(org: str, repo: str, gguf_filename: str, file_url: str):
    """Generate modelfile from GGUF filename using same logic as hf-llm-install.py."""
    try:
        # Use shared parsing function from hf-llm-install.py
        model_base, tag, full_name = hf_install_module.parse_model_name_from_gguf(gguf_filename)
        
        # Extract quantization for metadata
        quant_match = re.search(r'[._-](Q[0-9]+_[KLM0-9]+(?:_[LSM])?)', gguf_filename, re.IGNORECASE)
        quantization = quant_match.group(1).upper() if quant_match else 'unspecified'
        
        # Create Modelfile skeleton with relative path (like CLI does)
        modelfile_content = f"""# Modelfile for {full_name}
# hf_upstream: {file_url}
# quantization: {quantization}
# capabilities: tools
# sha256: <add_sha256_checksum_here>

FROM ./{gguf_filename}

# System prompt - customize for your use case
SYSTEM \"\"\"You are a helpful AI assistant.\"\"\"

# Parameters - refer to manufacturer's recommendations
# https://huggingface.co/{org}/{repo}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|end|>"
PARAMETER stop "</s>"

# Template - adjust based on model's chat template
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
\"\"\"
"""
        
        return jsonify({
            'success': True,
            'requires_selection': False,
            'model_name': model_base,
            'tag': tag,
            'full_name': full_name,
            'gguf_filename': gguf_filename,
            'file_url': file_url,
            'repo_url': f"https://huggingface.co/{org}/{repo}",
            'modelfile_content': modelfile_content
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/install/huggingface/create', methods=['POST'])
def api_create_from_modelfile():
    """Start HuggingFace model creation as background job."""
    data = request.get_json()
    model_name = data.get('model_name', '').strip()
    modelfile_content = data.get('modelfile_content', '')
    file_url = data.get('file_url', '')
    gguf_filename = data.get('gguf_filename', '')
    
    if not model_name or not modelfile_content or not file_url:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        # Create job ID
        job_id = f"hf_install_{int(time.time() * 1000)}"
        
        # Initialize job state
        with install_lock:
            install_jobs[job_id] = {
                'status': 'queued',
                'progress': 'Queued for download',
                'model_name': model_name,
                'error': None,
                'cancelled': False
            }
        
        # Start background thread
        thread = threading.Thread(
            target=run_huggingface_install_job,
            args=(job_id, model_name, modelfile_content, file_url, gguf_filename)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Installation started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/install/modelfile', methods=['POST'])
def api_install_from_modelfile():
    """Start installation of a model from an existing Modelfile as background job."""
    try:
        data = request.get_json()
        modelfile_path = data.get('modelfile_path', '')
        
        if not modelfile_path:
            return jsonify({'error': 'No modelfile path provided'}), 400
        
        if not os.path.exists(modelfile_path):
            return jsonify({'error': 'Modelfile not found'}), 404
        
        # Create job ID
        job_id = f"install_{int(time.time() * 1000)}"
        
        # Initialize job state
        with install_lock:
            install_jobs[job_id] = {
                'status': 'queued',
                'progress': 'Queued for installation',
                'modelfile_path': modelfile_path,
                'model_name': None,
                'error': None,
                'cancelled': False
            }
        
        # Start background thread
        thread = threading.Thread(target=run_install_job, args=(job_id, modelfile_path))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Installation started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/install/status/<job_id>', methods=['GET'])
def api_install_status(job_id):
    """Get status of an installation job."""
    with install_lock:
        if job_id not in install_jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = install_jobs[job_id].copy()
    
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'model_name': job.get('model_name'),
        'error': job.get('error')
    })


@app.route('/api/install/active', methods=['GET'])
def api_install_active():
    """Get all active (running or queued) installation jobs."""
    with install_lock:
        active = {}
        for job_id, job in install_jobs.items():
            if job['status'] in ['queued', 'running']:
                active[job_id] = {
                    'status': job['status'],
                    'modelfile_path': job['modelfile_path'],
                    'model_name': job.get('model_name')
                }
        return jsonify(active)


@app.route('/api/install/cancel/<job_id>', methods=['POST'])
def api_install_cancel(job_id):
    """Cancel an installation job."""
    with install_lock:
        if job_id not in install_jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        if install_jobs[job_id]['status'] in ['completed', 'failed', 'cancelled']:
            return jsonify({'error': 'Job already finished'}), 400
        
        install_jobs[job_id]['cancelled'] = True
    
    return jsonify({'success': True})


@app.route('/api/performance/vram-test/<path:model_name>', methods=['POST'])
def api_vram_test(model_name):
    """Test VRAM usage for a specific model or all models."""
    try:
        import time
        
        # Check if testing all models
        if model_name == '_all_':
            # Get all installed models
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            models_to_test = []
            for line in result.stdout.strip().split('\n')[1:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 1:
                        models_to_test.append(parts[0])
            
            # Test each model
            results = []
            for model in models_to_test:
                result = test_single_model_vram(model)
                results.append(result)
                time.sleep(0.5)  # Brief pause between tests
            
            return jsonify({
                'success': True,
                'results': results
            })
        else:
            # Test single model
            result = test_single_model_vram(model_name)
            return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def test_single_model_vram(model_name: str) -> Dict:
    """Test VRAM usage for a single model. Uses vram-test.py logic."""
    # Use the existing test_model_vram function from vram-test.py
    return vram_test_module.test_model_vram(model_name)


@app.route('/api/performance/optimize/<path:model_name>', methods=['POST'])
def api_optimize_context(model_name):
    """Run context optimizer for a specific model. Uses context-optimizer.py logic."""
    try:
        # Get parameters from request
        data = request.get_json() or {}
        overhead_gb = float(data.get('overhead_gb', 1.0))
        max_turns = int(data.get('max_turns', 20))
        
        # Use the existing find_optimal_context function from context-optimizer.py
        result = context_optimizer_module.find_optimal_context(model_name, max_turns=max_turns, overhead_gb=overhead_gb)
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'Optimization failed or no results returned'
            })
        
        # Check if optimization encountered an error
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error'],
                'model': model_name,
                'max_context': result.get('max_context', 0),
                'current_context': result.get('current_ctx', 0)
            })
        
        if 'results' not in result or len(result['results']) == 0:
            return jsonify({
                'success': False,
                'error': 'No test results available. Model may have failed to load.'
            })
        
        # Extract data from results
        test_results = []
        optimal_context = 0
        
        for r in result.get('results', []):
            test_results.append({
                'context_size': r.get('num_ctx', 0),
                'vram_gb': round(r.get('vram_gb', 0), 2),
                'offload_pct': round(r.get('offload_pct', 0), 1),
                'fits': r.get('offload_pct', 100) == 0
            })
            
            # Track optimal (largest that fits)
            if r.get('offload_pct', 100) == 0:
                optimal_context = max(optimal_context, r.get('num_ctx', 0))
        
        # Get VRAM info
        vram_total, vram_available = context_optimizer_module.get_gpu_vram()
        
        return jsonify({
            'success': True,
            'model': model_name,
            'max_context': result.get('max_context', 0),
            'current_context': result.get('current_ctx', 0),
            'optimal_context': result.get('recommended_ctx', optimal_context),
            'available_vram_gb': round(vram_available, 2) if vram_available else 0,
            'results': test_results,
            'summary': result.get('summary', '')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("Starting Ollama-Utils Web Interface...")
    print("Access the interface at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
