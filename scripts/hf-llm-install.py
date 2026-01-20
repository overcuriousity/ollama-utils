#!/usr/bin/env python3
"""
HuggingFace LLM Installer for Ollama
Automatically downloads GGUF files from HuggingFace and creates Ollama models.

Features:
- SHA256 checksum verification
- Disk space checking
- Dry run mode
- Parallel processing
- Skip existing models
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
import urllib.request


def parse_model_name_from_gguf(gguf_filename):
    """
    Parse model name and tag from GGUF filename.
    
    Args:
        gguf_filename: Name of the GGUF file
        
    Returns:
        Tuple of (model_base, tag, full_name) or (filename, 'latest', filename) if parsing fails
    """
    filename_stem = Path(gguf_filename).stem.lower()
    
    # Split on hyphens
    parts = filename_stem.split('-')
    if len(parts) >= 3:
        # Find where the size variant starts (e.g., "0.5b", "3b", "8b", "14b")
        base_parts = []
        tag_parts = []
        found_variant = False
        
        for part in parts:
            # Check if this looks like a size variant (e.g., "3b", "8b", "0.5b")
            if not found_variant and re.match(r'^\d+(\.\d+)?b$', part):
                found_variant = True
                tag_parts.append(part)
            elif found_variant:
                # Include everything after the variant (including quantization)
                tag_parts.append(part)
            else:
                # Before the variant = base name
                base_parts.append(part)
        
        if base_parts and tag_parts:
            model_base = '-'.join(base_parts)
            model_tag = '-'.join(tag_parts)
            full_name = f"{model_base}:{model_tag}"
            return (model_base, model_tag, full_name)
    
    # Fallback to filename without extension
    return (filename_stem, 'latest', filename_stem)


def parse_modelfile(modelfile_path):
    """
    Parse a Modelfile to extract HuggingFace upstream URL and model info.
    
    Args:
        modelfile_path: Path to the .Modelfile
        
    Returns:
        dict with model metadata or None if invalid
    """
    with open(modelfile_path, 'r') as f:
        content = f.read()
    
    # Look for hf_upstream in the header comments
    hf_match = re.search(r'#\s*hf_upstream:\s*(https://huggingface\.co/[^\s]+)', content)
    if not hf_match:
        return None
    
    hf_url = hf_match.group(1)
    
    # Look for optional quantization specification (default: q4_k_m)
    quant_match = re.search(r'#\s*quantization:\s*([a-zA-Z0-9_]+)', content)
    quantization = quant_match.group(1).upper() if quant_match else 'Q4_K_M'
    
    # Look for optional SHA256 checksum
    sha256_match = re.search(r'#\s*sha256:\s*([a-fA-F0-9]{64})', content)
    sha256 = sha256_match.group(1) if sha256_match else None
    
    # Look for optional capabilities (comma-separated list)
    # Format: # capabilities: tools, vision
    capabilities_match = re.search(r'#\s*capabilities:\s*([^\n]+)', content)
    capabilities = None
    if capabilities_match:
        # Parse comma-separated capabilities and clean whitespace
        caps_str = capabilities_match.group(1).strip()
        capabilities = [cap.strip() for cap in caps_str.split(',') if cap.strip()]
    
    # Check if URL points to a specific GGUF file or just the repo
    if hf_url.endswith('.gguf') or '/blob/' in hf_url or '/resolve/' in hf_url:
        # Specific file provided - use as-is
        resolve_url = hf_url.replace('/blob/', '/resolve/')
        gguf_filename = os.path.basename(urlparse(resolve_url).path)
    else:
        # Repository root provided - construct filename from repo name and quantization
        # URL format: https://huggingface.co/{org}/{repo}
        url_parts = urlparse(hf_url).path.strip('/').split('/')
        if len(url_parts) >= 2:
            repo_name = url_parts[1]  # e.g., "Ministral-3-3B-Instruct-2512-GGUF"
            
            # Remove -GGUF suffix if present (case-insensitive)
            if repo_name.upper().endswith('-GGUF'):
                repo_name = repo_name[:-5]
            
            # Construct filename: RepoName-Quantization.gguf
            gguf_filename = f"{repo_name}-{quantization}.gguf"
            resolve_url = f"{hf_url.rstrip('/')}/resolve/main/{gguf_filename}"
        else:
            print(f"✗ Invalid HuggingFace URL format: {hf_url}")
            return None
    
    # Extract model name and tag from the GGUF filename
    # Format: Model-Version-Variant-Year-Quant.gguf -> model:version-variant-year-quant
    # Example: Ministral-3-3B-Instruct-2512-Q5_K_M.gguf -> ministral-3:3b-instruct-2512-q5_k_m
    model_base, model_tag, model_name = parse_model_name_from_gguf(gguf_filename)
    
    return {
        'hf_url': hf_url,
        'resolve_url': resolve_url,
        'gguf_filename': gguf_filename,
        'model_name': model_name,
        'modelfile_path': modelfile_path,
        'sha256': sha256,
        'capabilities': capabilities
    }


def get_file_size(url):
    """
    Get the size of a file from URL without downloading it.
    
    Args:
        url: File URL
        
    Returns:
        Size in bytes or None if unavailable
    """
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as response:
            size = response.headers.get('Content-Length')
            return int(size) if size else None
    except Exception:
        return None


def check_disk_space(required_bytes, path='.'):
    """
    Check if there's enough disk space available.
    
    Args:
        required_bytes: Required space in bytes
        path: Path to check space on (default: current directory)
        
    Returns:
        Tuple of (has_space, available_bytes, required_bytes)
    """
    # Get absolute path to check actual filesystem
    abs_path = os.path.abspath(path)
    stat = shutil.disk_usage(abs_path)
    # Add 10% safety margin
    required_with_margin = int(required_bytes * 1.1)
    return (stat.free >= required_with_margin, stat.free, required_with_margin)


def calculate_sha256(filepath, chunk_size=8192):
    """
    Calculate SHA256 checksum of a file.
    
    Args:
        filepath: Path to file
        chunk_size: Bytes to read at once
        
    Returns:
        SHA256 hex digest
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_checksum(filepath, expected_sha256):
    """
    Verify file checksum matches expected value.
    
    Args:
        filepath: Path to file
        expected_sha256: Expected SHA256 hash
        
    Returns:
        True if match, False otherwise
    """
    print(f"  Verifying checksum...")
    actual = calculate_sha256(filepath)
    
    if actual.lower() == expected_sha256.lower():
        print(f"  ✓ Checksum verified: {actual[:16]}...")
        return True
    else:
        print(f"  ✗ Checksum mismatch!")
        print(f"    Expected: {expected_sha256}")
        print(f"    Actual:   {actual}")
        return False


def get_existing_models():
    """
    Get list of existing Ollama models.
    
    Returns:
        Set of model names
    """
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output to get model names
        # Format: NAME                    ID              SIZE      MODIFIED
        models = set()
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if line.strip():
                # Get first column (name)
                name = line.split()[0]
                # Remove tag if present
                base_name = name.split(':')[0]
                models.add(base_name)
        
        return models
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def download_file(url, dest_path, filename, should_cancel=None, progress_callback=None):
    """
    Download a file from URL to destination with progress indication.
    
    Args:
        url: Source URL
        dest_path: Destination file path
        filename: Name for display purposes
        should_cancel: Optional callback function that returns True if download should be cancelled
        progress_callback: Optional callback function to report progress messages
    """
    def log(msg):
        """Helper to print and optionally call progress callback."""
        print(msg)
        if progress_callback:
            progress_callback(msg)
    
    log(f"Downloading {filename}...")
    log(f"  From: {url}")
    log(f"  To: {dest_path}")
    
    def show_progress(block_num, block_size, total_size):
        # Check for cancellation
        if should_cancel and should_cancel():
            raise InterruptedError("Download cancelled")
        
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            msg = f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
            print(msg, end='')
            if progress_callback:
                progress_callback(f"Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
    
    try:
        urllib.request.urlretrieve(url, dest_path, show_progress)
        print()  # New line after progress
        log(f"✓ Download complete")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        if progress_callback:
            progress_callback(f"✗ Download failed: {e}")
        raise


def create_ollama_model(modelfile_path, gguf_path, model_name, capabilities=None):
    """
    Create an Ollama model from the Modelfile and GGUF file.
    
    Args:
        modelfile_path: Path to the .Modelfile
        gguf_path: Path to the downloaded GGUF file
        model_name: Name for the Ollama model
        capabilities: Optional list of capabilities to add (e.g., ['tools', 'vision'])
    """
    print(f"\nCreating Ollama model: {model_name}")
    
    # Note: Capabilities are detected from the GGUF file metadata by Ollama automatically
    if capabilities:
        print(f"  ℹ Expected capabilities from GGUF metadata: {', '.join(capabilities)}")
    
    # Read the Modelfile and update the FROM path to point to the downloaded GGUF
    with open(modelfile_path, 'r') as f:
        modelfile_content = f.read()
    
    # Replace the FROM line to use the actual GGUF path
    # Handle both relative paths like "./filename.gguf" and URLs like "https://..."
    original_content = modelfile_content
    modelfile_content = re.sub(
        r'FROM\s+(?:\./[^\s]+\.gguf|https?://[^\n]+)',
        f'FROM {gguf_path}',
        modelfile_content
    )
    
    # Debug: check if replacement happened
    if original_content == modelfile_content:
        print(f"  WARNING: FROM line was not replaced!")
        print(f"  Looking for pattern in: {original_content[:200]}")
    else:
        print(f"  ✓ Replaced FROM line with local path: {gguf_path}")
    
    # Create a temporary Modelfile with the correct path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.Modelfile', delete=False) as tmp_modelfile:
        tmp_modelfile.write(modelfile_content)
        tmp_modelfile_path = tmp_modelfile.name
    
    try:
        # Run ollama create
        cmd = ['ollama', 'create', model_name, '-f', tmp_modelfile_path]
        print(f"  Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Success - output will be shown by the caller
            if result.stdout:
                print(result.stdout.strip())
        else:
            print(f"✗ Failed to create model")
            if result.stderr:
                print(f"  Error: {result.stderr.strip()}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
            
    finally:
        # Clean up temporary Modelfile
        os.unlink(tmp_modelfile_path)


def install_model(modelfile_path, dry_run=False, skip_existing=False, existing_models=None, should_cancel=None, progress_callback=None):
    """
    Install a single model from a Modelfile.
    
    Args:
        modelfile_path: Path to the .Modelfile
        dry_run: If True, only simulate installation
        skip_existing: If True, skip models already in Ollama
        existing_models: Set of existing model names
        should_cancel: Optional callback function that returns True if installation should be cancelled
        progress_callback: Optional callback function to report progress messages
        
    Returns:
        Tuple of (success: bool, skipped: bool, model_name: str)
    """
    def log(msg):
        """Helper to print and optionally call progress callback."""
        print(msg)
        if progress_callback:
            progress_callback(msg)
    log(f"\n{'='*80}")
    log(f"Processing: {modelfile_path}")
    log(f"{'='*80}")
    
    # Parse the Modelfile
    model_info = parse_modelfile(modelfile_path)
    if not model_info:
        log(f"✗ No hf_upstream found in {modelfile_path}")
        return (False, False, None)
    
    log(f"Model name: {model_info['model_name']}")
    log(f"GGUF file: {model_info['gguf_filename']}")
    if model_info['sha256']:
        log(f"SHA256: {model_info['sha256'][:16]}...")
    if model_info.get('capabilities'):
        log(f"Capabilities: {', '.join(model_info['capabilities'])}")
    
    # Check if model already exists
    if skip_existing and existing_models and model_info['model_name'] in existing_models:
        log(f"⊘ Model '{model_info['model_name']}' already exists, skipping")
        return (True, True, model_info['model_name'])
    
    # Get file size and check disk space
    file_size = get_file_size(model_info['resolve_url'])
    if file_size:
        size_gb = file_size / (1024**3)
        log(f"File size: {size_gb:.2f} GB")
        
        if not dry_run:
            has_space, available, required = check_disk_space(file_size)
            if not has_space:
                log(f"✗ Insufficient disk space!")
                log(f"  Required: {required / (1024**3):.2f} GB (with 10% margin)")
                log(f"  Available: {available / (1024**3):.2f} GB")
                return (False, False, model_info['model_name'])
            else:
                log(f"✓ Disk space check passed ({available / (1024**3):.2f} GB available)")
    
    if dry_run:
        log(f"\n[DRY RUN] Would download and install model: {model_info['model_name']}")
        return (True, False, model_info['model_name'])
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as tmp_dir:
        gguf_path = os.path.join(tmp_dir, model_info['gguf_filename'])
        
        try:
            # Download the GGUF file
            download_file(model_info['resolve_url'], gguf_path, model_info['gguf_filename'], should_cancel, progress_callback)
            
            # Verify checksum if provided
            if model_info['sha256']:
                if not verify_checksum(gguf_path, model_info['sha256']):
                    print(f"✗ Checksum verification failed!")
                    return (False, False, model_info['model_name'])
            
            # Create the Ollama model
            create_ollama_model(
                modelfile_path,
                gguf_path,
                model_info['model_name'],
                model_info.get('capabilities')
            )
            
            print(f"\n✓ Successfully installed model: {model_info['model_name']}")
            return (True, False, model_info['model_name'])
            
        except Exception as e:
            print(f"\n✗ Failed to install model: {e}")
            return (False, False, model_info['model_name'])


def install_model_wrapper(args):
    """Wrapper for parallel execution."""
    return install_model(*args)


def main():
    parser = argparse.ArgumentParser(
        description='Install Ollama models from HuggingFace using Modelfiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install a single model
  %(prog)s path/to/model.Modelfile
  
  # Install all models in the default repo directory
  %(prog)s
  
  # Dry run to see what would be installed
  %(prog)s --dry-run
  
  # Skip models that already exist
  %(prog)s --skip-existing
  
  # Install with 3 parallel downloads
  %(prog)s --parallel 3
        """
    )
    
    parser.add_argument(
        'modelfile',
        nargs='?',
        help='Path to a specific .Modelfile to install (optional)'
    )
    
    parser.add_argument(
        '--dir',
        default='./modelfile-repo',
        help='Directory containing .Modelfile files (default: ./modelfile-repo)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate installation without downloading or creating models'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip models that already exist in Ollama'
    )
    
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        metavar='N',
        help='Number of parallel downloads/installations (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validate parallel argument
    if args.parallel < 1:
        print("✗ Error: --parallel must be at least 1")
        sys.exit(1)
    
    # Check if ollama is available
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Error: 'ollama' command not found. Please install Ollama first.")
        print("  Visit: https://ollama.ai")
        sys.exit(1)
    
    # Get existing models if skip_existing is enabled
    existing_models = None
    if args.skip_existing:
        existing_models = get_existing_models()
        if existing_models:
            print(f"Found {len(existing_models)} existing model(s)")
    
    # Determine which Modelfiles to process
    if args.modelfile:
        # Single file mode
        modelfile_path = Path(args.modelfile)
        if not modelfile_path.exists():
            print(f"✗ Error: File not found: {modelfile_path}")
            sys.exit(1)
        
        if not modelfile_path.suffix == '.Modelfile':
            print(f"✗ Error: File must have .Modelfile extension")
            sys.exit(1)
        
        modelfiles = [modelfile_path]
    else:
        # Batch mode - process all .Modelfile files in directory
        modelfile_dir = Path(args.dir)
        if not modelfile_dir.exists():
            print(f"✗ Error: Directory not found: {modelfile_dir}")
            sys.exit(1)
        
        modelfiles = sorted(modelfile_dir.glob('*.Modelfile'))
        if not modelfiles:
            print(f"✗ No .Modelfile files found in {modelfile_dir}")
            sys.exit(1)
        
        print(f"Found {len(modelfiles)} Modelfile(s) to process")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be downloaded or models created ***\n")
    
    # Process all Modelfiles
    results = []
    
    if args.parallel > 1 and len(modelfiles) > 1:
        # Parallel processing
        print(f"\nUsing {args.parallel} parallel worker(s)")
        
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            future_to_modelfile = {
                executor.submit(
                    install_model_wrapper,
                    (modelfile, args.dry_run, args.skip_existing, existing_models)
                ): modelfile
                for modelfile in modelfiles
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_modelfile):
                modelfile = future_to_modelfile[future]
                try:
                    success, skipped, model_name = future.result()
                    results.append((modelfile.name, success, skipped))
                except Exception as e:
                    print(f"\n✗ Exception processing {modelfile.name}: {e}")
                    results.append((modelfile.name, False, False))
    else:
        # Sequential processing
        for modelfile in modelfiles:
            success, skipped, model_name = install_model(
                modelfile,
                args.dry_run,
                args.skip_existing,
                existing_models
            )
            results.append((modelfile.name, success, skipped))
    
    # Summary
    print(f"\n{'='*80}")
    print("INSTALLATION SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for _, success, skipped in results if success and not skipped)
    skipped = sum(1 for _, success, skip in results if skip)
    failed = len(results) - successful - skipped
    
    for name, success, skip in results:
        if skip:
            status = "⊘"
        elif success:
            status = "✓"
        else:
            status = "✗"
        print(f"{status} {name}")
    
    print(f"\nTotal: {len(results)} | Successful: {successful} | Skipped: {skipped} | Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
