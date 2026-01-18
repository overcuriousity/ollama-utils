// Ollama Utils Web Interface - JavaScript

// Global state
let currentModels = [];
let statusUpdateInterval = null;

// ===== INITIALIZATION =====

document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeModals();
    initializeEventListeners();
    
    // Start real-time updates
    updateStatus();
    statusUpdateInterval = setInterval(updateStatus, 2000); // Update every 2 seconds
    
    // Load initial data
    loadModels();
});

// ===== TAB MANAGEMENT =====

function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tabName}`);
    });
    
    // Load tab-specific data
    if (tabName === 'performance') {
        populatePerformanceSelects();
    }
}

// ===== MODAL MANAGEMENT =====

function initializeModals() {
    // Install modal
    document.getElementById('modal-close').addEventListener('click', () => {
        closeModal('install-modal');
    });
    
    // Model details modal
    document.getElementById('model-details-close').addEventListener('click', () => {
        closeModal('model-details-modal');
    });
    
    // Modelfile editor modal
    document.getElementById('modelfile-editor-close').addEventListener('click', () => {
        closeModal('modelfile-editor-modal');
    });
    
    // Close modals on background click
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal(modal.id);
            }
        });
    });
}

function openModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

// ===== EVENT LISTENERS =====

function initializeEventListeners() {
    // Model management
    document.getElementById('btn-install').addEventListener('click', openInstallModal);
    document.getElementById('btn-refresh-models').addEventListener('click', loadModels);
    
    // Install options
    document.querySelectorAll('.install-option').forEach(option => {
        option.addEventListener('click', () => {
            switchInstallOption(option.dataset.option);
        });
    });
    
    // Ollama install
    document.getElementById('btn-install-ollama').addEventListener('click', installOllamaModel);
    
    // HuggingFace install
    document.getElementById('btn-fetch-hf').addEventListener('click', fetchHuggingFaceInfo);
    document.getElementById('btn-create-hf-model').addEventListener('click', createHuggingFaceModel);
    
    // Modelfile editor
    document.getElementById('btn-save-modelfile').addEventListener('click', saveModelfile);
    document.getElementById('btn-cancel-modelfile').addEventListener('click', () => {
        closeModal('modelfile-editor-modal');
    });
    
    // Performance tools
    document.getElementById('btn-run-vram-test').addEventListener('click', runVramTest);
    document.getElementById('btn-run-optimizer').addEventListener('click', runOptimizer);
    document.getElementById('btn-stop-optimizer').addEventListener('click', stopOptimizer);
}

function switchInstallOption(option) {
    // Update buttons
    document.querySelectorAll('.install-option').forEach(opt => {
        opt.classList.toggle('active', opt.dataset.option === option);
    });
    
    // Update forms
    document.querySelectorAll('.install-form').forEach(form => {
        form.classList.toggle('active', form.id === `install-${option}`);
    });
}

// ===== STATUS UPDATES =====

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        // Update CPU
        document.getElementById('cpu-load').textContent = data.cpu_load.toFixed(2);
        
        // Update RAM
        const ramPct = data.ram_used_pct;
        document.getElementById('ram-usage').textContent = 
            `${data.ram_used_mb} / ${data.ram_total_mb} MB (${ramPct}%)`;
        document.getElementById('ram-progress').style.width = `${ramPct}%`;
        
        // Update VRAM
        if (data.vram_used_mb !== null && data.vram_total_mb !== null) {
            const vramPct = data.vram_used_pct;
            document.getElementById('vram-usage').textContent = 
                `${data.vram_used_mb} / ${data.vram_total_mb} MB (${vramPct}%)`;
            document.getElementById('vram-progress').style.width = `${vramPct}%`;
        } else {
            document.getElementById('vram-usage').textContent = 'N/A';
            document.getElementById('vram-progress').style.width = '0%';
        }
        
        // Update GPU load
        if (data.gpu_load !== null) {
            document.getElementById('gpu-load').textContent = `${data.gpu_load}%`;
        } else {
            document.getElementById('gpu-load').textContent = 'N/A';
        }
        
        // Update running models
        updateRunningModels(data.running_models);
        
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

function updateRunningModels(models) {
    const container = document.getElementById('running-models-list');
    
    if (!models || models.length === 0) {
        container.innerHTML = '<div class="no-models">No models loaded</div>';
        return;
    }
    
    container.innerHTML = models.map(model => `
        <div class="running-model">
            <div class="running-model-name">${escapeHtml(model.name)}</div>
            <div class="running-model-stats">
                VRAM: ${model.vram_gb.toFixed(2)} GB
                ${model.offload_pct > 0 ? ` | CPU: ${model.offload_pct.toFixed(1)}%` : ''}
            </div>
        </div>
    `).join('');
}

// ===== MODEL MANAGEMENT =====

async function loadModels() {
    const container = document.getElementById('models-list');
    
    // Save active installation job IDs before refreshing
    const activeInstalls = new Map();
    document.querySelectorAll('.model-card[data-model-name]').forEach(card => {
        const modelName = card.dataset.modelName;
        const progressDiv = card.querySelector('.install-progress');
        if (progressDiv && progressDiv.style.display !== 'none') {
            const jobIdMatch = progressDiv.dataset.jobId;
            if (jobIdMatch) {
                activeInstalls.set(modelName, jobIdMatch);
            }
        }
    });
    
    container.innerHTML = '<div class="loading">Loading models...</div>';
    
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (data.error) {
            container.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.error)}</div>`;
            return;
        }
        
        currentModels = data.models;
        renderModels(data.models);
        
        // Check for active jobs from server (for page reloads)
        const activeJobsResponse = await fetch('/api/install/active');
        const activeJobs = await activeJobsResponse.json();
        
        // Combine saved jobs with server jobs
        for (const [jobId, jobInfo] of Object.entries(activeJobs)) {
            // Find model name from modelfile path
            const model = data.models.find(m => m.modelfile_path === jobInfo.modelfile_path);
            if (model && !activeInstalls.has(model.name)) {
                activeInstalls.set(model.name, jobId);
            }
        }
        
        // Restart polling for active installations
        activeInstalls.forEach((jobId, modelName) => {
            showInstallProgressInCard(jobId, modelName);
        });
        
    } catch (error) {
        container.innerHTML = `<div class="error-message">Error loading models: ${escapeHtml(error.message)}</div>`;
    }
}

function renderModels(models) {
    const container = document.getElementById('models-list');
    
    if (models.length === 0) {
        container.innerHTML = '<div class="loading">No models installed</div>';
        return;
    }
    
    container.innerHTML = models.map(model => `
        <div class="model-card ${!model.installed ? 'model-not-installed' : ''}" data-model-name="${escapeHtml(model.name)}">
            <div class="model-header">
                <div class="model-name">
                    ${escapeHtml(model.name)}
                    ${!model.installed ? '<span class="badge badge-warning">Not Installed</span>' : ''}
                </div>
                <div class="model-actions">
                    ${!model.installed && model.has_modelfile ? `
                        <button class="btn btn-small btn-primary install-btn" onclick="installFromModelfile('${escapeHtml(model.modelfile_path)}', '${escapeHtml(model.name)}')">
                            Install
                        </button>
                        <button class="btn btn-small btn-danger cancel-install-btn" style="display: none;">Cancel</button>
                        <button class="btn btn-small btn-secondary" onclick="editModelfile('${escapeHtml(model.name)}')">
                            Edit Modelfile
                        </button>
                    ` : ''}
                    ${model.installed && model.has_modelfile ? `
                        <button class="btn btn-small btn-secondary" onclick="editModelfile('${escapeHtml(model.name)}')">
                            Edit Modelfile
                        </button>
                    ` : ''}
                    ${model.installed ? `
                        <button class="btn btn-small btn-danger" onclick="deleteModel('${escapeHtml(model.name)}')">
                            Delete
                        </button>
                    ` : ''}
                </div>
            </div>
            
            <!-- Installation progress container -->
            <div class="install-progress" style="display: none;">
                <div class="install-progress-header">
                    <span class="install-status-icon">⏳</span>
                    <span class="install-status-text">Installing...</span>
                </div>
                <div class="install-progress-bar-container">
                    <div class="install-progress-bar">
                        <div class="install-progress-bar-fill" style="width: 0%;"></div>
                    </div>
                    <span class="install-progress-percent">0%</span>
                </div>
            </div>
            
            <div class="model-info">
                <div class="model-info-item">
                    <div class="model-info-label">Size</div>
                    <div class="model-info-value">${escapeHtml(model.size)}</div>
                </div>
                <div class="model-info-item">
                    <div class="model-info-label">Parameters</div>
                    <div class="model-info-value">${escapeHtml(model.params)}</div>
                </div>
                <div class="model-info-item">
                    <div class="model-info-label">Quantization</div>
                    <div class="model-info-value">${escapeHtml(model.quant)}</div>
                </div>
                <div class="model-info-item">
                    <div class="model-info-label">Architecture</div>
                    <div class="model-info-value">${escapeHtml(model.family)}</div>
                </div>
                ${model.installed ? `
                    <div class="model-info-item">
                        <div class="model-info-label">Context</div>
                        <div class="model-info-value">${model.context_used > 0 ? `${model.context_used}/${model.max_context}` : model.max_context}</div>
                    </div>
                    <div class="model-info-item">
                        <div class="model-info-label">Est. VRAM</div>
                        <div class="model-info-value">${escapeHtml(model.vram_estimate)}</div>
                    </div>
                ` : ''}
                ${!model.installed && model.hf_upstream ? `
                    <div class="model-info-item" style="grid-column: 1 / -1;">
                        <div class="model-info-label">Source</div>
                        <div class="model-info-value"><a href="${escapeHtml(model.hf_upstream)}" target="_blank" style="color: var(--accent-primary);">${escapeHtml(model.hf_upstream)}</a></div>
                    </div>
                ` : ''}
            </div>
            
            <div class="model-badges">
                ${model.has_modelfile ? '<span class="badge badge-modelfile">Has Modelfile</span>' : '<span class="badge badge-ollama">Ollama Library</span>'}
                ${model.capabilities.map(cap => `<span class="badge badge-capability">${escapeHtml(cap)}</span>`).join('')}
            </div>
        </div>
    `).join('');
}

async function deleteModel(modelName) {
    if (!confirm(`Are you sure you want to delete "${modelName}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/model/${encodeURIComponent(modelName)}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            loadModels(); // Reload the list
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        alert(`Error deleting model: ${error.message}`);
    }
}

async function installFromModelfile(modelfilePath, modelName) {
    try {
        // Start installation
        const response = await fetch('/api/install/modelfile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                modelfile_path: modelfilePath
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.job_id) {
            // Show progress in the model card
            showInstallProgressInCard(data.job_id, modelName);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        alert(`Error starting installation: ${error.message}`);
    }
}

async function showInstallProgressInCard(jobId, modelName) {
    // Find the model card
    const modelCard = document.querySelector(`.model-card[data-model-name="${modelName}"]`);
    if (!modelCard) return;
    
    const progressContainer = modelCard.querySelector('.install-progress');
    const progressBarFill = modelCard.querySelector('.install-progress-bar-fill');
    const progressPercent = modelCard.querySelector('.install-progress-percent');
    const statusIcon = modelCard.querySelector('.install-status-icon');
    const statusText = modelCard.querySelector('.install-status-text');
    const cancelBtn = modelCard.querySelector('.cancel-install-btn');
    const installBtn = modelCard.querySelector('.install-btn');
    
    // Store job ID in both cancel button and progress container
    cancelBtn.dataset.jobId = jobId;
    progressContainer.dataset.jobId = jobId;
    
    // If already polling this job, don't start a new interval
    if (progressContainer.dataset.polling === 'true') {
        return;
    }
    progressContainer.dataset.polling = 'true';
    
    // Hide install button, show cancel button and progress
    if (installBtn) installBtn.style.display = 'none';
    cancelBtn.style.display = 'inline-block';
    progressContainer.style.display = 'block';
    
    // Set up cancel button handler
    cancelBtn.onclick = async function() {
        const btnJobId = this.dataset.jobId;
        if (confirm(`Cancel installation?`)) {
            try {
                await fetch(`/api/install/cancel/${btnJobId}`, { method: 'POST' });
                this.disabled = true;
            } catch (error) {
                alert(`Error cancelling: ${error.message}`);
            }
        }
    };
    
    // Poll for progress
    const pollInterval = setInterval(async () => {
        try {
            const statusResponse = await fetch(`/api/install/status/${jobId}`);
            const statusData = await statusResponse.json();
            
            if (statusData.progress) {
                
                // Extract percentage from progress text (e.g., "Progress: 45.2% (1024/2048 MB)")
                const percentMatch = statusData.progress.match(/([0-9.]+)%/);
                if (percentMatch) {
                    const percent = parseFloat(percentMatch[1]);
                    progressBarFill.style.width = `${percent}%`;
                    progressPercent.textContent = `${percent.toFixed(1)}%`;
                }
            }
            
            if (statusData.status === 'completed') {
                clearInterval(pollInterval);
                progressContainer.dataset.polling = 'false';
                statusIcon.textContent = '✓';
                statusText.textContent = 'Installation completed!';
                progressBarFill.style.width = '100%';
                progressPercent.textContent = '100%';
                cancelBtn.style.display = 'none';
                
                // Reload models after short delay
                setTimeout(() => loadModels(), 2000);
            } else if (statusData.status === 'failed') {
                clearInterval(pollInterval);
                progressContainer.dataset.polling = 'false';
                statusIcon.textContent = '✗';
                statusText.textContent = 'Installation failed';
                cancelBtn.style.display = 'none';
                
                // Show install button again after delay
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    if (installBtn) installBtn.style.display = 'inline-block';
                }, 5000);
            } else if (statusData.status === 'cancelled') {
                clearInterval(pollInterval);
                progressContainer.dataset.polling = 'false';
                statusIcon.textContent = '⊘';
                statusText.textContent = 'Installation cancelled';
                cancelBtn.style.display = 'none';
                
                // Show install button again after delay
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    if (installBtn) installBtn.style.display = 'inline-block';
                }, 3000);
            }
        } catch (error) {
            clearInterval(pollInterval);
            statusIcon.textContent = '✗';
            statusText.textContent = 'Error';
            cancelBtn.style.display = 'none';
        }
    }, 1000); // Poll every second
}

async function editModelfile(modelName) {
    try {
        const response = await fetch(`/api/modelfile/${encodeURIComponent(modelName)}`);
        const data = await response.json();
        
        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }
        
        // Open editor modal
        document.getElementById('modelfile-editor-title').textContent = `Edit Modelfile - ${modelName}`;
        const editorContent = document.getElementById('modelfile-editor-content');
        editorContent.value = data.content;
        editorContent.dataset.modelName = modelName;
        // Store original content to detect changes
        editorContent.dataset.originalContent = data.content;
        openModal('modelfile-editor-modal');
        
    } catch (error) {
        alert(`Error loading Modelfile: ${error.message}`);
    }
}

async function saveModelfile() {
    const editorContent = document.getElementById('modelfile-editor-content');
    const content = editorContent.value;
    const modelName = editorContent.dataset.modelName;
    const originalContent = editorContent.dataset.originalContent;
    const outputBox = document.getElementById('modelfile-save-output');
    
    // Check if content has changed
    const hasChanges = content !== originalContent;
    
    try {
        const response = await fetch(`/api/modelfile/${encodeURIComponent(modelName)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                content,
                recreate_model: hasChanges
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.recreating && data.job_id) {
                outputBox.innerHTML = '<div class="success-message">Modelfile saved! Recreating model...</div>';
                
                // Close modal and show progress in the model card
                setTimeout(async () => {
                    closeModal('modelfile-editor-modal');
                    outputBox.innerHTML = '';
                    
                    // Wait for models to load before showing progress
                    await loadModels();
                    
                    // Start showing progress in the model card
                    showInstallProgressInCard(data.job_id, modelName);
                }, 1000);
            } else {
                outputBox.innerHTML = '<div class="success-message">Modelfile saved successfully!</div>';
                setTimeout(() => {
                    closeModal('modelfile-editor-modal');
                    outputBox.innerHTML = '';
                }, 2000);
            }
        } else {
            outputBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.error)}</div>`;
        }
    } catch (error) {
        outputBox.innerHTML = `<div class="error-message">Error saving Modelfile: ${escapeHtml(error.message)}</div>`;
    }
}

// ===== INSTALL MODELS =====

function openInstallModal() {
    // Reset the form
    document.getElementById('ollama-model-name').value = '';
    document.getElementById('hf-url').value = '';
    document.getElementById('ollama-install-output').innerHTML = '';
    document.getElementById('hf-install-output').innerHTML = '';
    document.getElementById('hf-modelfile-section').style.display = 'none';
    
    openModal('install-modal');
}

async function installOllamaModel() {
    const modelName = document.getElementById('ollama-model-name').value.trim();
    const outputBox = document.getElementById('ollama-install-output');
    
    if (!modelName) {
        outputBox.innerHTML = '<div class="error-message">Please enter a model name</div>';
        return;
    }
    
    outputBox.innerHTML = '<div class="loading">Pulling model... This may take a while.</div>';
    
    try {
        const response = await fetch('/api/install/ollama', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_name: modelName })
        });
        
        const data = await response.json();
        
        if (data.success) {
            outputBox.innerHTML = '<div class="success-message">Model installed successfully!</div>';
            loadModels(); // Refresh the models list
        } else {
            outputBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.error)}</div>`;
        }
    } catch (error) {
        outputBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(error.message)}</div>`;
    }
}

async function fetchHuggingFaceInfo() {
    const url = document.getElementById('hf-url').value.trim();
    const outputBox = document.getElementById('hf-install-output');
    const fileSelectSection = document.getElementById('hf-file-select-section');
    
    if (!url) {
        outputBox.innerHTML = '<div class="error-message">Please enter a HuggingFace URL</div>';
        return;
    }
    
    outputBox.innerHTML = '<div class="loading">Fetching model information...</div>';
    
    try {
        const response = await fetch('/api/install/huggingface', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        });
        
        const data = await response.json();
        
        if (data.success && data.requires_selection) {
            // Show dropdown for file selection
            let selectHtml = `
                <div class="success-message">
                    Found ${data.gguf_files.length} GGUF file(s) in repository. Please select one:
                </div>
                <div style="margin: 15px 0;">
                    <label for="hf-file-select" style="display: block; margin-bottom: 5px;">Select GGUF File:</label>
                    <select id="hf-file-select" class="input" style="width: 100%; padding: 8px;">
                        <option value="">-- Select a file --</option>
            `;
            
            for (const file of data.gguf_files) {
                selectHtml += `<option value="${escapeHtml(file.filename)}">${escapeHtml(file.filename)}</option>`;
            }
            
            selectHtml += `
                    </select>
                </div>
                <button class="btn btn-primary" onclick="generateModelfileFromSelection()">Generate Modelfile</button>
            `;
            
            fileSelectSection.innerHTML = selectHtml;
            fileSelectSection.style.display = 'block';
            outputBox.innerHTML = '';
            
            // Store org/repo for later use
            fileSelectSection.dataset.org = data.org;
            fileSelectSection.dataset.repo = data.repo;
            fileSelectSection.dataset.baseUrl = url;
            
        } else if (data.success && !data.requires_selection) {
            // Direct file - populate the Modelfile editor immediately
            fileSelectSection.style.display = 'none';
            document.getElementById('hf-model-name').value = data.full_name;
            document.getElementById('hf-modelfile').value = data.modelfile_content;
            document.getElementById('hf-repo-link').href = data.repo_url;
            
            // Store file_url and gguf_filename for later
            const modelfileSection = document.getElementById('hf-modelfile-section');
            modelfileSection.dataset.fileUrl = data.file_url;
            modelfileSection.dataset.ggufFilename = data.gguf_filename;
            modelfileSection.style.display = 'block';
            
            outputBox.innerHTML = '<div class="success-message">Model information fetched! Please review and customize the Modelfile below.</div>';
        } else {
            fileSelectSection.style.display = 'none';
            outputBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.error)}</div>`;
        }
    } catch (error) {
        fileSelectSection.style.display = 'none';
        outputBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(error.message)}</div>`;
    }
}

async function generateModelfileFromSelection() {
    const fileSelectSection = document.getElementById('hf-file-select-section');
    const selectedFile = document.getElementById('hf-file-select').value;
    const outputBox = document.getElementById('hf-install-output');
    
    if (!selectedFile) {
        outputBox.innerHTML = '<div class="error-message">Please select a file</div>';
        return;
    }
    
    const baseUrl = fileSelectSection.dataset.baseUrl;
    
    outputBox.innerHTML = '<div class="loading">Generating Modelfile...</div>';
    
    try {
        const response = await fetch('/api/install/huggingface', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                url: baseUrl,
                selected_file: selectedFile
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Populate the Modelfile editor
            document.getElementById('hf-model-name').value = data.full_name;
            document.getElementById('hf-modelfile').value = data.modelfile_content;
            document.getElementById('hf-repo-link').href = data.repo_url;
            
            // Store file_url and gguf_filename for later
            const modelfileSection = document.getElementById('hf-modelfile-section');
            modelfileSection.dataset.fileUrl = data.file_url;
            modelfileSection.dataset.ggufFilename = data.gguf_filename;
            modelfileSection.style.display = 'block';
            
            fileSelectSection.style.display = 'none';
            outputBox.innerHTML = '<div class="success-message">Modelfile generated! Please review and customize below.</div>';
        } else {
            outputBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.error)}</div>`;
        }
    } catch (error) {
        outputBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(error.message)}</div>`;
    }
}

async function createHuggingFaceModel() {
    const modelName = document.getElementById('hf-model-name').value.trim();
    const modelfileContent = document.getElementById('hf-modelfile').value.trim();
    const outputBox = document.getElementById('hf-install-output');
    const modelfileSection = document.getElementById('hf-modelfile-section');
    
    const fileUrl = modelfileSection.dataset.fileUrl;
    const ggufFilename = modelfileSection.dataset.ggufFilename;
    
    if (!modelName || !modelfileContent || !fileUrl) {
        outputBox.innerHTML = '<div class="error-message">Missing required information. Please fetch a model first.</div>';
        return;
    }
    
    try {
        const response = await fetch('/api/install/huggingface/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: modelName,
                modelfile_content: modelfileContent,
                file_url: fileUrl,
                gguf_filename: ggufFilename
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.job_id) {
            // Show progress bar with same structure as model card progress
            outputBox.innerHTML = `
                <div class="install-progress">
                    <div class="install-progress-bar-container">
                        <div class="install-progress-bar">
                            <div class="install-progress-bar-fill" style="width: 0%"></div>
                        </div>
                        <span class="install-progress-percent">0%</span>
                    </div>
                    <div class="install-status-text">Starting...</div>
                </div>
            `;
            
            // Poll for status
            pollHuggingFaceJobStatus(data.job_id, outputBox);
        } else {
            outputBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.error)}</div>`;
        }
    } catch (error) {
        outputBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(error.message)}</div>`;
    }
}

async function pollHuggingFaceJobStatus(jobId, outputBox) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/install/status/${jobId}`);
            const data = await response.json();
            
            if (data.status === 'running') {
                const statusText = outputBox.querySelector('.install-status-text');
                if (statusText) {
                    statusText.textContent = data.progress || 'Processing...';
                }
                
                // Extract percentage if present in progress text
                const percentMatch = data.progress?.match(/([0-9.]+)%/);
                if (percentMatch) {
                    const percent = parseFloat(percentMatch[1]);
                    const progressBarFill = outputBox.querySelector('.install-progress-bar-fill');
                    const progressPercent = outputBox.querySelector('.install-progress-percent');
                    if (progressBarFill) {
                        progressBarFill.style.width = `${percent}%`;
                    }
                    if (progressPercent) {
                        progressPercent.textContent = `${percent.toFixed(1)}%`;
                    }
                }
            } else if (data.status === 'completed') {
                clearInterval(pollInterval);
                outputBox.innerHTML = `<div class="success-message">Successfully created ${escapeHtml(data.model_name)}!</div>`;
                loadModels(); // Refresh models list
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                outputBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.error || 'Installation failed')}</div>`;
            } else if (data.status === 'cancelled') {
                clearInterval(pollInterval);
                outputBox.innerHTML = `<div class="error-message">Installation cancelled</div>`;
            }
        } catch (error) {
            clearInterval(pollInterval);
            outputBox.innerHTML = `<div class="error-message">Error polling status: ${escapeHtml(error.message)}</div>`;
        }
    }, 1000);
}

// ===== PERFORMANCE TOOLS =====

function populatePerformanceSelects() {
    const vramSelect = document.getElementById('vram-test-model');
    const optimizerSelect = document.getElementById('optimizer-model');
    
    // Filter only installed models
    const installedModels = currentModels.filter(m => m.installed);
    
    const options = installedModels.map(model => 
        `<option value="${escapeHtml(model.name)}">${escapeHtml(model.name)}</option>`
    ).join('');
    
    vramSelect.innerHTML = '<option value="">-- Select a model --</option><option value="_all_">Test All Models</option>' + options;
    optimizerSelect.innerHTML = '<option value="">-- Select a model --</option>' + options;
}

async function runVramTest() {
    const modelName = document.getElementById('vram-test-model').value;
    const resultsBox = document.getElementById('vram-test-results');
    
    if (!modelName) {
        resultsBox.innerHTML = '<div class="error-message">Please select a model</div>';
        return;
    }
    
    if (modelName === '_all_') {
        resultsBox.innerHTML = '<div class="loading">Testing all models... This may take a while.</div>';
    } else {
        resultsBox.innerHTML = '<div class="loading">Testing VRAM usage... Please wait.</div>';
    }
    
    try {
        const response = await fetch(`/api/performance/vram-test/${encodeURIComponent(modelName)}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.results) {
                // Batch results - display as table
                let tableHtml = `
                    <div class="success-message">
                        <strong>VRAM Test Results - All Models</strong>
                    </div>
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Params</th>
                                <th>Quant</th>
                                <th>num_ctx</th>
                                <th>Total Size</th>
                                <th>VRAM Usage</th>
                                <th>CPU Offload</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                for (const result of data.results) {
                    const status = result.success 
                        ? (result.offload_pct > 0 ? '⚠️ Offloading' : '✓ GPU Only')
                        : '✗ Failed';
                    
                    tableHtml += `
                        <tr>
                            <td>${escapeHtml(result.model)}</td>
                            <td>${result.params || 'N/A'}</td>
                            <td>${result.quant || 'N/A'}</td>
                            <td>${result.num_ctx || 'N/A'}</td>
                            <td>${result.success ? result.size_gb + ' GB' : 'N/A'}</td>
                            <td>${result.success ? result.vram_gb + ' GB' : 'N/A'}</td>
                            <td>${result.success ? result.offload_pct + '%' : 'N/A'}</td>
                            <td>${status}</td>
                        </tr>
                    `;
                }
                
                tableHtml += `
                        </tbody>
                    </table>
                `;
                
                resultsBox.innerHTML = tableHtml;
            } else {
                // Single result - show all details
                resultsBox.innerHTML = `
                    <div class="success-message">
                        <strong>VRAM Test Results for ${escapeHtml(data.model)}</strong><br>
                        Parameters: ${data.params || 'N/A'}<br>
                        Quantization: ${data.quant || 'N/A'}<br>
                        Context Size: ${data.num_ctx || 'N/A'}<br>
                        Total Size: ${data.size_gb} GB<br>
                        VRAM Usage: ${data.vram_gb} GB<br>
                        CPU Offload: ${data.offload_pct}%<br>
                        ${data.offload_pct > 0 ? '<br>⚠️ Model is using CPU offloading. Consider reducing num_ctx or using smaller quantization.' : '✓ Model fits entirely in VRAM'}
                    </div>
                `;
            }
        } else {
            resultsBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.error)}</div>`;
        }
    } catch (error) {
        resultsBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(error.message)}</div>`;
    }
}

let optimizerAbortController = null;

async function runOptimizer() {
    const modelName = document.getElementById('optimizer-model').value;
    const overheadGb = parseFloat(document.getElementById('optimizer-overhead').value);
    const maxTurns = parseInt(document.getElementById('optimizer-max-turns').value);
    const resultsBox = document.getElementById('optimizer-results');
    const runButton = document.getElementById('btn-run-optimizer');
    const stopButton = document.getElementById('btn-stop-optimizer');
    
    if (!modelName) {
        resultsBox.innerHTML = '<div class="error-message">Please select a model</div>';
        return;
    }
    
    // Show stop button, hide run button
    runButton.style.display = 'none';
    stopButton.style.display = 'inline-block';
    
    const iterations = maxTurns === 0 ? 'unlimited' : maxTurns;
    resultsBox.innerHTML = `<div class="loading">Running context optimizer (${iterations} iterations, ${overheadGb}GB overhead)... This may take several minutes.</div>`;
    
    // Create abort controller
    optimizerAbortController = new AbortController();
    
    try {
        const response = await fetch(`/api/performance/optimize/${encodeURIComponent(modelName)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                overhead_gb: overheadGb,
                max_turns: maxTurns
            }),
            signal: optimizerAbortController.signal
        });
        
        const data = await response.json();
        
        if (data.success) {
            let tableHtml = `
                <div class="success-message">
                    <strong>Context Optimization Results for ${escapeHtml(data.model)}</strong><br>
                    Max Context: ${data.max_context}<br>
                    Current Context: ${data.current_context || 'Default'}<br>
                    <strong>Recommended Context: ${data.optimal_context}</strong><br>
                    Available VRAM: ${data.available_vram_gb} GB
                </div>
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Context Size</th>
                            <th>VRAM Usage</th>
                            <th>CPU Offload</th>
                            <th>Fits in GPU?</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            for (const result of data.results) {
                const fits = result.fits ? '✓ Yes' : '✗ No';
                tableHtml += `
                    <tr ${result.context_size === data.optimal_context ? 'style="background-color: rgba(76, 175, 80, 0.1);"' : ''}>
                        <td>${result.context_size}</td>
                        <td>${result.vram_gb} GB</td>
                        <td>${result.offload_pct}%</td>
                        <td>${fits}</td>
                    </tr>
                `;
            }
            
            tableHtml += `
                    </tbody>
                </table>
                <div class="info-text" style="margin-top: 15px;">
                    To apply the recommended context size, edit the model's Modelfile and set:<br>
                    <code>PARAMETER num_ctx ${data.optimal_context}</code>
                </div>
            `;
            
            resultsBox.innerHTML = tableHtml;
        } else {
            resultsBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(data.error)}</div>`;
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            resultsBox.innerHTML = '<div class="error-message">Optimization stopped by user.</div>';
        } else {
            resultsBox.innerHTML = `<div class="error-message">Error: ${escapeHtml(error.message)}</div>`;
        }
    } finally {
        // Reset buttons
        runButton.style.display = 'inline-block';
        stopButton.style.display = 'none';
        optimizerAbortController = null;
    }
}

function stopOptimizer() {
    if (optimizerAbortController) {
        optimizerAbortController.abort();
    }
}

// ===== UTILITY FUNCTIONS =====

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (statusUpdateInterval) {
        clearInterval(statusUpdateInterval);
    }
});
