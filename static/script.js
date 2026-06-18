// AI Document Classifier - Dashboard Logic & Interactions

document.addEventListener('DOMContentLoaded', function() {
    // State Variables
    let selectedFile = null;
    let distributionChart = null;
    let modelsChart = null;
    let predictionsHistory = [];

    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfoBox = document.getElementById('file-info-box');
    const fileNameLabel = document.getElementById('file-name-label');
    const fileSizeLabel = document.getElementById('file-size-label');
    const fileIcon = document.getElementById('file-icon');
    const btnCancelUpload = document.getElementById('btn-cancel-upload');
    const btnClassify = document.getElementById('btn-classify');
    const uploadProgressBar = document.getElementById('upload-progress-bar');
    const progressPercent = document.getElementById('progress-percent');
    
    const resultCard = document.getElementById('result-card');
    const resultCategoryVal = document.getElementById('result-category-val');
    const resultFilenameVal = document.getElementById('result-filename-val');
    const resultTextPreview = document.getElementById('result-text-preview');
    const resultTime = document.getElementById('result-time');
    const btnClearResults = document.getElementById('btn-clear-results');
    
    const btnRetrain = document.getElementById('btn-retrain');
    const retrainIcon = document.getElementById('retrain-icon');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const themeIcon = document.getElementById('theme-icon');
    
    const searchHistory = document.getElementById('search-history');
    const historyTbody = document.getElementById('history-tbody');
    const historyCounter = document.getElementById('history-counter');

    // 1. Dark Mode Toggling
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        themeIcon.className = 'fa-solid fa-sun text-warning fs-4';
    }

    themeToggleBtn.addEventListener('click', function() {
        document.body.classList.toggle('dark-mode');
        if (document.body.classList.contains('dark-mode')) {
            localStorage.setItem('theme', 'dark');
            themeIcon.className = 'fa-solid fa-sun text-warning fs-4';
        } else {
            localStorage.setItem('theme', 'light');
            themeIcon.className = 'fa-solid fa-moon text-secondary fs-4';
        }
        updateChartsTheme();
    });

    // 2. Drag & Drop Handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });

    function handleFileSelection(file) {
        // Validate extension
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['txt', 'pdf', 'docx'].includes(ext)) {
            showToast('Unsupported File', 'Please upload a .pdf, .docx, or .txt file.', 'danger');
            return;
        }
        
        selectedFile = file;
        fileNameLabel.textContent = file.name;
        fileSizeLabel.textContent = formatBytes(file.size);
        
        // Match File Icon
        if (ext === 'pdf') {
            fileIcon.className = 'fa-solid fa-file-pdf text-danger fs-4 me-3';
        } else if (ext === 'docx') {
            fileIcon.className = 'fa-solid fa-file-word text-primary fs-4 me-3';
        } else {
            fileIcon.className = 'fa-solid fa-file-lines text-secondary fs-4 me-3';
        }
        
        fileInfoBox.classList.remove('d-none');
        btnClassify.classList.remove('d-none');
        
        // Reset progress bar
        uploadProgressBar.style.width = '0%';
        progressPercent.textContent = '0%';
    }

    btnCancelUpload.addEventListener('click', resetUploadZone);

    function resetUploadZone() {
        selectedFile = null;
        fileInput.value = '';
        fileInfoBox.classList.add('d-none');
        btnClassify.classList.add('d-none');
        uploadProgressBar.style.width = '0%';
        progressPercent.textContent = '0%';
    }

    // 3. Document Classification (using XMLHttpRequest for progress)
    btnClassify.addEventListener('click', function() {
        if (!selectedFile) return;
        
        btnClassify.disabled = true;
        btnClassify.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-1"></i> Classifying...';

        const formData = new FormData();
        formData.append('file', selectedFile);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload', true);

        // Upload progress
        xhr.upload.onprogress = function(e) {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                uploadProgressBar.style.width = percentComplete + '%';
                progressPercent.textContent = percentComplete + '%';
            }
        };

        xhr.onload = function() {
            btnClassify.disabled = false;
            btnClassify.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart me-1"></i> Classify Document';

            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                displayResults(response);
                showToast('Success', 'Document classified successfully!', 'success');
                loadStatsAndHistory(); // refresh graphs & tables
                resetUploadZone();
            } else {
                let errMsg = 'Failed to analyze document.';
                try {
                    const errRes = JSON.parse(xhr.responseText);
                    errMsg = errRes.error || errMsg;
                } catch(e) {}
                showToast('Classification Error', errMsg, 'danger');
            }
        };

        xhr.onerror = function() {
            btnClassify.disabled = false;
            btnClassify.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart me-1"></i> Classify Document';
            showToast('Network Error', 'A connection error occurred.', 'danger');
        };

        xhr.send(formData);
    });

    function displayResults(data) {
        resultCategoryVal.textContent = data.prediction;
        resultFilenameVal.textContent = data.filename;
        resultTextPreview.textContent = data.preview;
        resultTime.textContent = data.processing_time + 's';
        
        resultCard.classList.remove('d-none');
        // Scroll to results
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    btnClearResults.addEventListener('click', function() {
        resultCard.classList.add('d-none');
    });

    // 4. Load Analytics Stats & Prediction History
    function loadStatsAndHistory() {
        fetch('/stats')
            .then(res => res.json())
            .then(data => {
                // Update stats counts
                document.getElementById('stat-total-processed').textContent = data.total_predictions;
                
                const modelStats = data.model_stats;
                if (modelStats && modelStats.best_model) {
                    document.getElementById('stat-model-name').textContent = modelStats.best_model;
                    document.getElementById('stat-model-acc').textContent = `Accuracy: ${modelStats.accuracy}%`;
                    document.getElementById('stat-categories-count').textContent = modelStats.categories_count || '6';
                    document.getElementById('stat-dataset-size').textContent = modelStats.total_documents || '--';
                }
                
                predictionsHistory = data.recent_predictions || [];
                renderHistoryTable(predictionsHistory);
                updateCharts(data);
            })
            .catch(err => {
                console.error("Error loading stats:", err);
            });
    }

    // 5. Render History Table with search filter
    function renderHistoryTable(items) {
        if (!items || items.length === 0) {
            historyTbody.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center py-4 text-muted">
                        <i class="fa-regular fa-folder-open d-block fs-3 mb-2"></i>
                        No predictions found.
                    </td>
                </tr>
            `;
            historyCounter.textContent = "Showing 0 predictions";
            return;
        }

        let html = '';
        items.forEach(item => {
            html += `
                <tr>
                    <td><small class="text-muted">${item.timestamp}</small></td>
                    <td><span class="text-dark fw-medium text-truncate d-inline-block" style="max-width: 280px;" title="${item.filename}">${item.filename}</span></td>
                    <td><span class="badge bg-primary-subtle text-primary px-2.5 py-1.5">${item.prediction}</span></td>
                </tr>
            `;
        });
        historyTbody.innerHTML = html;
        historyCounter.textContent = `Showing ${items.length} predictions`;
    }

    // Client-side instant filter search
    searchHistory.addEventListener('input', function() {
        const query = searchHistory.value.toLowerCase().trim();
        if (!query) {
            renderHistoryTable(predictionsHistory);
            return;
        }
        
        const filtered = predictionsHistory.filter(item => {
            return item.filename.toLowerCase().includes(query) || 
                   item.prediction.toLowerCase().includes(query) ||
                   item.timestamp.toLowerCase().includes(query);
        });
        renderHistoryTable(filtered);
    });

    // 6. Chart.js Implementation
    function initCharts() {
        // Chart 1: Category Distribution Pie Chart
        const ctxDist = document.getElementById('chart-distribution').getContext('2d');
        distributionChart = new Chart(ctxDist, {
            type: 'doughnut',
            data: {
                labels: ['Resume', 'Invoice', 'Report', 'Letter', 'Legal', 'Research'],
                datasets: [{
                    data: [0, 0, 0, 0, 0, 0],
                    backgroundColor: [
                        '#3b82f6', // Blue
                        '#10b981', // Green
                        '#f59e0b', // Yellow
                        '#ef4444', // Red
                        '#8b5cf6', // Purple
                        '#06b6d4'  // Cyan
                    ],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false // Display custom legend or keep it simple
                    }
                },
                cutout: '65%'
            }
        });

        // Chart 2: Model Performance Comparison Bar Chart
        const ctxModels = document.getElementById('chart-models').getContext('2d');
        modelsChart = new Chart(ctxModels, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                datasets: [
                    {
                        label: 'Naive Bayes',
                        data: [0, 0, 0, 0],
                        backgroundColor: 'rgba(59, 130, 246, 0.4)',
                        borderColor: '#3b82f6',
                        borderWidth: 1.5,
                        borderRadius: 4
                    },
                    {
                        label: 'Logistic Regression',
                        data: [0, 0, 0, 0],
                        backgroundColor: 'rgba(16, 185, 129, 0.4)',
                        borderColor: '#10b981',
                        borderWidth: 1.5,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 12,
                            font: { size: 10 }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { font: { size: 9 } }
                    },
                    x: {
                        ticks: { font: { size: 9 } }
                    }
                }
            }
        });
        
        updateChartsTheme();
    }

    function updateCharts(data) {
        // Update Distribution Data
        const dist = data.distribution || {};
        const labels = distributionChart.data.labels;
        const distData = labels.map(label => dist[label] || 0);
        
        // If no data, show default state
        const total = distData.reduce((a, b) => a + b, 0);
        if (total === 0) {
            // Equal portions for styling placeholder
            distributionChart.data.datasets[0].data = [1, 1, 1, 1, 1, 1];
        } else {
            distributionChart.data.datasets[0].data = distData;
        }
        distributionChart.update();

        // Update Models Data
        const modelStats = data.model_stats;
        if (modelStats && modelStats.models_comparison) {
            const nb = modelStats.models_comparison.NaiveBayes || { accuracy: 0, precision: 0, recall: 0, f1_score: 0 };
            const lr = modelStats.models_comparison.LogisticRegression || { accuracy: 0, precision: 0, recall: 0, f1_score: 0 };

            modelsChart.data.datasets[0].data = [nb.accuracy, nb.precision, nb.recall, nb.f1_score];
            modelsChart.data.datasets[1].data = [lr.accuracy, lr.precision, lr.recall, lr.f1_score];
            modelsChart.update();
        }
    }

    function updateChartsTheme() {
        if (!distributionChart || !modelsChart) return;
        
        const isDark = document.body.classList.contains('dark-mode');
        const textColor = isDark ? '#94a3b8' : '#64748b';
        const gridColor = isDark ? '#1f2937' : '#e2e8f0';
        const cardBg = isDark ? '#111827' : '#ffffff';

        // Distribution Chart border
        distributionChart.data.datasets[0].borderColor = cardBg;
        distributionChart.update();

        // Models Chart scales
        modelsChart.options.scales.x.grid = { color: gridColor };
        modelsChart.options.scales.y.grid = { color: gridColor };
        modelsChart.options.scales.x.ticks.color = textColor;
        modelsChart.options.scales.y.ticks.color = textColor;
        
        // Legend text colors
        modelsChart.options.plugins.legend.labels.color = textColor;
        modelsChart.update();
    }

    // 7. Retrain Model Action
    btnRetrain.addEventListener('click', function() {
        btnRetrain.disabled = true;
        retrainIcon.classList.add('fa-spin');
        showToast('Training Active', 'Model training initiated in the background...', 'info');

        fetch('/train', { method: 'POST' })
            .then(res => {
                if (!res.ok) throw new Error('Training endpoint returned an error.');
                return res.json();
            })
            .then(data => {
                btnRetrain.disabled = false;
                retrainIcon.classList.remove('fa-spin');
                showToast('Training Complete', 'Models retrained and reloaded successfully!', 'success');
                loadStatsAndHistory();
            })
            .catch(err => {
                btnRetrain.disabled = false;
                retrainIcon.classList.remove('fa-spin');
                showToast('Training Failed', 'Model retraining failed: ' + err.message, 'danger');
            });
    });

    // 8. Utility Functions
    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // Bootstrap Toast Launcher
    function showToast(title, message, type = 'primary') {
        const toastEl = document.getElementById('status-toast');
        const toastTitle = document.getElementById('toast-title');
        const toastBody = document.getElementById('toast-body');
        
        toastTitle.textContent = title;
        toastBody.textContent = message;
        
        // Remove old header colors, assign new ones
        const header = toastEl.querySelector('.toast-header');
        header.className = 'toast-header';
        
        const notifyIcon = header.querySelector('i');
        notifyIcon.className = 'fa-solid me-2';
        
        if (type === 'success') {
            notifyIcon.classList.add('fa-circle-check', 'text-success');
        } else if (type === 'danger') {
            notifyIcon.classList.add('fa-circle-exclamation', 'text-danger');
        } else if (type === 'info') {
            notifyIcon.classList.add('fa-circle-info', 'text-info');
        } else {
            notifyIcon.classList.add('fa-bell', 'text-primary');
        }
        
        const toast = new bootstrap.Toast(toastEl, { delay: 5000 });
        toast.show();
    }

    // Initialize Dashboard
    initCharts();
    loadStatsAndHistory();
});
