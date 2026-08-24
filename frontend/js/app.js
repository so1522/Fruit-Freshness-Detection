document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const selectFileBtn = document.getElementById('selectFileBtn');
    const cameraBtn = document.getElementById('cameraBtn');
    
    const previewContainer = document.getElementById('previewContainer');
    const previewImg = document.getElementById('previewImg');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    
    const resultsCard = document.getElementById('resultsCard');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    const statusBanner = document.getElementById('statusBanner');
    const statusTitle = document.getElementById('statusTitle');
    const statusSubtitle = document.getElementById('statusSubtitle');
    const confidenceVal = document.getElementById('confidenceVal');
    
    const detailFruit = document.getElementById('detailFruit');
    const detailState = document.getElementById('detailState');
    const detailModel = document.getElementById('detailModel');
    const detailTime = document.getElementById('detailTime');
    
    const adviceTitle = document.getElementById('adviceTitle');
    const adviceText = document.getElementById('adviceText');
    
    const sampleChipsContainer = document.getElementById('sampleChips');

    // Camera Modal Elements
    const cameraModal = document.getElementById('cameraModal');
    const webcamVideo = document.getElementById('webcamVideo');
    const captureBtn = document.getElementById('captureBtn');
    const closeCameraBtn = document.getElementById('closeCameraBtn');

    // Chart.js Instances
    let probabilityChart = null;
    let metricsChart = null;
    let webcamStream = null;

    // Initialize Charts
    initCharts();

    // Load available sample images
    fetchSampleImages();

    // Event Listeners for Upload
    selectFileBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    });

    // Camera Modal Handlers
    cameraBtn.addEventListener('click', openCamera);
    closeCameraBtn.addEventListener('click', closeCamera);
    captureBtn.addEventListener('click', captureWebcamImage);

    // Functions
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WEBP).');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            showPreview(e.target.result);
            uploadAndPredictFile(file);
        };
        reader.readAsDataURL(file);
    }

    function showPreview(imageSrc) {
        previewImg.src = imageSrc;
        previewImg.style.display = 'block';
        previewPlaceholder.style.display = 'none';
    }

    function showLoading(show) {
        if (show) {
            loadingOverlay.classList.add('active');
        } else {
            loadingOverlay.classList.remove('active');
        }
    }

    async function fetchSampleImages() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                sampleChipsContainer.innerHTML = '';
                data.samples.forEach(sample => {
                    const chip = document.createElement('button');
                    chip.className = 'chip';
                    chip.innerText = sample.name;
                    chip.addEventListener('click', () => loadPresetSample(sample.filename));
                    sampleChipsContainer.appendChild(chip);
                });
            }
        } catch (err) {
            console.warn('Failed to load sample presets:', err);
        }
    }

    async function loadPresetSample(filename) {
        const sampleUrl = `/api/sample-image/${filename}`;
        showPreview(sampleUrl);
        showLoading(true);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sample: filename })
            });
            const result = await response.json();
            showLoading(false);
            if (result.error) {
                alert(result.error);
            } else {
                updateUIWithResults(result);
            }
        } catch (err) {
            showLoading(false);
            console.error('Error analyzing sample:', err);
            alert('Failed to connect to backend server.');
        }
    }

    async function uploadAndPredictFile(file) {
        showLoading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            showLoading(false);
            if (result.error) {
                alert(result.error);
            } else {
                updateUIWithResults(result);
            }
        } catch (err) {
            showLoading(false);
            console.error('Prediction API Error:', err);
            alert('Failed to connect to backend server.');
        }
    }

    async function uploadBase64Predict(base64Image) {
        showLoading(true);
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image })
            });
            const result = await response.json();
            showLoading(false);
            if (result.error) {
                alert(result.error);
            } else {
                updateUIWithResults(result);
            }
        } catch (err) {
            showLoading(false);
            console.error('Base64 Prediction API Error:', err);
            alert('Failed to connect to server.');
        }
    }

    function updateUIWithResults(data) {
        // Status Banner Styling
        statusBanner.className = 'status-banner ' + (data.is_fresh ? 'fresh' : 'rotten');
        statusTitle.innerText = data.display_name;
        statusSubtitle.innerText = data.is_fresh ? 'Fresh Fruit - Safe for Consumption' : 'Rotten / Decayed Fruit';
        confidenceVal.innerText = `${data.confidence}%`;

        // Details List
        detailFruit.innerText = data.fruit_type;
        detailState.innerText = data.is_fresh ? 'Fresh' : 'Rotten';
        detailModel.innerText = data.model_used;
        detailTime.innerText = `${data.processing_time_sec}s`;

        // Advice Box
        adviceTitle.innerText = data.advice.title || 'Storage Guidance';
        adviceText.innerText = `${data.advice.storage} (${data.advice.recommendation})`;

        // Update Charts
        updateProbabilityChart(data.probabilities);
        updateMetricsChart(data.metrics);
    }

    function initCharts() {
        // Probability Bar Chart
        const probCtx = document.getElementById('probabilityChart').getContext('2d');
        probabilityChart = new Chart(probCtx, {
            type: 'bar',
            data: {
                labels: ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange'],
                datasets: [{
                    label: 'Confidence (%)',
                    data: [0, 0, 0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(244, 63, 94, 0.7)',
                        'rgba(244, 63, 94, 0.7)',
                        'rgba(244, 63, 94, 0.7)'
                    ],
                    borderColor: [
                        '#10b981', '#10b981', '#10b981',
                        '#f43f5e', '#f43f5e', '#f43f5e'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    },
                    x: {
                        ticks: { color: '#cbd5e1', font: { size: 11 } },
                        grid: { display: false }
                    }
                }
            }
        });

        // Feature Radar/Bar Chart
        const metricsCtx = document.getElementById('metricsChart').getContext('2d');
        metricsChart = new Chart(metricsCtx, {
            type: 'radar',
            data: {
                labels: ['Red (R)', 'Green (G)', 'Blue (B)', 'Hue (H)', 'Saturation (S)', 'Value (V)'],
                datasets: [{
                    label: 'Color Space Intensity',
                    data: [0, 0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    pointBackgroundColor: '#60a5fa'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#cbd5e1' } }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#94a3b8', font: { size: 11 } },
                        ticks: { display: false }
                    }
                }
            }
        });
    }

    function updateProbabilityChart(probabilities) {
        if (!probabilityChart) return;
        const labels = Object.keys(probabilities);
        const dataValues = Object.values(probabilities);

        probabilityChart.data.labels = labels;
        probabilityChart.data.datasets[0].data = dataValues;
        probabilityChart.update();
    }

    function updateMetricsChart(metrics) {
        if (!metricsChart || !metrics || !metrics.color_stats) return;
        const color = metrics.color_stats;
        metricsChart.data.datasets[0].data = [
            color.R_mean || 0,
            color.G_mean || 0,
            color.B_mean || 0,
            color.H_mean || 0,
            color.S_mean || 0,
            color.V_mean || 0
        ];
        metricsChart.update();
    }

    // Camera Modal Functions
    async function openCamera() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            webcamVideo.srcObject = webcamStream;
            cameraModal.classList.add('active');
        } catch (err) {
            console.error('Webcam Access Error:', err);
            alert('Unable to access camera. Please check browser permissions.');
        }
    }

    function closeCamera() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        cameraModal.classList.remove('active');
    }

    function captureWebcamImage() {
        const canvas = document.createElement('canvas');
        canvas.width = webcamVideo.videoWidth || 640;
        canvas.height = webcamVideo.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
        
        const base64Data = canvas.toDataURL('image/jpeg');
        showPreview(base64Data);
        closeCamera();
        uploadBase64Predict(base64Data);
    }
});
