document.addEventListener('DOMContentLoaded', () => {
    // --- Get references to new/changed elements ---
    const topImageFileInput = document.getElementById('topImageFile');
    const frontImageFileInput = document.getElementById('frontImageFile');
    const processUploadedBtn = document.getElementById('processUploadedBtn');
    const loadStatus = document.getElementById('loadStatus'); // Re-purposed for file status
    const topImagePreview = document.getElementById('topImage'); // Renamed for clarity
    const frontImagePreview = document.getElementById('frontImage'); // Renamed for clarity

    const statusDiv = document.getElementById('status');
    const consoleDiv = document.getElementById('console');
    const resultsBody = document.getElementById('results-body');
    const visualizationDiv = document.getElementById('visualization');

    // --- Helper Functions (updateStatus, logToConsole remain the same) ---
    function updateStatus(message, isError = false) {
        statusDiv.textContent = `Status: ${message}`;
        statusDiv.style.color = isError ? 'red' : 'black';
        logToConsole(message); // Also log to the console area
    }

    function logToConsole(message) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.textContent = `[${timestamp}] ${message}`;
        consoleDiv.appendChild(logEntry);
        consoleDiv.scrollTop = consoleDiv.scrollHeight; // Auto-scroll
    }

    // --- Function to check if both files are selected ---
    function checkFilesSelected() {
        const topFile = topImageFileInput.files.length > 0;
        const frontFile = frontImageFileInput.files.length > 0;
        processUploadedBtn.disabled = !(topFile && frontFile);
        if (topFile && frontFile) {
             loadStatus.textContent = "Ready to process.";
        } else if (topFile) {
             loadStatus.textContent = "Front view image missing.";
        } else if (frontFile) {
             loadStatus.textContent = "Top view image missing.";
        } else {
             loadStatus.textContent = "Select top and front view images.";
        }
    }

    // --- Function to preview selected image ---
    function previewImage(fileInput, imgElement) {
        if (fileInput.files && fileInput.files[0]) {
            const reader = new FileReader();
            reader.onload = function (e) {
                imgElement.src = e.target.result;
                imgElement.alt = `Preview: ${fileInput.files[0].name}`;
            }
            reader.readAsDataURL(fileInput.files[0]);
        } else {
            imgElement.src = "";
            imgElement.alt = imgElement.id === 'topImage' ? "Top View Image Preview" : "Front View Image Preview";
        }
    }

    // --- Event Listeners for File Inputs ---
    topImageFileInput.addEventListener('change', () => {
        previewImage(topImageFileInput, topImagePreview);
        checkFilesSelected();
    });

    frontImageFileInput.addEventListener('change', () => {
        previewImage(frontImageFileInput, frontImagePreview);
        checkFilesSelected();
    });

    // --- Event Listener for the new Process Uploaded Button ---
    processUploadedBtn.addEventListener('click', async () => {
        if (!topImageFileInput.files[0] || !frontImageFileInput.files[0]) {
            updateStatus("Please select both top and front view images.", true);
            return;
        }

        updateStatus("Processing uploaded images...");
        processUploadedBtn.disabled = true;

        // Create FormData to send files
        const formData = new FormData();
        formData.append('topImageFile', topImageFileInput.files[0]);
        formData.append('frontImageFile', frontImageFileInput.files[0]);

        // TODO: Append settings from the UI to formData if needed
        // Example: formData.append('pipelineMode', document.querySelector('input[name="pipelineMode"]:checked').value);
        // formData.append('temperature', document.getElementById('temperature').value);
        // ... etc ...

        try {
            const response = await fetch('/process_uploaded_images', { // Use the new endpoint
                method: 'POST',
                body: formData, // Send FormData directly, no need for Content-Type header
            });

            const result = await response.json();

             if (!response.ok) {
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }

            updateStatus(result.message || "Processing complete.");
            logToConsole(`Volume: ${result.volume !== undefined ? result.volume.toFixed(3) : 'N/A'} m³`);
            logToConsole(`Processing Time: ${result.processing_time !== undefined ? result.processing_time.toFixed(2) : 'N/A'}s`);

            // TODO: Update results table (add a new row)
            const newRow = resultsBody.insertRow();
            newRow.insertCell(0).textContent = resultsBody.rows.length; // Simple index
            newRow.insertCell(1).textContent = result.volume !== undefined ? result.volume.toFixed(3) : 'Error';
            newRow.insertCell(2).textContent = result.timestamp || new Date().toLocaleTimeString();
            // Add more cells if needed

            // TODO: Update 3D visualization if data is available
            // if (result.visualization) {
            //     render3DVisualization(result.visualization); // Adapt render function
            // }

        } catch (error) {
            console.error('Error processing uploaded images:', error);
            updateStatus(`Error processing images: ${error.message}`, true);
        } finally {
            // Re-enable button only if files are still selected (or clear selection)
            checkFilesSelected();
        }
    });

    // --- Placeholder for 3D rendering (adapt as needed) ---
    // function render3DVisualization(vizData) { ... }

    // --- Initial setup ---
    updateStatus("Application ready. Please select top and front view images.");
    checkFilesSelected(); // Initial check for button state

    // Toggle enhanced options visibility (remains the same)
    const enhancedRadio = document.querySelector('input[name="pipelineMode"][value="enhanced"]');
    const enhancedOptionsDiv = document.getElementById('enhanced-options');
    if (enhancedRadio && enhancedOptionsDiv) {
        document.querySelectorAll('input[name="pipelineMode"]').forEach(radio => {
            radio.addEventListener('change', (event) => {
                enhancedOptionsDiv.style.display = event.target.value === 'enhanced' ? 'block' : 'none';
            });
        });
        // Initial check
        enhancedOptionsDiv.style.display = enhancedRadio.checked ? 'block' : 'none';
    }

});
