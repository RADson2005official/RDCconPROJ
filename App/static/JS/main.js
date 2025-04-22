document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements ---
    const topImageFileInput = document.getElementById('topImageFile');
    const frontImageFileInput = document.getElementById('frontImageFile');
    const topImagePreview = document.getElementById('topImage');
    const frontImagePreview = document.getElementById('frontImage');
    const topEdgeImage = document.getElementById('topEdgeImage');
    const frontEdgeImage = document.getElementById('frontEdgeImage');
    const topEdgePlaceholder = document.getElementById('topEdgePlaceholder');
    const frontEdgePlaceholder = document.getElementById('frontEdgePlaceholder');

    // Ensure placeholders are correctly selected (assuming they are the next sibling div)
    const topPlaceholder = topImagePreview?.nextElementSibling;
    const frontPlaceholder = frontImagePreview?.nextElementSibling;
    const processUploadedBtn = document.getElementById('processUploadedBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loadStatus = document.getElementById('loadStatus');
    const resultsBody = document.getElementById('results-body');
    const consoleDiv = document.getElementById('console');
    const statusDiv = document.getElementById('status');
    const visualizationDiv = document.getElementById('visualization');

    // --- Function to update status bar ---
    function updateStatus(message, isError = false) {
        if (!statusDiv) return; // Guard against missing element
        const statusDot = statusDiv.querySelector('.status-dot');
        const statusTextNode = statusDiv.childNodes[1]; // Assuming text is the second node

        if (statusTextNode) {
             statusTextNode.nodeValue = ` ${message}`; // Update text node
        } else {
             // Fallback if structure is different
             statusDiv.textContent = message;
        }

        if (statusDot) {
            if (isError) {
                statusDot.classList.remove('active');
                statusDot.classList.add('error');
            } else {
                statusDot.classList.remove('error');
                statusDot.classList.add('active');
            }
        }
        addConsoleEntry(message, isError ? 'error' : 'info');
    }

    // --- Function to add console entry ---
    function addConsoleEntry(message, type = 'info') {
        if (!consoleDiv) return; // Guard against missing element
        const entry = document.createElement('div');
        entry.classList.add('console-entry', `console-${type}`);
        // Sanitize message slightly before adding
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message.toString()}`;
        consoleDiv.insertBefore(entry, consoleDiv.firstChild);
        if (consoleDiv.children.length > 50) {
            consoleDiv.removeChild(consoleDiv.lastChild);
        }
    }

    // --- Function to check if both files are selected ---
    function checkFilesSelected() {
        const topFile = topImageFileInput?.files?.length > 0;
        const frontFile = frontImageFileInput?.files?.length > 0;
        if(processUploadedBtn) processUploadedBtn.disabled = !(topFile && frontFile);

        if (loadStatus) {
            if (topFile && frontFile) {
                 loadStatus.textContent = "Ready to process.";
                 loadStatus.classList.remove('text-danger');
                 loadStatus.classList.add('text-success');
            } else if (topFile) {
                 loadStatus.textContent = "Front view image missing.";
                 loadStatus.classList.add('text-danger');
                 loadStatus.classList.remove('text-success');
            } else if (frontFile) {
                 loadStatus.textContent = "Top view image missing.";
                 loadStatus.classList.add('text-danger');
                 loadStatus.classList.remove('text-success');
            } else {
                 loadStatus.textContent = "Select top and front view images.";
                 loadStatus.classList.remove('text-danger', 'text-success');
            }
        }
    }

    // --- Function to preview selected image ---
    function previewImage(fileInput, imgElement, placeholderElement) {
        // Ensure elements exist before proceeding
        if (!fileInput || !imgElement) {
             console.warn("Preview function called with missing elements.");
             return;
        }

        if (fileInput.files && fileInput.files[0]) {
            const file = fileInput.files[0];
            // Basic check for image type (optional but good practice)
            if (!file.type.startsWith('image/')){
                updateStatus(`Selected file is not an image: ${file.name}`, true);
                // Reset input if needed: fileInput.value = '';
                return;
            }

            const reader = new FileReader();

            reader.onload = function (e) {
                imgElement.src = e.target.result;
                imgElement.alt = `Preview: ${file.name}`;
                imgElement.style.display = 'block'; // Show the image
                if(placeholderElement) placeholderElement.style.display = 'none'; // Hide the placeholder
            }

            reader.onerror = function(e) {
                console.error("FileReader error:", e);
                updateStatus(`Error reading file: ${file.name}`, true);
                imgElement.src = ""; // Clear preview on error
                imgElement.style.display = 'none'; // Hide image element
                if(placeholderElement) placeholderElement.style.display = 'block'; // Show placeholder
            }

            reader.readAsDataURL(file); // Read the file as Data URL
        } else {
            // No file selected or selection cleared
            imgElement.src = "";
            imgElement.alt = imgElement.id === 'topImage' ? "Top View Image Preview" : "Front View Image Preview";
            imgElement.style.display = 'none'; // Hide the image
            if(placeholderElement) placeholderElement.style.display = 'block'; // Show the placeholder
        }
    }

    // --- Function to update edge detection images ---
    function updateEdgeDetectionImages(topEdgeUrl, frontEdgeUrl) {
        if (topEdgeUrl && topEdgeImage) {
            topEdgeImage.src = topEdgeUrl;
            topEdgeImage.onload = function() {
                topEdgeImage.style.display = 'block';
                if (topEdgePlaceholder) {
                    topEdgePlaceholder.style.display = 'none';
                }
            };
            addConsoleEntry("Top view edge detection updated");
        }

        if (frontEdgeUrl && frontEdgeImage) {
            frontEdgeImage.src = frontEdgeUrl;
            frontEdgeImage.onload = function() {
                frontEdgeImage.style.display = 'block';
                if (frontEdgePlaceholder) {
                    frontEdgePlaceholder.style.display = 'none';
                }
            };
            addConsoleEntry("Front view edge detection updated");
        }
    }

    // --- Event Listeners for File Inputs ---
    if (topImageFileInput) {
        topImageFileInput.addEventListener('change', () => {
            previewImage(topImageFileInput, topImagePreview, topPlaceholder);
            checkFilesSelected();
        });
    } else {
         console.error("Element with ID 'topImageFile' not found.");
    }

    if (frontImageFileInput) {
        frontImageFileInput.addEventListener('change', () => {
            previewImage(frontImageFileInput, frontImagePreview, frontPlaceholder);
            checkFilesSelected();
        });
     } else {
         console.error("Element with ID 'frontImageFile' not found.");
    }

    // --- Event Listener for the Process Uploaded Button ---
    if (processUploadedBtn) {
        processUploadedBtn.addEventListener('click', async () => {
            if (!topImageFileInput.files[0] || !frontImageFileInput.files[0]) {
                updateStatus("Please select both top and front view images.", true);
                return;
            }

            updateStatus("Processing uploaded images...");
            addConsoleEntry("Starting image processing...");
            processUploadedBtn.disabled = true;
            if(clearBtn) clearBtn.disabled = true; // Disable clear button during processing

            // Clear previous results and visualization
            if(resultsBody) resultsBody.innerHTML = '<tr><td colspan="5" class="text-center p-3"><div class="spinner-border spinner-border-sm" role="status"><span class="visually-hidden">Loading...</span></div> Processing...</td></tr>';
            if(visualizationDiv) visualizationDiv.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>';

            // Create FormData to send files
            const formData = new FormData();
            formData.append('topImageFile', topImageFileInput.files[0]);
            formData.append('frontImageFile', frontImageFileInput.files[0]);

            // Add material type and condition
            const materialTypeSelect = document.getElementById('materialType');
            if (materialTypeSelect) {
                formData.append('materialType', materialTypeSelect.value);
            }

            try {
                const response = await fetch('/process_uploaded_images', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.status === 'success') {
                    updateStatus("Processing successful.");
                    addConsoleEntry(`Volume calculated: ${data.volume} ${data.unit_volume}`);
                    addConsoleEntry(`Mass calculated: ${data.mass} ${data.unit_mass}`);
                    addConsoleEntry(`Density: ${data.density} ${data.unit_density}`);
                    addConsoleEntry(`Processing time: ${data.processing_time}s`);

                    // Display results in table
                    if (resultsBody) {
                        const newRow = `
                            <tr>
                                <td>${resultsBody.rows.length}</td>
                                <td>${data.volume} ${data.unit_volume}</td>
                                <td>${data.density} ${data.unit_density}</td>
                                <td>${data.mass} ${data.unit_mass}</td>
                                <td>${data.timestamp}</td>
                            </tr>
                        `;
                        // Clear placeholder/spinner before adding row
                        if (resultsBody.rows.length === 1 && resultsBody.rows[0].cells.length === 1 && resultsBody.rows[0].cells[0].querySelector('.spinner-border')) {
                             resultsBody.innerHTML = '';
                        }
                        resultsBody.insertAdjacentHTML('beforeend', newRow);
                    }

                    // Update edge detection images
                    updateEdgeDetectionImages(data.top_edge_url, data.front_edge_url);

                    // Log edge detection aperture information if available
                    if (data.top_aperture_used) {
                        addConsoleEntry(`Top view edge detection used aperture size: ${data.top_aperture_used}`);
                    }
                    if (data.front_aperture_used) {
                        addConsoleEntry(`Front view edge detection used aperture size: ${data.front_aperture_used}`);
                    }

                    // Display visualization if URL provided
                    if (visualizationDiv) {
                        if (data.visualization_3d_url) {
                            visualizationDiv.innerHTML = `<img src="${data.visualization_3d_url}" alt="3D Visualization" class="img-fluid rounded">`;
                            addConsoleEntry("3D Visualization generated.");
                        } else {
                            visualizationDiv.innerHTML = '<p class="text-muted">No visualization available.</p>';
                            addConsoleEntry("Visualization not generated or available.", 'warning');
                        }
                    }

                } else {
                    const errorMsg = data.error || `Server responded with status ${response.status}`;
                    updateStatus(`Processing failed: ${errorMsg}`, true);
                    if(resultsBody) resultsBody.innerHTML = `<tr><td colspan="5" class="text-center text-danger p-3">Processing failed: ${errorMsg}</td></tr>`;
                    if(visualizationDiv) visualizationDiv.innerHTML = '<p class="text-danger">Processing failed.</p>';
                }

            } catch (error) {
                console.error('Fetch error:', error);
                updateStatus(`Network or client-side error: ${error.message}`, true);
                if(resultsBody) resultsBody.innerHTML = `<tr><td colspan="5" class="text-center text-danger p-3">Error: ${error.message}</td></tr>`;
                if(visualizationDiv) visualizationDiv.innerHTML = '<p class="text-danger">An error occurred.</p>';
            } finally {
                // Re-enable buttons after processing attempt
                checkFilesSelected(); // Re-evaluates process button state based on file inputs
                if(clearBtn) clearBtn.disabled = false;
            }
        });
    } else {
         console.error("Element with ID 'processUploadedBtn' not found.");
    }

    // --- Event Listener for Clear Button ---
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            // Reset file inputs
            if(topImageFileInput) topImageFileInput.value = '';
            if(frontImageFileInput) frontImageFileInput.value = '';

            // Reset previews
            previewImage(topImageFileInput, topImagePreview, topPlaceholder);
            previewImage(frontImageFileInput, frontImagePreview, frontPlaceholder);

            // Reset edge detection images
            if(topEdgeImage) {
                topEdgeImage.src = '';
                topEdgeImage.style.display = 'none';
            }
            if(frontEdgeImage) {
                frontEdgeImage.src = '';
                frontEdgeImage.style.display = 'none';
            }

            // Show edge detection placeholders
            if(topEdgePlaceholder) topEdgePlaceholder.style.display = 'block';
            if(frontEdgePlaceholder) frontEdgePlaceholder.style.display = 'block';

            // Reset results table (add placeholder)
            if(resultsBody) resultsBody.innerHTML = '<tr><td colspan="5" class="text-center text-muted p-3">No results yet.</td></tr>';

            // Reset visualization area
            if(visualizationDiv) visualizationDiv.innerHTML = '<p class="text-muted">Visualization Area</p>';

            // Reset console (optional, or add clear message)
            addConsoleEntry("Inputs and results cleared.", 'info');

            // Reset status and buttons
            checkFilesSelected();
            updateStatus("Application ready.");
        });
    } else {
         console.error("Element with ID 'clearBtn' not found.");
    }

    // Edge view toggle button functionality
    const edgeViewToggleBtn = document.getElementById('edgeViewToggleBtn');
    if (edgeViewToggleBtn) {
        edgeViewToggleBtn.addEventListener('click', function() {
            // Get the edge detection card's body
            const cardBody = this.closest('.card').querySelector('.card-body');

            // Toggle expanded class for larger view
            if (cardBody.classList.contains('expanded')) {
                cardBody.classList.remove('expanded');
                this.innerHTML = '<i class="bi bi-arrows-angle-expand"></i>';
            } else {
                cardBody.classList.add('expanded');
                this.innerHTML = '<i class="bi bi-arrows-angle-contract"></i>';
            }
        });
    }

    // Initial setup
    updateStatus("Application initialized.");
    addConsoleEntry("System ready. Select images to begin.");
    checkFilesSelected(); // Set initial button state
}); // End DOMContentLoaded