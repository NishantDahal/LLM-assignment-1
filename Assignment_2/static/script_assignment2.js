document.addEventListener('DOMContentLoaded', function () {
    console.log("Assignment 2 script loaded.");

    // Elements for the UI
    const wordInput = document.getElementById('wordInput');
    const getEmbeddingButton = document.getElementById('getEmbeddingButton');
    const embeddingResultDiv = document.getElementById('embeddingResult');
    const neighborsResultDiv = document.getElementById('neighborsResult');
    const visualizationDiv = document.getElementById('visualization');
    const statusDiv = document.getElementById('modelStatus');
    
    // Check if models are loaded when the page loads
    checkModelStatus();
    
    // Set an interval to check model status every 5 seconds if they're not loaded
    const statusInterval = setInterval(() => {
        checkModelStatus().then(loaded => {
            if (loaded) {
                clearInterval(statusInterval);
                // Once models are loaded, fetch visualization data
                fetchVisualizationData();
            }
        });
    }, 5000);

    // Button event listener
    if (getEmbeddingButton) {
        getEmbeddingButton.addEventListener('click', function () {
            const word = wordInput.value.trim();
            const embeddingType = document.getElementById('embeddingTypeSelect').value; // tfidf or glove

            if (word) {
                fetchEmbedding(word, embeddingType);
                fetchNearestNeighbors(word, embeddingType);
            } else {
                embeddingResultDiv.textContent = 'Please enter a word.';
                neighborsResultDiv.textContent = '';
            }
        });
    }
    
    // Visualization method selection
    const vizMethodSelect = document.getElementById('vizMethodSelect');
    const vizTypeSelect = document.getElementById('vizTypeSelect');
    
    if (vizMethodSelect && vizTypeSelect) {
        vizMethodSelect.addEventListener('change', function() {
            fetchVisualizationData(
                vizMethodSelect.value,
                vizTypeSelect.value
            );
        });
        
        vizTypeSelect.addEventListener('change', function() {
            fetchVisualizationData(
                vizMethodSelect.value,
                vizTypeSelect.value
            );
        });
    }
});

async function checkModelStatus() {
    const statusDiv = document.getElementById('modelStatus');
    
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.models_loaded) {
            statusDiv.innerHTML = '<span class="status-loaded">✓ Models loaded</span>';
            statusDiv.classList.remove('status-loading');
            statusDiv.classList.add('status-loaded');
            return true;
        } else if (data.loading_in_progress) {
            statusDiv.innerHTML = '<span class="status-loading">⟳ Loading models...</span>';
            statusDiv.classList.add('status-loading');
            statusDiv.classList.remove('status-loaded');
            return false;
        } else {
            statusDiv.innerHTML = '<span class="status-error">⚠ Models not loaded</span>';
            statusDiv.classList.remove('status-loading');
            statusDiv.classList.remove('status-loaded');
            statusDiv.classList.add('status-error');
            return false;
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        statusDiv.innerHTML = '<span class="status-error">⚠ Error checking model status</span>';
        statusDiv.classList.add('status-error');
        return false;
    }
}

async function fetchEmbedding(word, type) {
    const embeddingResultDiv = document.getElementById('embeddingResult');
    embeddingResultDiv.innerHTML = '<div class="loading">Fetching embedding...</div>';
    
    try {
        const response = await fetch('/api/embedding', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ word: word, type: type }),
        });
        const data = await response.json();

        if (response.ok && data.embedding) {
            // Format the embedding vector for display - show first 10 dimensions
            const embeddingDisplay = data.embedding.slice(0, 10).map(val => val.toFixed(4)).join(', ');
            
            embeddingResultDiv.innerHTML = `
                <h3>Word: "${data.word}" <span class="embedding-type">${type}</span></h3>
                <div class="embedding-vector">
                    <strong>Embedding (first 10 dimensions):</strong><br>
                    [${embeddingDisplay}, ...]
                </div>
                <div class="embedding-info">
                    <p>Full vector has ${data.embedding.length} dimensions</p>
                </div>
            `;
        } else {
            embeddingResultDiv.innerHTML = `
                <div class="error-message">
                    <strong>Error:</strong> ${data.error || 'Failed to fetch embedding.'}
                </div>
            `;
        }
    } catch (error) {
        console.error('Error fetching embedding:', error);
        embeddingResultDiv.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> Could not connect to the server. See console for details.
            </div>
        `;
    }
}

async function fetchNearestNeighbors(word, type) {
    const neighborsResultDiv = document.getElementById('neighborsResult');
    neighborsResultDiv.innerHTML = '<div class="loading">Fetching nearest neighbors...</div>';

    try {
        const response = await fetch('/api/nearest_neighbors', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ word: word, type: type, n: 5 }), // Requesting 5 neighbors
        });
        const data = await response.json();

        if (response.ok && data.neighbors) {
            if (data.neighbors.length > 0) {
                let neighborsHtml = `<h3>Nearest Neighbors for "${data.word}" <span class="embedding-type">${type}</span></h3><ul class="neighbors-list">`;
                
                data.neighbors.forEach(neighbor => {
                    // Each neighbor is a tuple [word, similarity_score]
                    const similarity = (neighbor[1] * 100).toFixed(2); // Convert to percentage
                    neighborsHtml += `
                        <li>
                            <span class="neighbor-word">${neighbor[0]}</span>
                            <div class="similarity-bar">
                                <div class="similarity-fill" style="width: ${similarity}%"></div>
                                <span class="similarity-text">${similarity}%</span>
                            </div>
                        </li>`;
                });
                
                neighborsHtml += '</ul>';
                neighborsResultDiv.innerHTML = neighborsHtml;
            } else {
                neighborsResultDiv.innerHTML = `
                    <div class="info-message">
                        No neighbors found for "${data.word}" with ${type} embeddings.
                    </div>
                `;
            }
        } else {
            neighborsResultDiv.innerHTML = `
                <div class="error-message">
                    <strong>Error:</strong> ${data.error || 'Failed to fetch nearest neighbors.'}
                </div>
            `;
        }
    } catch (error) {
        console.error('Error fetching nearest neighbors:', error);
        neighborsResultDiv.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> Could not connect to the server. See console for details.
            </div>
        `;
    }
}

async function fetchVisualizationData(method = 'tsne', type = 'glove') {
    const visualizationDiv = document.getElementById('visualization');
    visualizationDiv.innerHTML = '<div class="loading">Fetching visualization data...</div>';
    
    try {
        const response = await fetch(`/api/visualize_embeddings?method=${method}&type=${type}`);
        const data = await response.json();
        
        if (response.ok && data.embeddings_data) {
            if (data.embeddings_data.length > 0) {
                visualizationDiv.innerHTML = '<canvas id="embeddingChart" width="600" height="400"></canvas>';
                renderScatterPlot(data.embeddings_data, method, type);
            } else {
                visualizationDiv.innerHTML = `
                    <div class="info-message">
                        No data available for visualization with ${method} reduction on ${type} embeddings.
                    </div>
                `;
            }
        } else {
            visualizationDiv.innerHTML = `
                <div class="error-message">
                    <strong>Error:</strong> ${data.error || 'Failed to fetch visualization data.'}
                </div>
            `;
        }
    } catch (error) {
        console.error('Error fetching visualization data:', error);
        visualizationDiv.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> Could not connect to the server. See console for details.
            </div>
        `;
    }
}

function renderScatterPlot(data, method, type) {
    const canvas = document.getElementById('embeddingChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Extract coordinates and labels
    const labels = data.map(item => item.word);
    const xValues = data.map(item => item.x);
    const yValues = data.map(item => item.y);
    
    // Find min and max for normalization
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    // Get canvas dimensions
    const width = canvas.width;
    const height = canvas.height;
    const padding = 50; // Padding for axes and labels
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw title and axes labels
    ctx.fillStyle = '#333';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`Word Embedding Visualization (${method.toUpperCase()} on ${type.toUpperCase()})`, width / 2, 20);
    
    // Draw points with labels
    ctx.font = '12px Arial';
    
    // Background style for points
    ctx.fillStyle = 'rgba(75, 192, 192, 0.2)';
    ctx.strokeStyle = 'rgba(75, 192, 192, 1)';
    
    // Draw each data point
    data.forEach(item => {
        // Normalize coordinates to canvas space
        const x = padding + ((item.x - xMin) / (xMax - xMin)) * (width - 2 * padding);
        const y = height - padding - ((item.y - yMin) / (yMax - yMin)) * (height - 2 * padding);
        
        // Draw point
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        
        // Draw label
        ctx.fillStyle = '#333';
        ctx.textAlign = 'center';
        ctx.fillText(item.word, x, y - 10);
    });
    
    // Draw axes
    ctx.beginPath();
    ctx.strokeStyle = '#999';
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(padding, padding);
    ctx.stroke();
    
    // Axes labels
    ctx.fillStyle = '#666';
    ctx.textAlign = 'center';
    ctx.fillText('Dimension 1', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Dimension 2', 0, 0);
    ctx.restore();
} 