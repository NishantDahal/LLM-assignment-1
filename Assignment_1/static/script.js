document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const operationSelect = document.getElementById('operation-select');
    const processBtn = document.getElementById('process-btn');
    const outputDisplay = document.getElementById('output-display');

    const wordsInput = document.getElementById('words-input');
    const compareBtn = document.getElementById('compare-btn');
    const comparisonTableBody = document.querySelector('#comparison-table tbody');
    const comparisonExplanation = document.getElementById('comparison-explanation');

    processBtn.addEventListener('click', async () => {
        const text = textInput.value;
        const operation = operationSelect.value;

        if (!text.trim()) {
            outputDisplay.textContent = 'Please enter some text.';
            return;
        }

        outputDisplay.textContent = 'Processing...';

        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text, operation })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            outputDisplay.textContent = JSON.stringify(data, null, 2);

        } catch (error) {
            outputDisplay.textContent = `Error: ${error.message}`;
            console.error('Error processing text:', error);
        }
    });

    compareBtn.addEventListener('click', async () => {
        const wordsString = wordsInput.value;
        if (!wordsString.trim()) {
            comparisonTableBody.innerHTML = '';
            comparisonExplanation.textContent = 'Please enter some words.';
            return;
        }

        const wordsArray = wordsString.split(',').map(word => word.trim()).filter(word => word.length > 0);

        if (wordsArray.length === 0) {
            comparisonTableBody.innerHTML = '';
            comparisonExplanation.textContent = 'Please enter valid, comma-separated words.';
            return;
        }
        
        comparisonTableBody.innerHTML = ''; // Clear previous results
        comparisonExplanation.textContent = 'Comparing...';

        try {
            const response = await fetch('/compare_stem_lemma', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ words: wordsArray })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Populate table
            data.comparison.forEach(item => {
                const row = comparisonTableBody.insertRow();
                row.insertCell().textContent = item.word;
                row.insertCell().textContent = item.stem;
                row.insertCell().textContent = item.lemma;
            });

            // Display explanation
            comparisonExplanation.textContent = data.explanation.join('\n');

        } catch (error) {
            comparisonExplanation.textContent = `Error: ${error.message}`;
            console.error('Error comparing words:', error);
        }
    });
}); 