document.addEventListener('DOMContentLoaded', () => {
    const labelMapping = {
        'Irrelevant': 0,
        'Attack as a service': 1,
        'Attacker as a service': 2,
        'Malware as a service': 3
    };
    const reverseLabelMapping = Object.fromEntries(Object.entries(labelMapping).map(([k, v]) => [v, k]));

    let pseudolabels = [];
    let currentIndex = 0;
    let originalData = []; // To store the original data for comparison or full save

    // DOM Elements
    const currentFileDisplay = document.getElementById('current-file-path');
    const currentLabelDisplay = document.getElementById('current-label-display');
    const confidenceDisplay = document.getElementById('confidence-display');
    const fileContentFrame = document.getElementById('file-content-frame');
    const labelButtonsContainer = document.getElementById('label-buttons');
    const acceptButton = document.getElementById('accept-button');
    const saveButton = document.getElementById('save-button');
    const statusMessage = document.getElementById('status-message');
    const totalEntriesSpan = document.getElementById('total-entries');
    const currentIndexSpan = document.getElementById('current-index');

    // Function to load CSV data
    async function loadCSV(filePath) {
        try {
            const response = await fetch(filePath);
            const text = await response.text();
            const lines = text.trim().split('\n');
            const headers = lines[0].split(',');
            const data = lines.slice(1).map(line => {
                const values = line.split(',');
                let row = {};
                headers.forEach((header, i) => {
                    row[header.trim()] = values[i].trim();
                });
                return row;
            });
            originalData = JSON.parse(JSON.stringify(data)); // Deep copy
            pseudolabels = data;
            totalEntriesSpan.textContent = pseudolabels.length;
            displayEntry(currentIndex);
            createLabelButtons();
        } catch (error) {
            console.error("Error loading CSV:", error);
            statusMessage.textContent = "Error loading CSV data. Make sure 'pseudolabels.csv' exists.";
            statusMessage.style.backgroundColor = '#f8d7da';
            statusMessage.style.borderColor = '#f5c6cb';
            statusMessage.style.color = '#721c24';
        }
    }

    // Function to display an entry
    function displayEntry(index) {
        if (index >= 0 && index < pseudolabels.length) {
            const entry = pseudolabels[index];
            currentIndex = index; // Update global current index
            currentIndexSpan.textContent = currentIndex + 1;

            currentFileDisplay.textContent = `File: ${entry.file}`;
            currentLabelDisplay.textContent = entry.label;
            confidenceDisplay.textContent = parseFloat(entry.confidence).toFixed(2);

            // Load HTML file into iframe
            fileContentFrame.src = entry.file;

            // Highlight the currently selected label button
            Array.from(labelButtonsContainer.children).forEach(button => {
                if (button.textContent === entry.label) {
                    button.classList.add('selected');
                } else {
                    button.classList.remove('selected');
                }
            });
            statusMessage.textContent = ''; // Clear status
        } else if (index >= pseudolabels.length) {
            statusMessage.textContent = "All entries reviewed! Click 'Save Reviewed & Quit' to download.";
            statusMessage.style.backgroundColor = '#d4edda';
            statusMessage.style.borderColor = '#c3e6cb';
            statusMessage.style.color = '#155724';
            acceptButton.disabled = true;
            Array.from(labelButtonsContainer.children).forEach(button => button.disabled = true);
            currentFileDisplay.textContent = 'All Done!';
            currentLabelDisplay.textContent = '-';
            confidenceDisplay.textContent = '-';
            fileContentFrame.srcdoc = '<h1>Review Complete!</h1><p>Please save your work.</p>';
        }
    }

    // Function to create label buttons
    function createLabelButtons() {
        labelButtonsContainer.innerHTML = ''; // Clear existing buttons
        for (const labelName in labelMapping) {
            const button = document.createElement('button');
            button.textContent = labelName;
            button.classList.add('label-button');
            button.addEventListener('click', () => {
                // Update the label in the data
                pseudolabels[currentIndex].label = labelName;
                // Visually update the current label display
                currentLabelDisplay.textContent = labelName;
                // Highlight the selected button
                Array.from(labelButtonsContainer.children).forEach(btn => btn.classList.remove('selected'));
                button.classList.add('selected');
                statusMessage.textContent = `Label changed to '${labelName}'.`;
                statusMessage.style.backgroundColor = '#e0f7fa';
                statusMessage.style.borderColor = '#b2ebf2';
                statusMessage.style.color = '#006064';
            });
            labelButtonsContainer.appendChild(button);
        }
    }

    // Event Listener for Accept & Next
    acceptButton.addEventListener('click', () => {
        if (currentIndex < pseudolabels.length - 1) {
            currentIndex++;
            displayEntry(currentIndex);
        } else if (currentIndex === pseudolabels.length - 1) {
            currentIndex++; // Move past the last entry
            displayEntry(currentIndex); // Trigger "All entries reviewed" state
        }
    });

    // Event Listener for Save & Quit
    saveButton.addEventListener('click', () => {
        downloadCSV(pseudolabels, 'reviewed_pseudolabels.csv');
        statusMessage.textContent = "Reviewed data downloaded! You can now close this page.";
        statusMessage.style.backgroundColor = '#d4edda';
        statusMessage.style.borderColor = '#c3e6cb';
        statusMessage.style.color = '#155724';
    });

    // Function to download data as CSV
    function downloadCSV(data, filename) {
        const headers = Object.keys(data[0]);
        const csvRows = [];
        csvRows.push(headers.join(',')); // Add headers

        for (const row of data) {
            const values = headers.map(header => {
                const escaped = ('' + row[header]).replace(/"/g, '""'); // Escape double quotes
                return `"${escaped}"`; // Enclose all values in quotes
            });
            csvRows.push(values.join(','));
        }

        const csvString = csvRows.join('\n');
        const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.setAttribute('download', filename);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // Initial load
    loadCSV('labeled_manual_master.csv');
});