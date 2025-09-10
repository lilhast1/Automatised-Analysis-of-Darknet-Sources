function shuffleArray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}
document.addEventListener('DOMContentLoaded', () => {
    const iframe = document.getElementById('html-viewer');
    const currentFileSpan = document.getElementById('current-file');
    const progressSpan = document.getElementById('progress');
    const totalFilesSpan = document.getElementById('total-files');
    const exportBtn = document.getElementById('export-csv');
    const skipBtn = document.getElementById('skip-file');
    const labelButtons = document.querySelectorAll('.label-btn');

    let currentIndex = 0;
    const labeledData = [];
    //shuffleArray(filesToLabel);
    const totalFiles = filesToLabel.length;

    totalFilesSpan.textContent = totalFiles;

    function loadNextFile() {
        if (currentIndex < totalFiles) {
            const filePath = filesToLabel[currentIndex];
            iframe.src = filePath;
            currentFileSpan.textContent = filePath;
            progressSpan.textContent = labeledData.length;
        } else {
            iframe.src = 'about:blank';
            currentFileSpan.textContent = 'All files labeled!';
            progressSpan.textContent = labeledData.length;
            alert('Congratulations! You have labeled all the files.');
        }
    }

    function handleLabelClick(event) {
        const label = event.target.getAttribute('data-label');
        const file = filesToLabel[currentIndex];
        labeledData.push({ file, label });
        currentIndex++;
        loadNextFile();
    }

    function handleSkipClick() {
        currentIndex++;
        loadNextFile();
    }

    function exportToCsv() {
        if (labeledData.length === 0) {
            alert('No files have been labeled yet.');
            return;
        }

        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "file,label\n"; // CSV Header

        labeledData.forEach(row => {
            csvContent += `${row.file},${row.label}\n`;
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "labeled_html_files.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    labelButtons.forEach(button => {
        button.addEventListener('click', handleLabelClick);
    });

    exportBtn.addEventListener('click', exportToCsv);
    skipBtn.addEventListener('click', handleSkipClick);

    // Load the first file
    loadNextFile();
});