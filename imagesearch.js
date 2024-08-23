document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select an image file.');
            return;
        }

        // Create a FormData object to send the file to the server
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', { // Replace with your upload endpoint
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Upload successful:', result);
                // You might want to update the UI or notify the user here
            } else {
                console.error('Upload failed:', response.statusText);
            }
        } catch (error) {
            console.error('Error:', error);
        }
    });
});
