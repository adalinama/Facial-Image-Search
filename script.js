const searchForm = document.getElementById("search-form");
const searchBox = document.getElementById("search-box");
const searchResult = document.getElementById("search-result");
const showMoreBtn = document.getElementById("show-more-btn");
const modal = document.getElementById('image-modal');
const modalImage = document.getElementById('modal-image');
const captionText = document.getElementById('caption');
const closeBtn = document.querySelector('.close');
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");


let keyword = "";
let page = 1;
let imageData = [];
const itemsPerPage = 15; // Number of items per page

// Load and parse CSV file
async function loadCSVData() {
    const response = await fetch('clustering_results.csv'); // Ensure the path is correct
    const csvData = await response.text();

    Papa.parse(csvData, {
        header: true,
        complete: function(results) {
            imageData = results.data;
            console.log('Parsed image data:', imageData); // Check the parsed data in console
        }
    });
}

async function searchImages() {
    if (imageData.length === 0) {
        console.warn('Image data not loaded yet.');
        return; // Do not proceed if the data isn't loaded
    }

    keyword = searchBox.value.toLowerCase().trim();
    const results = imageData.filter(image =>
        image['person'] && image['person'].toLowerCase().includes(keyword) ||
        image['relationship'] && image['relationship'].toLowerCase().includes(keyword)
    );

    if (page === 1) {
        searchResult.innerHTML = "";
    }

    displayResults(results);
    showMoreBtn.style.display = results.length > page * itemsPerPage ? "block" : "none";
}

function displayResults(results) {
    const start = (page - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const paginatedResults = results.slice(start, end);

    paginatedResults.forEach(image => {
        const imgElement = document.createElement("img");
        imgElement.src = image['image path'];
        imgElement.alt = image['image path']; // Use image path as alt text
        imgElement.classList.add('thumbnail'); // Add this class for styling and modal functionality
        imgElement.style.width = "150px"; // Adjust size as needed
        imgElement.style.margin = "10px";
        searchResult.appendChild(imgElement);
    });
}

// Event listener for image clicks to open the modal
searchResult.addEventListener('click', (event) => {
    if (event.target && event.target.classList.contains('thumbnail')) {
        modal.style.display = 'block';
        modalImage.src = event.target.src;
        captionText.innerHTML = event.target.src; // Set caption to image path
    }
});

// Close the modal when the user clicks on <span> (x)
closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
});

// Close the modal when clicking anywhere outside the modal
window.addEventListener('click', (event) => {
    if (event.target === modal) {
        modal.style.display = 'none';
    }
});

searchForm.addEventListener("submit", (e) => {
    e.preventDefault();
    page = 1;
    searchImages();
});

showMoreBtn.addEventListener("click", () => {
    page++;
    searchImages();
});
document.addEventListener('DOMContentLoaded', () => {
    const textSearchBtn = document.getElementById('text-search-btn');
    const imageSearchBtn = document.getElementById('image-search-btn');
    const searchForm = document.getElementById('search-form');
    const uploadForm = document.getElementById('upload-form');
    
    textSearchBtn.addEventListener('click', () => {
        searchForm.classList.add('active');
        uploadForm.classList.remove('active');
        textSearchBtn.classList.add('active');
        imageSearchBtn.classList.remove('active');
    });
    
    imageSearchBtn.addEventListener('click', () => {
        searchForm.classList.remove('active');
        uploadForm.classList.add('active');
        textSearchBtn.classList.remove('active');
        imageSearchBtn.classList.add('active');
    });

    // Ensure the text search form is shown by default
    searchForm.classList.add('active');
    textSearchBtn.classList.add('active');
});



uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!fileInput.files[0]) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        if (data.error) {
            alert(data.error);
            return;
        }

        const results = data.results;
        displayResults(results);

        showMoreBtn.style.display = results.length > itemsPerPage ? "block" : "none";
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to upload image.');
    }
});

function displayResults(results) {
    searchResult.innerHTML = "";
    results.forEach(image => {
        const imgElement = document.createElement("img");
        imgElement.src = image['imagePath']; // Adjust based on your server response
        imgElement.alt = image['imagePath']; // Use image path as alt text
        imgElement.classList.add('thumbnail');
        imgElement.style.width = "150px"; // Adjust size as needed
        imgElement.style.margin = "10px";
        searchResult.appendChild(imgElement);
    });
}

loadCSVData();
