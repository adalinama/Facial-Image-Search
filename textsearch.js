document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const searchBox = document.getElementById('search-box');
    const searchResult = document.getElementById('search-result');
    const showMoreBtn = document.getElementById('show-more-btn');
    const itemsPerPage = 15; // Number of items per page

    let keyword = "";
    let page = 1;
    let imageData = [];

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

        if(page === 1){
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
            imgElement.alt = `${image['person']} - ${image['relationship']}`;
            imgElement.style.width = "150px"; // Adjust size as needed
            imgElement.style.margin = "10px";
            imgElement.addEventListener('click', () => {
                showImageModal(image['image path']);
            });
            searchResult.appendChild(imgElement);
        });
    }

    function showImageModal(imagePath) {
        const modal = document.getElementById('image-modal');
        const modalImage = document.getElementById('modal-image');
        const caption = document.getElementById('caption');

        modal.style.display = 'block';
        modalImage.src = imagePath;
        caption.innerText = imagePath;

        document.querySelector('.close').onclick = () => {
            modal.style.display = 'none';
        };
    }

    searchForm.addEventListener("submit", (e) => {
        e.preventDefault();
        page = 1;
        searchImages();
    });

    showMoreBtn.addEventListener("click", () => {
        page++;
        searchImages();
    });

    loadCSVData();
});
