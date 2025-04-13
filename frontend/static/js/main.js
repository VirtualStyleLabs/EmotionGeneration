document.addEventListener('DOMContentLoaded', () => {
    // Theme switching functionality
    const themeToggle = document.getElementById('themeToggle');
    const html = document.documentElement;

    // Check for saved theme preference, otherwise use system preference
    const savedTheme = localStorage.getItem('theme') ||
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    html.setAttribute('data-theme', savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });

    const uploadBox = document.getElementById('uploadBox');
    const previewBox = document.getElementById('previewBox');
    const previewImage = document.getElementById('previewImage');
    const imageInput = document.getElementById('imageInput');
    const generateBtn = document.getElementById('generateBtn');
    const resultSection = document.getElementById('resultSection');
    const resultImage = document.getElementById('resultImage');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const emotionSelect = document.getElementById('emotionSelect');

    let currentImage = null;

    // Handle drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = getComputedStyle(document.documentElement)
            .getPropertyValue('--primary-color');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = getComputedStyle(document.documentElement)
            .getPropertyValue('--border-color');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = getComputedStyle(document.documentElement)
            .getPropertyValue('--border-color');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageUpload(file);
        }
    });

    // Handle click to upload
    uploadBox.addEventListener('click', () => {
        imageInput.click();
    });

    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageUpload(file);
        }
    });

    // Handle image upload
    function handleImageUpload(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImage = e.target.result;
            previewImage.src = currentImage;
            uploadBox.style.display = 'none';
            previewBox.style.display = 'block';
            generateBtn.disabled = false;
            resultSection.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // Handle generate button click
    generateBtn.addEventListener('click', async () => {
        if (!currentImage) return;

        // Show loading spinner
        loadingSpinner.style.display = 'block';
        resultImage.style.display = 'none';

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: currentImage,
                    emotion: parseInt(emotionSelect.value)
                })
            });

            const data = await response.json();

            if (response.ok) {
                resultImage.src = data.image;
                resultImage.style.display = 'block';
                resultSection.style.display = 'block';
            } else {
                alert(data.error || 'An error occurred while generating the image.');
            }
        } catch (error) {
            alert('An error occurred while communicating with the server.');
            console.error('Error:', error);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });

    // Handle emotion select change
    emotionSelect.addEventListener('change', () => {
        if (currentImage) {
            resultSection.style.display = 'none';
        }
    });
}); 