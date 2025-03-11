import os
import time
import requests
import shutil
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from urllib.parse import quote

# List of supported emotions
emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Base directory for storing images
BASE_DIR = "./dataset/Train"

# Ensure all emotion directories exist
for emotion in emotions:
    path = os.path.join(BASE_DIR, emotion)
    os.makedirs(path, exist_ok=True)

# Headers for downloading images
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def setup_driver():
    """Set up headless Chrome driver"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def download_image(url, save_path, retries=3):
    """Download an image from URL with retry logic"""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, stream=True, timeout=10)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return False

def get_image_urls(driver, query, num_images=50):
    """Scrape actual image URLs from Google Images using Selenium"""
    image_urls = set()  # Use set to avoid duplicates
    url = f"https://www.google.com/search?q={quote(query)}&tbm=isch"
    
    try:
        driver.get(url)
        time.sleep(2)  # Wait for initial load

        # Scroll to load more images
        last_height = driver.execute_script("return document.body.scrollHeight")
        while len(image_urls) < num_images:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for images to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:  # No more images loaded
                break
            last_height = new_height

            # Find all image elements
            images = driver.find_elements(By.CLASS_NAME, "YQ4gaf") # Change class name from your browser usage
            for img in images:
                src = img.get_attribute("src") or img.get_attribute("data-src")
                if src and "google" not in src.lower() and "base64" not in src:
                    image_urls.add(src)
                if len(image_urls) >= num_images:
                    break

    except Exception as e:
        print(f"Error scraping {query}: {e}")

    print(f"Found {len(image_urls)} URLs for {query}")
    return list(image_urls)[:num_images]

def main():
    # Set up Selenium driver
    driver = setup_driver()
    
    try:
        for emotion in emotions:
            print(f"Processing {emotion}...")
            save_dir = os.path.join(BASE_DIR, emotion)
            
            # Get image URLs
            query = f"{emotion.lower()} emotion face human"
            image_urls = get_image_urls(driver, query, num_images=50)
            
            if not image_urls:
                print(f"No valid image URLs found for {emotion}")
                continue

            # Download images
            for i, url in enumerate(image_urls):
                try:
                    filename = f"{emotion.lower()}_{i:03d}.jpg"
                    save_path = os.path.join(save_dir, filename)
                    
                    if os.path.exists(save_path):
                        continue

                    success = download_image(url, save_path)
                    if success:
                        print(f"Downloaded {filename}")
                    else:
                        print(f"Failed to download {url}")
                    
                    time.sleep(random.uniform(0.5, 1.5))

                except Exception as e:
                    print(f"Error processing {url}: {e}")
                    continue

            print(f"Completed {emotion}: {len(os.listdir(save_dir))} images downloaded")

    finally:
        driver.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")