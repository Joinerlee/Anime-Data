import csv
import os
import requests
import re
import time

# Create posters directory if it doesn't exist
if not os.path.exists('posters'):
    os.makedirs('posters')

# Function to sanitize filenames
def sanitize_filename(filename):
    if not filename:
        return "unknown"
    return re.sub(r'[\\/*?:",<>|]', "", filename)

# Open log file
with open('download_log.txt', 'w', encoding='utf-8') as log_file:
    # Read CSV
    with open('C:\\Users\\NSMUSER\\PycharmProjects\\Anime-Data\\anilife_data_20250915_214030.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            korean_title = row.get('title_korean')
            english_title = row.get('title_english')
            
            search_title = english_title if english_title else korean_title
            
            if search_title:
                try:
                    print(f"Searching for '{search_title}'...")
                    # Search for the anime on Jikan API
                    response = requests.get(f"https://api.jikan.moe/v4/anime?q={search_title}&limit=1")
                    response.raise_for_status()
                    data = response.json()

                    if data['data']:
                        anime_data = data['data'][0]
                        image_url = anime_data.get('images', {}).get('jpg', {}).get('large_image_url')
                        
                        if image_url:
                            # Sanitize title for filename
                            sanitized_title = sanitize_filename(korean_title if korean_title else english_title)
                            
                            # Get file extension
                            file_extension = os.path.splitext(image_url)[1].split('?')[0]
                            if not file_extension:
                                file_extension = '.jpg'

                            filename = os.path.join('posters', f"{sanitized_title}{file_extension}")

                            # Download and save image
                            image_response = requests.get(image_url, stream=True)
                            image_response.raise_for_status()
                            with open(filename, 'wb') as img_file:
                                for chunk in image_response.iter_content(chunk_size=8192):
                                    img_file.write(chunk)
                            
                            log_message = f"Downloaded: {korean_title} -> {filename}\n"
                            print(log_message)
                            log_file.write(log_message)
                        else:
                            log_message = f"Image URL not found for {korean_title} in Jikan API response.\n"
                            print(log_message)
                            log_file.write(log_message)
                    else:
                        log_message = f"No results found for {search_title} on Jikan API.\n"
                        print(log_message)
                        log_file.write(log_message)

                except requests.exceptions.RequestException as e:
                    log_message = f"Error searching for {search_title}: {e}\n"
                    print(log_message)
                    log_file.write(log_message)
                except Exception as e:
                    log_message = f"An error occurred for title '{search_title}': {e}\n"
                    print(log_message)
                    log_file.write(log_message)
                
                # Add a delay to avoid hitting API rate limits
                time.sleep(1)
