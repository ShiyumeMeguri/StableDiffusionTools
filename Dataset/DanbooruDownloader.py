import os
import sys
import requests
from PIL import Image
from bs4 import BeautifulSoup

def get_image_ids(tag, page):
    url = f'https://danbooru.donmai.us/posts.json?tags={tag}&page={page}&limit=200'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error getting image IDs: {response.status_code}")
        return []
    data = response.json()

    return [str(post['id']) for post in data]
    
def download_image(image_id, save_path):
        
    url = f'https://danbooru.donmai.us/posts/{image_id}.json'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error getting image {image_id}: {response.status_code}")
        return None
    data = response.json()

    # Check if file_url exists
    if 'file_url' not in data:
        print(f'Image {image_id} has no file_url, skipping.')
        return None

    # Get image tags
    tags = data['tag_string'].split(' ')
    
    # Get file extension
    file_ext = data['file_ext']

    # Update save_path with the correct file extension
    save_path = os.path.splitext(save_path)[0] + '.' + file_ext

    # Check if the image file already exists
    if os.path.exists(save_path):
        return "File already exists, skipping"
        
    # If file extension is not jpg or png, skip
    if file_ext not in ['jpg', 'png']:
        return "Skipping other formats"
        
    # If file extension is jpg, check if png version exists
    if file_ext == 'jpg':
        png_path = os.path.splitext(save_path)[0] + '.png'
        if os.path.exists(png_path):
            print(f'PNG version of {image_id} already exists, skipping JPG.')
            return None
            
        
    # Get the image URL
    file_url = data['file_url']
    print(file_url)
    
    # Download the image
    response = requests.get(file_url)
    if response.status_code != 200:
        print(f"Error downloading image {image_id}: {response.status_code}")
        return None

    with open(save_path, 'wb') as f:
        f.write(response.content)

    # Save tags in a text file
    tag_file = os.path.splitext(save_path)[0] + '.txt'
    with open(tag_file, 'w') as f:
        f.write(', '.join(tags))

    # Return tags and file extension as a tuple
    return tags

def download_artist_images(artist_tag, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    page = 1
    while True:
        image_ids = get_image_ids(artist_tag, page)
        if not image_ids:
            break

        for image_id in image_ids:
            save_path = os.path.join(output_dir, f'{image_id}')
            tags = download_image(image_id, save_path)
            print(f'Downloaded {image_id} with {tags}')  # Correct the file extension in the print statement

        page += 1


if __name__ == '__main__':
    artist_tag = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = artist_tag

    download_artist_images(artist_tag, output_dir)

download_artist_images(artist_tag, output_dir)
