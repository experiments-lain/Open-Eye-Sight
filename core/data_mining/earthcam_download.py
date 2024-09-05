# https://video2archives.earthcam.com/archives/_definst_/MP4:network/25225/2024/09/02/1101.mp4/media_w2053893251_15.ts
import requests
from tqdm import tqdm

def download_video(url, output_file):
    """
    Download a video from the given URL and save it to the specified output file.
    
    Args:
        url (str): The URL of the video to be downloaded.
        output_file (str): The path and filename to save the downloaded video.
    """
    with open(output_file, 'wb') as file:
        for fragment in range(1, 280):
            response = requests.get(url + str(fragment) + ".ts", stream=True)
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)
    
    progress_bar.close()
    
    print(f"Video downloaded and saved to '{output_file}'")

# Example usage
video_url = "https://video2archives.earthcam.com/archives/_definst_/MP4:network/25225/2024/09/02/1101.mp4/media_w2053893251_"
output_filename = "video.ts"
download_video(video_url, output_filename)