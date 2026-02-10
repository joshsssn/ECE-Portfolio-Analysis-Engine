import os
import requests
from tqdm import tqdm

def download_file(url, destination):
    """
    Downloads a file with a progress bar.
    """
    print(f"Downloading {url} to {destination}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(destination, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    
    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong during download.")
    else:
        print("Download complete.")

if __name__ == "__main__":
    # Target directory
    target_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(target_dir, "v1.pth")
    
    # Direct download URL for HuggingFace (resolve instead of blob)
    model_url = "https://huggingface.co/Vincent05R/FinCast/resolve/main/v1.pth?download=true"
    
    if os.path.exists(target_file):
        print(f"Model file already exists at {target_file}")
        choice = input("Overwrite? (y/n): ").lower()
        if choice != 'y':
            print("Download cancelled.")
            exit()
            
    download_file(model_url, target_file)
