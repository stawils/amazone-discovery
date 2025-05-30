import os
import gdown

def download_from_drive(file_id, output_path):
    """
    Download a file from Google Drive using gdown.
    Args:
        file_id (str): The Google Drive file ID.
        output_path (str): Local path to save the file.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdown.download(url, output_path, quiet=False) 