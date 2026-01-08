import os
from huggingface_hub import hf_hub_download
import shutil

# Configuration
REPO_ID = "ayushirathour/chest-xray-pneumonia-detection"
FILENAME = "best_chest_xray_model.h5"
LOCAL_SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')

def download_model():
    if not os.path.exists(LOCAL_SAVED_MODEL_DIR):
        os.makedirs(LOCAL_SAVED_MODEL_DIR)
        print(f"Created directory: {LOCAL_SAVED_MODEL_DIR}")

    print(f"Downloading {FILENAME} from {REPO_ID}...")
    try:
        # Download file from HF Hub
        # This caches it in ~/.cache/huggingface/hub and returns the path
        cached_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        print(f"Downloaded to cache: {cached_path}")

        # Move/Copy to our desired location
        target_path = os.path.join(LOCAL_SAVED_MODEL_DIR, FILENAME)
        shutil.copy(cached_path, target_path)
        print(f"Model successfully saved to: {target_path}")

    except Exception as e:
        print(f"Failed to download model: {e}")

if __name__ == "__main__":
    download_model()
