import os
import gdown
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

# Create directories
os.makedirs("./data/longmemeval", exist_ok=True)
os.makedirs("./data/investigathon", exist_ok=True)

print("=" * 80)
print("Downloading LongMemEval datasets from HuggingFace...")
print("=" * 80)

for filename in [
    "longmemeval_oracle.json",
    "longmemeval_s_cleaned.json"
]:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename=filename,
        local_dir="./data/longmemeval",
        local_dir_use_symlinks=False,
        repo_type="dataset",
    )
    print(f"✓ Downloaded {filename}")

print("\n" + "=" * 80)
print("Downloading Investigathon datasets from Google Drive...")
print("=" * 80)

# Google Drive folder ID (extracted from the URL)
folder_id = "1i6LHdv-sY_AryNTINHv6t4f4mTRjQP3x"

try:
    # Download entire folder from Google Drive
    print(f"Downloading folder contents...")
    gdown.download_folder(
        id=folder_id,
        output="./data/investigathon",
        quiet=False,
        use_cookies=False
    )
    print("✓ Downloaded Investigathon datasets")
except Exception as e:
    print(f"Error downloading from Google Drive: {e}")
    print("\nAlternative: You can manually download the files from:")
    print("https://drive.google.com/drive/folders/1i6LHdv-sY_AryNTINHv6t4f4mTRjQP3x")
    print("And place them in the ./data/investigathon/ folder")

print("\n" + "=" * 80)
print("Download complete!")
print("=" * 80)
