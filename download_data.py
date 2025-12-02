"""
Download BDG2 dataset files from GitHub
"""
import os
import requests
from pathlib import Path

# Create data directory
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# BDG2 GitHub raw data URLs
base_url = "https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/master/data"

files_to_download = {
    'metadata.csv': f'{base_url}/metadata.csv',
    'weather.csv': f'{base_url}/weather.csv',
    # Note: meter data files are large and may be split by site
    # We'll download a sample or use a smaller subset
}

print("Downloading BDG2 dataset files...")
for filename, url in files_to_download.items():
    filepath = data_dir / filename
    if not filepath.exists():
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                filepath.write_bytes(response.content)
                print(f"  ✓ Downloaded {filename} ({len(response.content)/1024/1024:.2f} MB)")
            else:
                print(f"  ✗ Failed to download {filename}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
    else:
        print(f"  ⊙ {filename} already exists")

print("\nNote: Meter data files are large. For this analysis, we'll generate")
print("synthetic data based on BDG2 structure, or download specific site files.")
print("The BDG2 repository contains meter data split by site (e.g., site_0.csv).")
