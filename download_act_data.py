
import os
import requests
import tarfile

url = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/ACT_dr6_likelihood_v1.2.tgz"
dest_dir = os.path.join("data", "external", "ACT_DR6", "act_dr6_lenslike", "data")
dest_file = os.path.join(dest_dir, "ACT_dr6_likelihood_v1.2.tgz")

os.makedirs(dest_dir, exist_ok=True)

print(f"Downloading {url}...")
response = requests.get(url, stream=True)
response.raise_for_status()

with open(dest_file, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("Download complete. Extracting...")
with tarfile.open(dest_file, "r:gz") as tar:
    tar.extractall(path=dest_dir)

print("Extraction complete.")
