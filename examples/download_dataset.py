from searchad.download_dataset import download_dataset

SEARCHAD_DIR = "/path/to/searchad"  # Update this to the directory where SearchAD should be downloaded
HF_TOKEN = None  # Set to your HuggingFace access token, or leave None to use the HF_TOKEN env variable

print("Starting SearchAD dataset download...")
download_dataset(
    searchad_dir=SEARCHAD_DIR,
    hf_token=HF_TOKEN,
)
print("Finished downloading SearchAD dataset.")
