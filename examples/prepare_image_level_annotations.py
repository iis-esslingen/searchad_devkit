from searchad.prepare_image_level_annotations import prepare_image_level_annotations

SEARCHAD_DIR = "/path/to/searchad"  # Update this path to your SearchAD directory

# Prepare annotations for the 'val' split. You can also prepare for 'train' if needed by changing the split argument.
print("Starting image-level annotation preparation...")
prepare_image_level_annotations(searchad_dir=SEARCHAD_DIR, split="val")

print("Finished image-level annotation preparation!")
