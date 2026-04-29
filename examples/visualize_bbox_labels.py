from searchad.visualize_bbox_labels import visualize_bbox_labels

SEARCHAD_DIR = "/path/to/searchad"  # Update this path to your SearchAD directory
OUTPUT_DIR = "/path/to/output_bbox_visualizations"  # Update this path to your desired output directory
SEARCHAD_LABEL = None  # Update this to the label you want to visualize, None visualizes all labels
SPLIT = "train"  # Update this to "train" or "val"
NUM_IMAGES = 5  # Update this to the number of images to randomly select and visualize
SHORTEN_LABELS = True  # Set to True to shorten label names for display

print(f"Starting bounding box visualization for label '{SEARCHAD_LABEL}' (random {NUM_IMAGES} images)...")
visualize_bbox_labels(
    searchad_dir=SEARCHAD_DIR,
    output_dir=OUTPUT_DIR,
    searchad_label=SEARCHAD_LABEL,
    split=SPLIT,
    num_images=NUM_IMAGES,
    shorten_labels=SHORTEN_LABELS,
)
print(f"Finished bounding box visualization for label '{SEARCHAD_LABEL}'!")
