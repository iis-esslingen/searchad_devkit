from searchad.visualize_support_sets import visualize_support_sets

SEARCHAD_DIR = "/path/to/searchad"  # Update this path to your SearchAD directory
OUTPUT_COLLAGE_DIR = "/path/to/support_set_collage_visu_output"  # Update this path to your desired output directory
CROP_SIZE = 256  # Width and height in pixels for resizing each cropped image in the collage
SEARCHAD_LABEL = None  # Update this to the label you want to visualize, None visualizes all labels

print("Starting support set visualization...")
visualize_support_sets(
    searchad_dir=SEARCHAD_DIR,
    output_collage_dir=OUTPUT_COLLAGE_DIR,
    crop_size=(CROP_SIZE, CROP_SIZE),
    searchad_label=SEARCHAD_LABEL,
)
print("Finished support set visualization!")
