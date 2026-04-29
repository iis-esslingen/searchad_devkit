from searchad.visualize_retrieval import visualize_retrieval

PREDICTIONS_FILE = "/path/to/your/results.json"  # Update this path to your results file
SEARCHAD_DIR = "/path/to/searchad"  # Update this path to your SearchAD directory
OUTPUT_DIR = "/path/to/output_retrieval_visualizations"  # Update this path to your desired output directory
SPLIT = "val"  # Update this to "train" or "val"
TOP_K = 5  # Update this to the number of top retrievals to visualize
SEARCHAD_LABEL = None  # Update this to the label you want to visualize, None visualizes all labels
RESIZE_FOR_COLLAGE = True  # Set to True to resize images for collage output
SHORTEN_LABELS = True  # Set to True to shorten label names for display

print("Starting retrieval visualization...")
visualize_retrieval(
    predictions_file=PREDICTIONS_FILE,
    searchad_dir=SEARCHAD_DIR,
    visualization_output_dir=OUTPUT_DIR,
    split=SPLIT,
    topk=TOP_K,
    searchad_label=SEARCHAD_LABEL,
    resize_for_collage=RESIZE_FOR_COLLAGE,
    shorten_labels=SHORTEN_LABELS,
)
print("Finished retrieval visualization!")
