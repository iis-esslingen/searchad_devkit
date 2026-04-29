from searchad.create_dummy_predictions import create_dummy_predictions

SEARCHAD_DIR = "/path/to/searchad"  # Update this path to your SearchAD directory
PREDICTIONS_FILE = "/path/to/results_dummy_val.json"  # Update this path to your desired output file
SPLIT = "val"  # Dataset split: "train", "val", or "test"

print(f"Creating dummy predictions for split '{SPLIT}'...")
create_dummy_predictions(
    searchad_dir=SEARCHAD_DIR,
    predictions_file=PREDICTIONS_FILE,
    split=SPLIT,
)
print("Finished creating dummy predictions!")
