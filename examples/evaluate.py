from searchad.evaluate import evaluate

PREDICTIONS_FILE = "/path/to/your/results.json"  # Update this path to your validation results file
SEARCHAD_DIR = "/path/to/searchad"  # Update this path to your SearchAD directory
SCORES_OUTPUT_DIR = "/path/to/scores_output"  # Update this path to your desired scores output directory

print("Starting evaluation for validation split...")
evaluate(
    predictions_file=PREDICTIONS_FILE,
    split="val",
    searchad_dir=SEARCHAD_DIR,
    scores_output_dir=SCORES_OUTPUT_DIR,
)
print("Finished evaluation for validation split!")
