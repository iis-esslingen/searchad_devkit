from searchad.create_submission_file import create_submission_file

SEARCHAD_DIR = "/path/to/searchad"  # Update this path to your SearchAD directory
PREDICTIONS_FILE = "/path/to/your/results.json"  # Update this path to your test results file
SUBMISSION_OUTPUT_DIR = "/path/to/submission_output"  # Update this path to your desired submission output directory

# Metadata for the submission
TEAM_NAME = "YOUR_TEAM_NAME"
MODEL_NAME = "YOUR_MODEL_NAME"
PAPER_CODE_AFFILIATION = "YOUR_PAPER_CODE_OR_AFFILIATION"
SEARCH_MODE = "Language"  # Choose from "Language", "Vision", "Multimodal"
SEARCHAD_TRAIN = "No"  # Choose from "Yes", "No"
DEFAULT_QUERIES = "Yes"  # Choose from "Yes", "No"
OVERWRITE = False  # Set to True to overwrite an existing submission.json

print("Starting submission preparation...")
create_submission_file(
    searchad_dir=SEARCHAD_DIR,
    predictions_file=PREDICTIONS_FILE,
    submission_output_dir=SUBMISSION_OUTPUT_DIR,
    team_name=TEAM_NAME,
    model_name=MODEL_NAME,
    paper_code_affiliation=PAPER_CODE_AFFILIATION,
    search_mode=SEARCH_MODE,
    searchad_train=SEARCHAD_TRAIN,
    default_queries=DEFAULT_QUERIES,
    overwrite=OVERWRITE,
)
print("Finished submission preparation!")
