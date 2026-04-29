from searchad.check_searchad_setup import check_searchad_setup

SEARCHAD_DIR = "/path/to/searchad"  # Update this path to your SearchAD directory

print("Starting SearchAD setup check...")
check_searchad_setup(searchad_dir=SEARCHAD_DIR)
print("Finished SearchAD setup check!")
