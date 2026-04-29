from searchad.prune_searchad_datasets import prune_searchad_datasets

SEARCHAD_DIR = "/path/to/searchad"  # Update this path to your SearchAD directory
DATASETS_TO_PRUNE = [  # Update this list with the datasets you want to prune
    "acdc",
    "bdd100k_images_100k",
    "cityscapes",
    "ECP",
    "IDD_Segmentation",
    "kitti",
    "lostandfound",
    "mapillary_sign",
    "mapillary_vistas",
    "nuscenes",
    "wd_both02",
    "wd_publicv2p0",
]

print("Starting SearchAD dataset pruning...")
prune_searchad_datasets(
    searchad_dir=SEARCHAD_DIR,
    datasets_to_prune=DATASETS_TO_PRUNE,
)
print("Finished SearchAD dataset pruning!")
