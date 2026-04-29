# SearchAD Devkit

Official development toolkit for **[SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](https://iis-esslingen.github.io/searchad/)** — accepted at **CVPR 2026**.

*Felix Embacher, Jonas Uhrig, Marius Cordts, Markus Enzweiler*<br>
*Mercedes-Benz AG &amp; Esslingen University of Applied Sciences*

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2604.08008-b31b1b)](https://arxiv.org/abs/2604.08008)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/iis-esslingen/SearchAD)
[![Benchmark](https://img.shields.io/badge/Benchmark-Leaderboard-blue)](https://huggingface.co/spaces/SearchADBenchmark/SearchADLargeScaleRareImageRetrievalDatasetforAutonomousDriving)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://iis-esslingen.github.io/searchad/)

---

## About SearchAD

Retrieving rare and safety-critical driving scenarios from large-scale datasets is essential for building robust autonomous driving systems. As dataset sizes grow, the key challenge shifts from collecting more data to efficiently identifying the most relevant samples.

**SearchAD** tackles the *needle-in-a-haystack* problem: locating extremely rare classes — some appearing fewer than 50 times across the entire SearchAD dataset with more than 423k frames.

![Overview of SearchAD classes grouped by category](https://iis-esslingen.github.io/searchad/static/images/searchad_class_overview.svg)

### Key Facts

| Property | Value |
|---|---|
| Total frames | 423,798 |
| Bounding box annotations | 513,265+ |
| Rare classes | 90 |
| Additional classes | Animal-Real-Other, Animal-Statue-Other, Human-Duty-Other, Object-Movable-Other, Vehicle-Construction-Other |
| Category groups | 9 (Animal, Human, Marking, Object, Rideable, Scene, Sign, Trailer, Vehicle) |
| Source datasets | 11 established AD datasets |

### Source Datasets

| Dataset | Val. Set | Frames | Original Classes | SearchAD Classes | Annotations | Download Instructions |
|:---|:---:|---:|:---:|:---:|---:|:---|
| [Lost and Found](https://huggingface.co/datasets/iis-esslingen/LostAndFoundDataset) | | 2,239 | 42 | 18 | 2,098 | Download **leftImg8bit/** |
| [WildDash2](https://www.wilddash.cc/accounts/login?next=/download) | ✓ | 5,068 | 26 | 80 | 5,032 | Download **wd_public_v2p0.zip** and **wd_both_02.zip** |
| [ACDC](https://acdc.vision.ee.ethz.ch/download) | ✓ | 8,012 | 19 | 60 | 7,471 | Download **rgb_anon_trainvaltest.zip** |
| [IDD Segmentation](https://idd.insaan.iiit.ac.in/accounts/login/?next=/dataset/download/) | ✓ | 10,003 | 30 | 52 | 12,192 | Download **IDD Segmentation (IDD 20k Part I) (18.5 GB)** |
| [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) | | 14,999 | 8 | 47 | 9,840 | Download **left color images of object data set (12 GB)** |
| [Cityscapes](https://www.cityscapes-dataset.com/downloads/) | ✓ | 24,998 | 30 | 75 | 31,037 | Download **leftImg8bit_trainvaltest.zip** and **leftImg8bit_trainextra.zip** |
| [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) | ✓ | 25,000 | 66 | 86 | 35,093 | Download **mapillary-vistas-dataset_public_v2.0.zip** |
| [ECP](https://eurocity-dataset.tudelft.nl/eval/downloads/detection) | ✓ | 47,335 | 8 | 76 | 33,081 | Download **ECP day and night, train, val, test (6 zip files)** |
| [nuScenes](https://www.nuscenes.org/nuscenes#download) | ✓ | 80,314 | 32 | 56 | 166,152 | Download **Trainval and Test** |
| [BDD100K](http://bdd-data.berkeley.edu/download.html) | ✓ | 100,000 | 12 | 80 | 83,102 | Download **100K Images** |
| [Mapillary Sign](https://www.mapillary.com/dataset/trafficsign) | ✓ | 105,830 | 313 | 90 | 128,167 | Click **Download dataset** |
| **SearchAD (combined)** | | **423,798** | — | **90** | **513,265** | |

### Tasks

SearchAD supports a range of retrieval and perception tasks:

**Retrieval & Search**
- **Text-to-image retrieval** — language-guided search using text queries
- **Image-to-image retrieval** — vision-guided search using image support sets
- **Few-shot learning** — few-shot learning using samples of the SearchAD train dataset
- **Fine-tuning** — fine-tuning VLMs on the SearchAD train dataset

**Beyond Retrieval**
- **Long-tail perception research** — benchmark model robustness on extremely rare real-world classes
- **Open-world and open-vocabulary object detection** — evaluate detectors on rare, unseen, or underrepresented categories
- **Out-of-distribution (OOD) detection** — identify rare or anomalous objects outside typical training distributions
- **Retrieval-driven data curation** — use search to actively mine relevant training samples from large unlabeled datasets
- **Domain adaptation** — leverage the diversity of 11 source datasets for cross-domain generalization research

---

## Installation

Install the devkit via pip directly from the repository:

```bash
pip install .
```

Or install in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10. Dependencies (`torch`, `opencv-python`, `numpy`, `matplotlib`) are installed automatically.

---

## Scripts

All scripts are run from the root of the repository and expect the `searchad/` package to be installed or on the Python path.


### Download Dataset

Downloads the SearchAD annotations and default queries from HuggingFace and prints step-by-step instructions for downloading the 11 source dataset image archives.

```bash
python searchad/download_dataset.py \
    --searchad-dir "/path/to/searchad" \
    --hf-token "hf_xxxxxxxxxxxx"
```

| Argument | Description |
|---|---|
| `--searchad-dir` | Directory where the SearchAD folder will be created |
| `--hf-token` | HuggingFace access token for gated datasets (or set `HF_TOKEN` env variable) |

The script:
- Downloads `searchad.zip` (~11 MB) from [HuggingFace](https://huggingface.co/datasets/iis-esslingen/SearchAD)
- Extracts annotation JSON files (`train`, `val`, test mapping) and the `default_queries/` folder into `--output-dir`
- Prints download instructions for all 11 source datasets

> **Note:** Due to the dataset licenses, the 423k source images must be downloaded manually from the individual dataset hosts (see printed instructions or the table above). Only the annotations and default queries are hosted on HuggingFace.

After downloading and extracting everything, your SearchAD directory should look like this:

```
searchad/
├── ECP/
│   ├── ...
├── IDD_Segmentation/
│   ├── ...
├── acdc/
│   ├── ...
├── bdd100k_images_100k/
│   ├── ...
├── cityscapes/
│   ├── ...
├── kitti/
│   ├── ...
├── lostandfound/
│   ├── ...
├── mapillary_sign/
│   ├── ...
├── mapillary_vistas/
│   ├── ...
├── nuscenes/
│   ├── ...
├── wd_both02/
│   ├── ...
├── wd_publicv2p0/
│   ├── ...
├── searchad_annotations_train.json
├── searchad_annotations_val.json
├── searchad_test_mapping_id_to_imagepath.json
└── default_queries/
    ├── ...
```

---

### Prepare Image-Level Annotations

Derives image-level annotations from bounding-box annotations for a given split.

```bash
python searchad/prepare_image_level_annotations.py \
    --searchad-dir "/path/to/searchad" \
    --split "val"
```

| Argument | Description |
|---|---|
| `--searchad-dir` | Path to the SearchAD directory |
| `--split` | Dataset split to process: `train` or `val` |

---

### Check Dataset Setup

Verify that your SearchAD directory is complete and correctly structured before running any other scripts.

```bash
python searchad/check_searchad_setup.py \
    --searchad-dir "/path/to/searchad"
```

| Argument | Description |
|---|---|
| `--searchad-dir` | Path to the SearchAD directory to check |

The script checks for:
- Required annotation JSON files (`train`, `val`, test mapping)
- Optional generated image-level annotation files
- Presence of all expected SearchAD labels in the annotations
- `default_queries/` folder with one JSON per label
- All subdataset directories and that referenced image paths exist on disk
- Summary of total image counts per split

---

### Prune SearchAD Subdatasets

Removes all non-essential files from specific subdatasets, keeping only the required image data and license files. Since SearchAD only uses front and back camera images, this significantly reduces disk usage for datasets that include additional sensors, annotations, or camera perspectives.

```bash
python searchad/prune_searchad_datasets.py \
    --searchad-dir "/path/to/searchad" \
    --datasets-to-prune \
        "acdc" \
        "bdd100k_images_100k" \
        "cityscapes" \
        "ECP" \
        "IDD_Segmentation" \
        "kitti" \
        "lostandfound" \
        "mapillary_sign" \
        "mapillary_vistas" \
        "nuscenes" \
        "wd_both02" \
        "wd_publicv2p0"
```

| Argument | Description |
|---|---|
| `--searchad-dir` | Path to the SearchAD directory |
| `--datasets-to-prune` | Space-separated list of dataset names to prune (e.g., `acdc mapillary_sign`) |

---

### Create Dummy Predictions

Generate a random baseline predictions file for a given split, useful for verifying the evaluation script (val set) and the submission script (test set) below.

```bash
python searchad/create_dummy_predictions.py \
    --searchad-dir "/path/to/searchad" \
    --predictions-file "/path/to/results_dummy_val.json" \
    --split "val"
```

| Argument | Description |
|---|---|
| `--searchad-dir` | Path to the SearchAD directory |
| `--predictions-file` | Path where the output predictions JSON file will be saved |
| `--split` | Split to generate predictions for: `train`, `val`, or `test` |
| `--seed` | Random seed for reproducibility (default: `42`) |

---

### Evaluate Predictions

Compute metrics (MAP, R-Precision, P@5) for your predictions against the ground-truth annotations.

```bash
python searchad/evaluate.py \
    --predictions-file "/path/to/your/results.json" \
    --split "val" \
    --searchad-dir "/path/to/searchad" \
    --scores-output-dir "/path/to/scores_output"
```

| Argument | Description |
|---|---|
| `--predictions-file` | Path to your predictions JSON file |
| `--split` | Dataset split to evaluate on (`val` or `train`) |
| `--searchad-dir` | Path to the SearchAD directory |
| `--scores-output-dir` | Directory where score outputs will be saved |

The predictions file contains a ranked imagelist (37,325 entries for `val`) for each SearchAD label. Paths are relative to the SearchAD directory and must start with the subdataset name:

```json
{
    "Animal-Real-Cat": [
        "acdc/rgb_anon/train/fog/GOPR0475/GOPR0475_frame_000001_rgb_anon.png",
        "bdd100k_images_100k/images/100k/train/0000f77c-6257be58.jpg",
        "..."
    ],
    "Animal-Real-Cow": ["..."],
    "...",
    "Vehicle-Train": ["..."]
}
```

---

### Create Submission File

Packages your predictions into a `submission.json` file ready for upload to the [SearchAD Benchmark Server](https://huggingface.co/spaces/SearchADBenchmark/SearchADLargeScaleRareImageRetrievalDatasetforAutonomousDriving).

```bash
python searchad/create_submission_file.py \
    --searchad-dir "/path/to/searchad" \
    --predictions-file "/path/to/your/results.json" \
    --submission-output-dir "/path/to/submission_output" \
    --team-name "YOUR_TEAM_NAME" \
    --model-name "YOUR_MODEL_NAME" \
    --paper-code-affiliation "YOUR_PAPER_CODE_OR_AFFILIATION" \
    --search-mode "Language" \
    --searchad-train "No" \
    --default-queries "Yes"
```

| Argument | Description |
|---|---|
| `--searchad-dir` | Path to the SearchAD directory |
| `--predictions-file` | Path to your predictions JSON file |
| `--submission-output-dir` | Directory where `submission.json` will be saved |
| `--team-name` | Name of your team |
| `--model-name` | Name of your model |
| `--paper-code-affiliation` | Link to paper/code or affiliation |
| `--search-mode` | Search mode used: `Language`, `Vision`, or `Multimodal` |
| `--searchad-train` | Whether the SearchAD training set was used: `Yes` or `No` |
| `--default-queries` | Whether default queries were used: `Yes` or `No` |
| `--overwrite` | If set, overwrite an existing `submission.json` instead of raising an error |

> **Note:** Before uploading `submission.json` on the benchmark page, click the blue **"Login with Hugging Face"** button in the bottom left. If the button does not respond, try a different browser — Safari has been observed to silently fail the login, causing an "invalid token" error on submission.

> **Note:** If your submission stays in **PROCESSING** for more than an hour, your HuggingFace authorization token may be missing a required scope. Go to [huggingface.co/settings/connected-applications](https://huggingface.co/settings/connected-applications), revoke the benchmark Space's access, then log in again — you should be prompted to grant read access to private and gated repositories.

---

### Dataset Statistics

Print the label distribution table for the SearchAD dataset, broken down by split optionally per-subdataset using the `--by-subdataset` flag.

```bash
python searchad/print_dataset_statistics.py \
    --searchad-dir "/path/to/searchad" \
    --splits "train" "val" \
    --level "object" \
    --statistics-type "absolute" \
    --output-dir "/path/to/dataset_statistics" \
    --by-subdataset
```

| Argument | Description |
|---|---|
| `--searchad-dir` | Path to the SearchAD directory |
| `--splits` | One or more splits to include: `train`, `val` (default: both) |
| `--level` | `object` counts total bounding-box annotations per label; `image` counts images containing each label (default: `object`) |
| `--statistics-type` | `absolute` for raw counts; `relative` for each split's fraction of the label total (default: `absolute`) |
| `--output-dir` | Optional directory to save the table(s) as `.txt` files |
| `--by-subdataset` | If set, also prints a second table with per-subdataset counts |

---

### Visualize Bounding Boxes

Visualize bounding box annotations for a specific label on sample images from a given split.

```bash
python searchad/visualize_bbox_labels.py \
    --searchad-dir "/path/to/searchad" \
    --output-dir "/path/to/output_bbox_visualizations" \
    --searchad-label "Animal-Real-Cat" \
    --split "train" \
    --num-images 5
```

| Argument | Description |
|---|---|
| `--searchad-dir` | Path to the SearchAD directory |
| `--output-dir` | Directory where visualizations will be saved |
| `--searchad-label` | Label to visualize (e.g., `Animal-Real-Cat`); omit to visualize all labels |
| `--split` | Dataset split: `train` or `val` |
| `--num-images` | Number of images to visualize |
| `--shorten-labels` | If set, display shortened label names instead of full ones |
| `--hide-labels` | If set, draw bounding boxes without any label text |
| `--only-target-label` | If set, only draw boxes for the target label; all other labels in the image are suppressed |

---

### Visualize Retrieval Results

Visualize the top-k retrieved images for each query label, highlighted with green borders for correct and red borders for incorrect retrievals, plus bounding box annotations.

```bash
python searchad/visualize_retrieval.py \
    --predictions-file "/path/to/your/results.json" \
    --searchad-dir "/path/to/searchad" \
    --visualization-output-dir "/path/to/output_retrieval_visualizations" \
    --split "val" \
    --topk 5 \
    --searchad-label "Animal-Real-Cat" \
    --resize-for-collage \
    --shorten-labels
```

| Argument | Description |
|---|---|
| `--predictions-file` | Path to your predictions JSON file |
| `--searchad-dir` | Path to the SearchAD directory |
| `--visualization-output-dir` | Directory where visualized images and collages will be saved |
| `--split` | Dataset split: `train` or `val` |
| `--topk` | Number of top retrieved images to visualize per query (default: `10`) |
| `--searchad-label` | Optional: visualize only this label (e.g., `Animal-Real-Cat`); omit to visualize all |
| `--resize-for-collage` | If set, resize images to a consistent 16:9 resolution before drawing |
| `--shorten-labels` | If set, display shortened label names instead of full ones |

---

### Visualize Support Sets

Generate collage visualizations of the vision support sets for each query.

```bash
python searchad/visualize_support_sets.py \
    --searchad-dir "/path/to/searchad" \
    --output-collage-dir "/path/to/support_set_collage_visu_output" \
    --crop-size 256 \
    --searchad-label "Animal-Real-Cat"
```

| Argument | Description |
|---|---|
| `--searchad-dir` | Path to the SearchAD directory |
| `--output-collage-dir` | Directory where collage images will be saved |
| `--crop-size` | Size (in pixels) to crop each support set image to |
| `--searchad-label` | Optional: visualize only this label's support set (e.g. `Animal-Real-Cat`); omit to visualize all |

---

## Citation

If you use SearchAD in your research, please cite:

```bibtex
@article{embacher2026searchadlargescalerareimage,
    title={SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving},
    author={Felix Embacher and Jonas Uhrig and Marius Cordts and Markus Enzweiler},
    year={2026},
    eprint={2604.08008},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2604.08008},
}
```
