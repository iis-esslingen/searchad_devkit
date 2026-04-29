import argparse
import os

import cv2
import numpy as np

from searchad.utils.io import load_query_files


def _extract_vision_support_set_candidates(query_data: dict) -> list[dict[str, str]]:
    bbox_candidates: list[dict[str, str]] = []
    vision_support_sets = query_data.get("supportsets", {}).get("vision", {})

    for image_path_relative, annotations in vision_support_sets.items():
        for annotation in annotations:
            bbox_data = annotation.get("bbox")
            if not bbox_data:
                print(f"Warning: Annotation missing 'bbox' data for image {image_path_relative}. Skipping.")
                continue
            bbox_candidates.append(
                {
                    "image_path_relative": image_path_relative,
                    "bbox_data": bbox_data,
                }
            )
    return bbox_candidates


def _create_collage(
    bbox_candidates: list[dict[str, str]],
    query_filename_base: str,
    output_dir: str,
    base_path: str,
    crop_size: tuple[int, int],
) -> str | None:
    print(f"\nCreating support set collage for query: '{query_filename_base}'...")
    cropped_images = []

    for idx, candidate in enumerate(bbox_candidates):
        image_path_relative = candidate["image_path_relative"]
        bbox_data = candidate["bbox_data"]
        image_full_path = os.path.join(base_path, image_path_relative)

        x_min, y_min, x_max, y_max = map(
            int, [bbox_data["min_x"], bbox_data["min_y"], bbox_data["max_x"], bbox_data["max_y"]]
        )

        img = cv2.imread(image_full_path)
        if img is None:
            print(f"  Error: Could not read image at {image_full_path}. Skipping candidate {idx}.")
            continue

        cropped_img = img[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0:
            print(
                f"  Warning: Cropped image is empty at {image_full_path} "
                f"(bbox: {x_min, y_min, x_max, y_max}). Skipping candidate {idx}."
            )
            continue

        resized_crop = cv2.resize(cropped_img, crop_size, interpolation=cv2.INTER_AREA)

        number_to_display = idx + 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        text_color = (0, 0, 255)
        text_size = cv2.getTextSize(str(number_to_display), font, font_scale, font_thickness)[0]
        cv2.putText(
            resized_crop,
            str(number_to_display),
            (5, text_size[1] + 5),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )
        cropped_images.append(resized_crop)

    if not cropped_images:
        print(f"  No valid cropped images found for query '{query_filename_base}'. Skipping.")
        return None

    num_crops = len(cropped_images)
    print(f"  {num_crops} crops collected.")

    rows = int(np.ceil(np.sqrt(num_crops)))
    cols = int(np.ceil(num_crops / rows))
    if rows * cols < num_crops:
        cols += 1

    collage_width = cols * crop_size[0]
    collage_height = rows * crop_size[1]
    collage_image = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)

    for idx, crop in enumerate(cropped_images):
        row_idx = idx // cols
        col_idx = idx % cols
        start_y = row_idx * crop_size[1]
        end_y = start_y + crop_size[1]
        start_x = col_idx * crop_size[0]
        end_x = start_x + crop_size[0]
        if end_y <= collage_height and end_x <= collage_width:
            collage_image[start_y:end_y, start_x:end_x] = crop
        else:
            print(f"  Warning: Crop {idx} (row {row_idx}, col {col_idx}) exceeds collage dimensions. Skipping.")

    collage_filename = f"{query_filename_base}_support_set_collage.jpg"
    collage_path = os.path.join(output_dir, collage_filename)
    cv2.imwrite(collage_path, collage_image)
    print(f"  Saved collage to: {collage_path}")
    return collage_path


def visualize_support_sets(
    searchad_dir: str | os.PathLike,
    output_collage_dir: str | os.PathLike,
    crop_size: tuple[int, int] = (256, 256),
    searchad_label: str | None = None,
) -> None:
    """
    Visualizes vision support sets for all query JSON files in the given directory by
    creating collage images of the cropped bounding boxes.

    Args:
        searchad_dir (str): Path to the SearchAD directory. Query JSON files are read
                            from <searchad_dir>/default_queries/ and image paths in the JSONs
                            are resolved relative to this directory.
        output_collage_dir (str): Path to the directory where collage images will be saved.
        crop_size (tuple): Width and height in pixels to resize each crop to. Default: (256, 256).
        searchad_label (str | None): If set, only visualize the support set for this label.
    """
    queries_dir = os.path.join(searchad_dir, "default_queries")

    print("--- SearchAD Support Set Visualization ---")
    print(f"SearchAD directory: {searchad_dir}")
    print(f"Queries directory:  {queries_dir}")
    print(f"Output directory:   {output_collage_dir}")
    print(f"Crop size:          {crop_size[0]}x{crop_size[1]}")
    if searchad_label:
        print(f"Label filter:       {searchad_label}")

    os.makedirs(output_collage_dir, exist_ok=True)

    all_queries = load_query_files(queries_dir)

    if not all_queries:
        print("No query files found or loaded. Exiting.")
        return

    collages_created = 0
    for query_filename_base, query_data in all_queries.items():
        if searchad_label and query_filename_base != searchad_label:
            continue
        bbox_candidates = _extract_vision_support_set_candidates(query_data)
        if bbox_candidates:
            result = _create_collage(
                bbox_candidates=bbox_candidates,
                query_filename_base=query_filename_base,
                output_dir=output_collage_dir,
                base_path=searchad_dir,
                crop_size=crop_size,
            )
            if result is not None:
                collages_created += 1
        else:
            print(f"No vision support sets found for query '{query_filename_base}'. Skipping.")

    print(f"\nVisualization complete. {collages_created} collage(s) saved to: {output_collage_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize SearchAD vision support sets by creating collage images of cropped bounding boxes."
    )
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="Path to the SearchAD directory. Queries are read from <searchad-dir>/default_queries/",
    )
    parser.add_argument(
        "--output-collage-dir",
        type=str,
        required=True,
        help="Path to the directory where collage images will be saved.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Size (width and height) in pixels to resize each crop to. Default: 256.",
    )
    parser.add_argument(
        "--searchad-label",
        type=str,
        default=None,
        help="Optional: visualize only this label's support set (e.g. 'Animal-Real-Cat'). "
        "If omitted, all labels are visualized.",
    )

    args = parser.parse_args()

    visualize_support_sets(
        searchad_dir=args.searchad_dir,
        output_collage_dir=args.output_collage_dir,
        crop_size=(args.crop_size, args.crop_size),
        searchad_label=args.searchad_label,
    )
