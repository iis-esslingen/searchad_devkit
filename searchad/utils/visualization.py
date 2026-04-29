import cv2
import numpy as np


def save_placeholder(
    output_path: str,
    size: tuple[int, int],
) -> None:
    """Save a grey placeholder image indicating the source image was not found on disk."""
    w, h = size
    img = 80 * np.ones((h, w, 3), dtype="uint8")  # dark-grey background
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = ["Image not found on disk"]
    font_scale = min(w, h) / 400
    thickness = max(1, int(font_scale * 2))
    line_gap = int(min(w, h) * 0.06)
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    total_h = sum(s[1] for s in sizes) + line_gap * (len(lines) - 1)
    y = (h - total_h) // 2
    for line, (tw, th) in zip(lines, sizes, strict=False):
        x = (w - tw) // 2
        y += th
        cv2.putText(img, line, (x, y), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)
        y += line_gap
    cv2.imwrite(output_path, img)


def shorten_label(label: str) -> str:
    """
    Shortens a SearchAD label name for display.

    Rules:
    - For non-Vehicle labels, remove the first segment
      (e.g. "Animal-Real-Cat" -> "Real-Cat").
    - For Vehicle labels, remove the second segment but keep "Vehicle"
      (e.g. "Vehicle-Duty-Fire" -> "Vehicle-Fire",
       "Vehicle-Construction-Tractor" -> "Vehicle-Tractor").
    """
    parts = label.split("-")
    if parts[0] == "Vehicle":
        return "-".join([parts[0]] + parts[2:]) if len(parts) > 2 else label
    return "-".join(parts[1:]) if len(parts) > 1 else label


def annotation_color_bgr(
    label: str,
    label_colors: dict,
    default_color_rgb: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Return the BGR draw colour for *label* (falls back to *default_color_rgb*)."""
    r, g, b = label_colors.get(label, default_color_rgb)
    return (b, g, r)


def clamp_label_position(
    text_x: int,
    text_y: int,
    text_w: int,
    text_h: int,
    img_w: int,
    img_h: int,
    padding: int,
) -> tuple[int, int]:
    """
    Shifts the text anchor point so the label background rectangle stays
    fully within all four image borders.

    The background rectangle occupies:
        x: [text_x - padding,  text_x + text_w + padding]
        y: [text_y - text_h - padding,  text_y + padding]
    """
    if text_x + text_w + padding > img_w:
        text_x = img_w - text_w - padding
    if text_x - padding < 0:
        text_x = padding
    if text_y - text_h - padding < 0:
        text_y = text_h + padding
    if text_y + padding > img_h:
        text_y = img_h - padding
    return text_x, text_y


def resolve_label_overlaps(
    label_positions: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    padding: int,
    max_attempts: int = 30,
) -> list[tuple[int, int]]:
    """
    Given initial label anchor positions [(tx, ty, tw, th), ...], iteratively
    shifts labels that overlap already-placed labels, staying within image borders.

    Each label background rectangle occupies:
        x: [tx - padding,  tx + tw + padding]
        y: [ty - th - padding,  ty + padding]

    Returns:
        List of resolved (tx, ty) anchor points in the same order as input.
    """

    def to_rect(tx: int, ty: int, tw: int, th: int) -> tuple[int, int, int, int]:
        return (tx - padding, ty - th - padding, tx + tw + padding, ty + padding)

    def overlaps(r1: tuple, r2: tuple) -> bool:
        return not (r1[2] <= r2[0] or r2[2] <= r1[0] or r1[3] <= r2[1] or r2[3] <= r1[1])

    def overlap_shift(r1: tuple, r2: tuple) -> tuple[int, int]:
        dx = min(r1[2] - r2[0], r2[2] - r1[0]) + 1
        dy = min(r1[3] - r2[1], r2[3] - r1[1]) + 1
        return dx, dy

    placed: list[tuple[int, int, int, int]] = []
    resolved: list[tuple[int, int]] = []

    for tx, ty, tw, th in label_positions:
        best_tx, best_ty = tx, ty

        for _ in range(max_attempts):
            current_rect = to_rect(best_tx, best_ty, tw, th)
            conflicts = [r for r in placed if overlaps(current_rect, r)]

            if not conflicts:
                break

            conf = conflicts[0]
            dx, dy = overlap_shift(current_rect, conf)
            candidates = [
                (best_tx, best_ty + dy),
                (best_tx, best_ty - dy),
                (best_tx + dx, best_ty),
                (best_tx - dx, best_ty),
            ]

            improved = False
            for ctx, cty in candidates:
                ctx, cty = clamp_label_position(ctx, cty, tw, th, img_w, img_h, padding)
                r = to_rect(ctx, cty, tw, th)
                remaining = sum(1 for pr in placed if overlaps(r, pr))
                if remaining < len(conflicts):
                    best_tx, best_ty = ctx, cty
                    improved = True
                    break

            if not improved:
                break

        best_tx, best_ty = clamp_label_position(best_tx, best_ty, tw, th, img_w, img_h, padding)
        placed.append(to_rect(best_tx, best_ty, tw, th))
        resolved.append((best_tx, best_ty))

    return resolved


def draw_annotations(
    img: np.ndarray,
    annotations: list[dict],
    bbox_line_width: int,
    font_scale: float,
    font_thickness: int,
    text_padding: int,
    shorten_labels: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Draws bounding boxes and labels onto *img* using a two-pass approach:
      Pass 1 — collect annotations, shorten labels, compute text sizes,
               resolve overlapping label positions.
      Pass 2a — draw all bounding box rectangles.
      Pass 2b — draw all labels (in label colour, no background box).

    Args:
        img: BGR image array (will be modified in-place and returned).
        annotations: List of dicts, each containing:
            "bbox"      — [x1, y1, x2, y2] in image pixel coords (already scaled/offset)
            "label"     — full SearchAD label name (will be shortened for display)
            "color_bgr" — (B, G, R) tuple for this annotation

    Returns:
        The annotated image array.
    """
    img_h, img_w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Pass 1: build valid entries with resolved label positions ─────────────
    valid = []
    for ann in annotations:
        bbox = ann.get("bbox")
        label = ann.get("label")
        color_bgr = ann.get("color_bgr")
        if not (bbox and label and color_bgr is not None):
            continue

        x1, y1, x2, y2 = map(int, bbox)
        display_label = shorten_label(label) if shorten_labels else label
        tw, th = cv2.getTextSize(display_label, font, font_scale, font_thickness)[0]

        init_tx, init_ty = clamp_label_position(x1, y1 - text_padding - 4, tw, th, img_w, img_h, text_padding)

        valid.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": display_label,
                "color": color_bgr,
                "tw": tw,
                "th": th,
                "init_tx": init_tx,
                "init_ty": init_ty,
            }
        )

    if not valid:
        return img

    initial_positions = [(v["init_tx"], v["init_ty"], v["tw"], v["th"]) for v in valid]
    resolved_positions = resolve_label_overlaps(initial_positions, img_w, img_h, text_padding)

    # ── Pass 2a: draw bounding boxes ──────────────────────────────────────────
    for v in valid:
        cv2.rectangle(img, (v["x1"], v["y1"]), (v["x2"], v["y2"]), v["color"], bbox_line_width)

    # ── Pass 2b: draw labels ──────────────────────────────────────────────────
    if show_labels:
        for v, (text_x, text_y) in zip(valid, resolved_positions, strict=False):
            cv2.putText(
                img,
                v["label"],
                (text_x, text_y),
                font,
                font_scale,
                v["color"],
                font_thickness,
                cv2.LINE_AA,
            )

    return img


def draw_image_with_annotations(
    image_full_path: str,
    object_annotations_for_image: list[dict],
    output_path: str,
    label_colors: dict,
    default_color_rgb: tuple[int, int, int],
    bbox_line_width: int,
    font_scale: float,
    font_thickness: int,
    text_padding: int,
    is_correct_retrieval: bool = False,
    border_width: int = 0,
    resize_for_drawing: bool = False,
    target_resize_dim: tuple[int, int] | None = None,
    shorten_labels: bool = True,
    show_labels: bool = True,
) -> None:
    """Loads an image, optionally resizes it, draws a green/red correctness border,
    overlays all bounding-box annotations and saves the result to *output_path*.

    Args:
        image_full_path: Absolute path to the source image.
        object_annotations_for_image: List of dicts with "bbox" and "label" keys.
        output_path: Where to write the annotated image.
        label_colors: Mapping of label name to (R, G, B) colour tuple.
        default_color_rgb: Fallback (R, G, B) colour when a label has no entry in label_colors.
        bbox_line_width: Thickness of bounding-box rectangles.
        font_scale: OpenCV font scale for label text.
        font_thickness: OpenCV font thickness for label text.
        text_padding: Padding around label text.
        is_correct_retrieval: True → green border, False → red border.
        border_width: Pixel width of the correctness border (0 = no border).
        resize_for_drawing: If True, resize the image to *target_resize_dim* before drawing.
        target_resize_dim: (width, height) to resize to when *resize_for_drawing* is True.
        shorten_labels: Whether to shorten label names for display.
    """
    img = cv2.imread(image_full_path)
    if img is None:
        print(f"Error: Could not read image at {image_full_path}. Skipping visualization for this image.")
        return

    original_h, original_w = img.shape[:2]
    scale_x, scale_y = 1.0, 1.0

    if resize_for_drawing:
        scale_x = target_resize_dim[0] / original_w
        scale_y = target_resize_dim[1] / original_h
        img = cv2.resize(img, target_resize_dim, interpolation=cv2.INTER_AREA)

    border_color_rgb = (0, 255, 0) if is_correct_retrieval else (255, 0, 0)
    border_color_bgr = (border_color_rgb[2], border_color_rgb[1], border_color_rgb[0])

    img_with_border = cv2.copyMakeBorder(
        img,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_CONSTANT,
        value=border_color_bgr,
    )

    annotated_img = img_with_border.copy()

    prepared = []
    for annotation in object_annotations_for_image:
        bbox_orig = annotation.get("bbox")
        label = annotation.get("label")
        if not (bbox_orig and label):
            continue

        x1 = int(bbox_orig[0] * scale_x) + border_width
        y1 = int(bbox_orig[1] * scale_y) + border_width
        x2 = int(bbox_orig[2] * scale_x) + border_width
        y2 = int(bbox_orig[3] * scale_y) + border_width

        prepared.append(
            {
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "color_bgr": annotation_color_bgr(label, label_colors, default_color_rgb),
            }
        )

    draw_annotations(
        annotated_img,
        prepared,
        bbox_line_width,
        font_scale,
        font_thickness,
        text_padding,
        shorten_labels=shorten_labels,
        show_labels=show_labels,
    )
    cv2.imwrite(output_path, annotated_img)
