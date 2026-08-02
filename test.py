"""Unified evaluation and prediction entry point for MS-DM."""

import argparse
import csv
import inspect
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.crowd_test import Crowd_qnrf
from models import vgg19


SPECIES = {
    "wf": {"label": "whitefly", "class_id": 1, "color": (0, 0, 255)},
    "ff": {"label": "fruit-fly", "class_id": 2, "color": (0, 255, 0)},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and predict with MS-DM")
    parser.add_argument(
        "--model-path",
        default=r"pretrained_models\msdm_final_v3_legacy.pth",
        help="model checkpoint (.pth or .tar)",
    )
    parser.add_argument(
        "--data-path",
        default=r"data\whitefly",
        help="whitefly dataset root containing the test directory",
    )
    parser.add_argument(
        "--fruit-fly-data-path",
        default=r"data\fruit_fly",
        help="fruit-fly dataset root containing the test directory",
    )
    parser.add_argument(
        "--output-dir",
        default=r"output\test",
        help="root directory for all prediction outputs",
    )
    parser.add_argument(
        "--mode", choices=("count", "points", "both"), default="both",
        help="count only, point localization only, or both",
    )
    parser.add_argument(
        "--inference-mode", choices=("full", "tiled"), default="full",
        help="predict the full image or stitch overlapping tiles",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="square tile size used by tiled inference",
    )
    parser.add_argument(
        "--tile-overlap", type=int, default=64,
        help="overlap in pixels between neighboring tiles",
    )
    parser.add_argument(
        "--tile-batch-size", type=int, default=1,
        help="number of tiles predicted in one model call",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="torch device, for example cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="maximum images per subset; 0 processes all images",
    )
    parser.add_argument(
        "--threshold", type=int, default=10,
        help="0-255 density threshold used for point localization",
    )
    parser.add_argument(
        "--overlay-alpha", type=float, default=0.3,
        help="density heatmap opacity on the source image (0-1)",
    )
    parser.add_argument(
        "--visualization-style", choices=("points", "density"),
        default="points",
        help="draw local density peaks or a smooth density overlay",
    )
    parser.add_argument(
        "--point-radius", type=int, default=2,
        help="point marker radius on the original image",
    )
    parser.add_argument(
        "--point-alpha", type=float, default=0.6,
        help="point marker opacity (0-1)",
    )
    parser.add_argument(
        "--peak-min-distance", type=int, default=1,
        help="local-maximum suppression radius in density-map pixels",
    )
    parser.add_argument(
        "--show-boxes", action="store_true",
        help="draw connected-component bounding boxes; disabled by default",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_device(requested):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def load_model(model_path, device):
    model = vgg19(pretrained=False)
    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    checkpoint = torch.load(model_path, **load_kwargs)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def normalize_density(density):
    minimum = float(density.min())
    maximum = float(density.max())
    if maximum <= minimum:
        return np.zeros_like(density, dtype=np.uint8)
    return ((density - minimum) * 255.0 / (maximum - minimum)).astype(np.uint8)


def locate_points(density, image_shape, threshold):
    """Return bounding boxes for thresholded density-map components."""
    normalized = normalize_density(density)
    mask = (normalized >= threshold).astype(np.uint8)
    # An odd-sized kernel keeps component coordinates spatially centered.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

    image_height, image_width = image_shape[:2]
    density_height, density_width = normalized.shape
    scale_x = image_width / float(density_width)
    scale_y = image_height / float(density_height)
    boxes = []

    # Index 0 is the background component and must not become a prediction.
    for component in stats[1:]:
        x, y, width, height = component[:4]
        boxes.append((
            max(0, int(round(x * scale_x))),
            max(0, int(round(y * scale_y))),
            min(image_width - 1, int(round((x + width) * scale_x)) - 1),
            min(image_height - 1, int(round((y + height) * scale_y)) - 1),
        ))
    return boxes, normalized


def locate_peaks(density, image_shape, threshold, min_distance):
    """Map local density maxima to point centers on the original image."""
    if min_distance < 1:
        raise ValueError("--peak-min-distance must be at least 1")
    normalized = normalize_density(density)
    smoothed = cv2.GaussianBlur(normalized, (3, 3), 0)
    kernel_size = 2 * min_distance + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_maximum = smoothed == cv2.dilate(smoothed, kernel)
    peak_mask = (local_maximum & (smoothed >= threshold)).astype(np.uint8)

    # Collapse flat maximum plateaus to one center point.
    _, _, _, centroids = cv2.connectedComponentsWithStats(
        peak_mask, connectivity=8)
    image_height, image_width = image_shape[:2]
    density_height, density_width = normalized.shape
    scale_x = image_width / float(density_width)
    scale_y = image_height / float(density_height)
    peaks = []
    for center_x, center_y in centroids[1:]:
        peaks.append((
            min(image_width - 1, max(0, int(round((center_x + 0.5) * scale_x)))),
            min(image_height - 1, max(0, int(round((center_y + 0.5) * scale_y)))),
        ))
    return peaks


def apply_density_overlay(
        image, normalized_density, threshold, color, alpha):
    """Blend a smoothly resized density map using species-specific color."""
    image_height, image_width = image.shape[:2]
    resized_density = cv2.resize(
        normalized_density, (image_width, image_height),
        interpolation=cv2.INTER_CUBIC,
    )
    strength = resized_density.astype(np.float32) / float(max(threshold, 1))
    strength = np.clip(strength, 0.0, 1.0)
    strength[resized_density < threshold] = 0.0
    alpha_map = (alpha * strength)[:, :, np.newaxis]
    color_layer = np.empty_like(image, dtype=np.float32)
    color_layer[:] = color
    blended = image.astype(np.float32) * (1.0 - alpha_map) + color_layer * alpha_map
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_point_overlay(image, points, color, radius, alpha):
    """Draw semi-transparent point markers without hiding source objects."""
    point_layer = image.copy()
    for point in points:
        cv2.circle(point_layer, point, radius, color, -1, cv2.LINE_AA)
    return cv2.addWeighted(point_layer, alpha, image, 1.0 - alpha, 0.0)


def tile_starts(length, tile_size, overlap):
    """Return tile origins that cover an axis and align the final tile."""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def predict_full_image(model, inputs, device):
    """Predict both density maps with one full-image model call."""
    with torch.no_grad():
        wf_density, _, ff_density, _ = model(
            inputs.to(device, non_blocking=True))
    return wf_density[0, 0].cpu().numpy(), ff_density[0, 0].cpu().numpy(), 1


def predict_tiled_image(
        model, inputs, device, tile_size, overlap, tile_batch_size,
        downsample_ratio=8):
    """Predict overlapping tiles and average them in density-map space."""
    _, _, image_height, image_width = inputs.shape
    if image_height % downsample_ratio or image_width % downsample_ratio:
        raise ValueError(
            "tiled inference requires image dimensions divisible by {}: {}x{}".format(
                downsample_ratio, image_width, image_height))
    if tile_size <= 0 or tile_size % downsample_ratio:
        raise ValueError(
            "--tile-size must be positive and divisible by {}".format(
                downsample_ratio))
    if overlap < 0 or overlap >= tile_size or overlap % downsample_ratio:
        raise ValueError(
            "--tile-overlap must be non-negative, smaller than tile size, "
            "and divisible by {}".format(downsample_ratio))
    if tile_batch_size <= 0:
        raise ValueError("--tile-batch-size must be positive")

    effective_height = min(tile_size, image_height)
    effective_width = min(tile_size, image_width)
    y_starts = tile_starts(image_height, effective_height, overlap)
    x_starts = tile_starts(image_width, effective_width, overlap)
    coordinates = [(y, x) for y in y_starts for x in x_starts]

    density_height = image_height // downsample_ratio
    density_width = image_width // downsample_ratio
    # Keep every tile output and the complete stitching process on the target
    # device. Both species are transferred to CPU together only after all
    # batches have been accumulated and normalized.
    wf_sum = torch.zeros(
        (density_height, density_width), dtype=torch.float32, device=device)
    ff_sum = torch.zeros_like(wf_sum)
    weights = torch.zeros_like(wf_sum)

    for offset in range(0, len(coordinates), tile_batch_size):
        batch_coordinates = coordinates[offset:offset + tile_batch_size]
        tiles = torch.cat([
            inputs[:, :, y:y + effective_height, x:x + effective_width]
            for y, x in batch_coordinates
        ], dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            wf_batch, _, ff_batch, _ = model(tiles)
        wf_batch = wf_batch[:, 0]
        ff_batch = ff_batch[:, 0]

        for tile_index, (y, x) in enumerate(batch_coordinates):
            density_y = y // downsample_ratio
            density_x = x // downsample_ratio
            tile_height, tile_width = wf_batch[tile_index].shape
            region = (
                slice(density_y, density_y + tile_height),
                slice(density_x, density_x + tile_width),
            )
            wf_sum[region] += wf_batch[tile_index]
            ff_sum[region] += ff_batch[tile_index]
            weights[region] += 1.0

    stitched = torch.stack((wf_sum / weights, ff_sum / weights))
    stitched = stitched.cpu().numpy()
    if not np.isfinite(stitched).all():
        raise RuntimeError("tiled inference produced uncovered density pixels")
    return stitched[0], stitched[1], len(coordinates)


def save_positions(path, class_id, boxes):
    with open(path, "w", encoding="utf-8") as stream:
        for x1, y1, x2, y2 in boxes:
            stream.write("{} 1 {} {} {} {}\n".format(class_id, x1, y1, x2, y2))


def image_path_for(subset_path, name):
    for extension in (".jpg", ".JPG", ".jpeg", ".png"):
        candidate = os.path.join(subset_path, name + extension)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("source image not found for {} in {}".format(name, subset_path))


def ensure_output_dirs(root):
    directories = {
        "visualizations": os.path.join(root, "visualizations"),
        "wf_density": os.path.join(root, "density", "whitefly"),
        "ff_density": os.path.join(root, "density", "fruit-fly"),
        "wf_points": os.path.join(root, "positions", "whitefly"),
        "ff_points": os.path.join(root, "positions", "fruit-fly"),
    }
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    return directories


def main():
    args = parse_args()
    if not 0 <= args.threshold <= 255:
        raise ValueError("--threshold must be between 0 and 255")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")
    if args.point_radius <= 0:
        raise ValueError("--point-radius must be positive")
    if not 0.0 <= args.point_alpha <= 1.0:
        raise ValueError("--point-alpha must be between 0 and 1")
    device = resolve_device(args.device)
    output_dirs = ensure_output_dirs(args.output_dir)
    summary_path = os.path.join(args.output_dir, "summary.csv")

    print("[CONFIG] model: {}".format(os.path.abspath(args.model_path)), flush=True)
    print("[CONFIG] input: {}".format(os.path.abspath(args.data_path)), flush=True)
    print("[CONFIG] fruit-fly input: {}".format(
        os.path.abspath(args.fruit_fly_data_path)), flush=True)
    print("[CONFIG] output: {}".format(os.path.abspath(args.output_dir)), flush=True)
    print("[CONFIG] mode: {}; device: {}".format(args.mode, device), flush=True)
    print("[CONFIG] inference: {}".format(args.inference_mode), flush=True)
    if args.inference_mode == "tiled":
        print(
            "[CONFIG] tiles: size={}, overlap={}, batch-size={}".format(
                args.tile_size, args.tile_overlap, args.tile_batch_size),
            flush=True,
        )
    print("[CONFIG] boxes: {}".format(
        "enabled" if args.show_boxes else "disabled"), flush=True)
    print("[CONFIG] visualization: {}".format(
        args.visualization_style), flush=True)

    model = load_model(args.model_path, device)
    rows = []
    total_inference_seconds = 0.0

    subset_name = "test"
    subset_path = os.path.join(args.data_path, subset_name)
    fruit_fly_subset_path = os.path.join(args.fruit_fly_data_path, subset_name)
    if not os.path.isdir(subset_path):
        raise FileNotFoundError("test directory not found: {}".format(subset_path))
    if not os.path.isdir(fruit_fly_subset_path):
        raise FileNotFoundError(
            "fruit-fly test directory not found: {}".format(fruit_fly_subset_path))

    dataset = Crowd_qnrf(
        subset_path, fruit_fly_subset_path, 512, 8, method="val")
    if args.max_images > 0:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    for index, (inputs, wf_truth, ff_truth, names) in enumerate(loader, start=1):
            name = names[0]
            source_path = image_path_for(subset_path, name)
            source = cv2.imread(source_path)
            if source is None:
                raise RuntimeError("OpenCV could not read {}".format(source_path))

            started = time.time()
            if args.inference_mode == "tiled":
                wf_array, ff_array, tile_count = predict_tiled_image(
                    model,
                    inputs,
                    device,
                    args.tile_size,
                    args.tile_overlap,
                    args.tile_batch_size,
                )
            else:
                wf_array, ff_array, tile_count = predict_full_image(
                    model, inputs, device)
            elapsed = time.time() - started
            total_inference_seconds += elapsed
            image_fps = 1.0 / elapsed if elapsed > 0 else float("inf")
            tile_fps = tile_count / elapsed if elapsed > 0 else float("inf")

            wf_count = float(wf_array.sum())
            ff_count = float(ff_array.sum())
            wf_actual = float(wf_truth[0])
            ff_actual = float(ff_truth[0])

            print(
                "[PREDICT] {}/{} {}/{} {} | WF {:.2f}/{:.0f} | FF {:.2f}/{:.0f} | "
                "tiles {} | {:.2f}s | FPS {:.2f} | tile FPS {:.2f}".format(
                    subset_name, name, index, len(loader), name,
                    wf_count, wf_actual, ff_count, ff_actual, tile_count,
                    elapsed, image_fps, tile_fps,
                ),
                flush=True,
            )

            wf_boxes = []
            ff_boxes = []
            wf_peaks = []
            ff_peaks = []
            if args.mode in ("points", "both"):
                wf_boxes, wf_normalized = locate_points(
                    wf_array, source.shape, args.threshold)
                ff_boxes, ff_normalized = locate_points(
                    ff_array, source.shape, args.threshold)
                wf_peaks = locate_peaks(
                    wf_array, source.shape, args.threshold,
                    args.peak_min_distance)
                ff_peaks = locate_peaks(
                    ff_array, source.shape, args.threshold,
                    args.peak_min_distance)

                wf_text = os.path.join(output_dirs["wf_points"], "{}_{}.txt".format(subset_name, name))
                ff_text = os.path.join(output_dirs["ff_points"], "{}_{}.txt".format(subset_name, name))
                save_positions(wf_text, SPECIES["wf"]["class_id"], wf_boxes)
                save_positions(ff_text, SPECIES["ff"]["class_id"], ff_boxes)
                print("[SAVE] whitefly positions: {}".format(os.path.abspath(wf_text)), flush=True)
                print("[SAVE] fruit-fly positions: {}".format(os.path.abspath(ff_text)), flush=True)

                if args.visualization_style == "density":
                    overlay = apply_density_overlay(
                        source, wf_normalized, args.threshold,
                        SPECIES["wf"]["color"], args.overlay_alpha,
                    )
                    overlay = apply_density_overlay(
                        overlay, ff_normalized, args.threshold,
                        SPECIES["ff"]["color"], args.overlay_alpha,
                    )
                else:
                    overlay = apply_point_overlay(
                        source, wf_peaks, SPECIES["wf"]["color"],
                        args.point_radius, args.point_alpha)
                    overlay = apply_point_overlay(
                        overlay, ff_peaks, SPECIES["ff"]["color"],
                        args.point_radius, args.point_alpha)
                if args.show_boxes:
                    for box in wf_boxes:
                        cv2.rectangle(
                            overlay, box[:2], box[2:], SPECIES["wf"]["color"], 2)
                    for box in ff_boxes:
                        cv2.rectangle(
                            overlay, box[:2], box[2:], SPECIES["ff"]["color"], 2)
                cv2.putText(
                    overlay, "Whitefly: {:.2f}".format(wf_count),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, SPECIES["wf"]["color"], 2, cv2.LINE_AA,
                )
                cv2.putText(
                    overlay, "Fruit fly: {:.2f}".format(ff_count),
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, SPECIES["ff"]["color"], 2, cv2.LINE_AA,
                )
                visualization = os.path.join(
                    output_dirs["visualizations"], "{}_{}.jpg".format(subset_name, name))
                cv2.imwrite(visualization, overlay)
                print("[SAVE] visualization: {}".format(os.path.abspath(visualization)), flush=True)
            else:
                wf_normalized = normalize_density(wf_array)
                ff_normalized = normalize_density(ff_array)

            wf_heatmap = cv2.applyColorMap(wf_normalized, cv2.COLORMAP_JET)
            ff_heatmap = cv2.applyColorMap(ff_normalized, cv2.COLORMAP_JET)
            wf_density_path = os.path.join(
                output_dirs["wf_density"], "{}_{}.png".format(subset_name, name))
            ff_density_path = os.path.join(
                output_dirs["ff_density"], "{}_{}.png".format(subset_name, name))
            cv2.imwrite(wf_density_path, wf_heatmap)
            cv2.imwrite(ff_density_path, ff_heatmap)
            print("[SAVE] whitefly density: {}".format(os.path.abspath(wf_density_path)), flush=True)
            print("[SAVE] fruit-fly density: {}".format(os.path.abspath(ff_density_path)), flush=True)

            rows.append({
                "subset": subset_name,
                "image": name,
                "whitefly_actual": wf_actual,
                "whitefly_predicted": wf_count,
                "whitefly_points": len(wf_peaks),
                "fruit_fly_actual": ff_actual,
                "fruit_fly_predicted": ff_count,
                "fruit_fly_points": len(ff_peaks),
                "inference_seconds": elapsed,
                "inference_mode": args.inference_mode,
                "tile_count": tile_count,
                "fps": image_fps,
                "tile_fps": tile_fps,
            })

    fieldnames = [
        "subset", "image", "whitefly_actual", "whitefly_predicted", "whitefly_points",
        "fruit_fly_actual", "fruit_fly_predicted", "fruit_fly_points",
        "inference_seconds", "inference_mode", "tile_count", "fps", "tile_fps",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[SAVE] summary: {}".format(os.path.abspath(summary_path)), flush=True)
    print("[DONE] processed {} images".format(len(rows)), flush=True)
    if rows and total_inference_seconds > 0:
        overall_fps = len(rows) / total_inference_seconds
        total_tiles = sum(row["tile_count"] for row in rows)
        overall_tile_fps = total_tiles / total_inference_seconds
        print("[FPS] images: {:.2f}; tiles: {:.2f}; inference time: {:.2f}s".format(
            overall_fps, overall_tile_fps, total_inference_seconds), flush=True)
    print("[DONE] output root: {}".format(os.path.abspath(args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
