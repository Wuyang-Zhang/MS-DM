"""Run MS-DM prediction on unlabeled images."""

import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional

from test import (
    SPECIES,
    apply_density_overlay,
    apply_point_overlay,
    load_model,
    locate_peaks,
    locate_points,
    predict_full_image,
    predict_tiled_image,
    resolve_device,
    save_positions,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict whitefly and fruit-fly density on unlabeled images")
    parser.add_argument(
        "--input-path", default=r"data\predict",
        help="an image file or directory; defaults to data/predict",
    )
    parser.add_argument(
        "--model-path", default=r"pretrained_models\msdm_final_v3_legacy.pth",
        help="model checkpoint (.pth or .tar)",
    )
    parser.add_argument("--output-dir", default=r"output\predict")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend", choices=("pytorch", "tensorrt"), default="pytorch")
    parser.add_argument(
        "--engine-path", default=r"output\tensorrt\msdm_tiled_fp16.engine")
    parser.add_argument(
        "--inference-mode", choices=("full", "tiled"), default=None,
        help="defaults to full for PyTorch and tiled for TensorRT",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument(
        "--visualization-style", choices=("points", "density"),
        default="points",
    )
    parser.add_argument("--point-radius", type=int, default=4)
    parser.add_argument("--point-alpha", type=float, default=0.55)
    parser.add_argument("--peak-min-distance", type=int, default=1)
    parser.add_argument(
        "--show-boxes", action="store_true",
        help="draw connected-component boxes; disabled by default",
    )
    return parser.parse_args()


def find_images(input_path):
    """Return images below a file or directory path."""
    absolute_path = os.path.abspath(input_path)
    if os.path.isfile(absolute_path):
        if os.path.splitext(absolute_path)[1].lower() not in IMAGE_EXTENSIONS:
            raise ValueError("unsupported image extension: {}".format(absolute_path))
        return [absolute_path], os.path.dirname(absolute_path)
    if not os.path.isdir(absolute_path):
        raise FileNotFoundError("input path not found: {}".format(absolute_path))

    images = []
    for root, _, names in os.walk(absolute_path):
        for name in names:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, name))
    images.sort()
    if not images:
        raise FileNotFoundError("no supported images found in {}".format(absolute_path))
    return images, absolute_path


def image_tensor(source):
    """Convert an OpenCV BGR image to a normalized model tensor."""
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    height, width = source.shape[:2]
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    if pad_height or pad_width:
        tensor = torch_functional.pad(
            tensor, (0, pad_width, 0, pad_height), mode="replicate")
    return tensor


def output_directories(root):
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


def safe_output_name(image_path, input_root):
    relative_path = os.path.relpath(image_path, input_root)
    stem = os.path.splitext(relative_path)[0]
    return stem.replace("..", "parent").replace("\\", "__").replace("/", "__")


def save_visualization(source, wf_density, ff_density, wf_peaks, ff_peaks,
                       wf_boxes, ff_boxes, wf_count, ff_count,
                       args, output_path):
    if args.visualization_style == "density":
        overlay = apply_density_overlay(
            source, wf_density, args.threshold,
            SPECIES["wf"]["color"], args.overlay_alpha,
        )
        overlay = apply_density_overlay(
            overlay, ff_density, args.threshold,
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
            cv2.rectangle(overlay, box[:2], box[2:], SPECIES["wf"]["color"], 2)
        for box in ff_boxes:
            cv2.rectangle(overlay, box[:2], box[2:], SPECIES["ff"]["color"], 2)
    cv2.putText(
        overlay, "Whitefly: {:.2f}".format(wf_count),
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, SPECIES["wf"]["color"], 2, cv2.LINE_AA)
    cv2.putText(
        overlay, "Fruit fly: {:.2f}".format(ff_count),
        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, SPECIES["ff"]["color"], 2, cv2.LINE_AA)
    cv2.imwrite(output_path, overlay)


def main():
    args = parse_args()
    if args.inference_mode is None:
        args.inference_mode = (
            "tiled" if args.backend == "tensorrt" else "full")
    if not 0 <= args.threshold <= 255:
        raise ValueError("--threshold must be between 0 and 255")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")
    if args.point_radius <= 0:
        raise ValueError("--point-radius must be positive")
    if not 0.0 <= args.point_alpha <= 1.0:
        raise ValueError("--point-alpha must be between 0 and 1")

    images, input_root = find_images(args.input_path)
    device = resolve_device(args.device)
    if args.backend == "tensorrt":
        if device.type != "cuda":
            raise ValueError("TensorRT backend requires --device cuda")
        from tools.tensorrt.runtime import TensorRTRunner
        model = TensorRTRunner(args.engine_path, device)
    else:
        model = load_model(args.model_path, device)
    directories = output_directories(args.output_dir)
    rows = []
    total_inference_seconds = 0.0

    print("[CONFIG] input: {}".format(os.path.abspath(args.input_path)), flush=True)
    print("[CONFIG] images: {}".format(len(images)), flush=True)
    print("[CONFIG] output: {}".format(os.path.abspath(args.output_dir)), flush=True)
    print("[CONFIG] backend: {}; inference: {}; visualization: {}; boxes: {}".format(
        args.backend, args.inference_mode, args.visualization_style,
        "enabled" if args.show_boxes else "disabled"), flush=True)

    for index, image_path in enumerate(images, start=1):
        source = cv2.imread(image_path)
        if source is None:
            print("[WARN] skipped unreadable image: {}".format(image_path), flush=True)
            continue
        inputs = image_tensor(source)
        started = time.time()
        if args.inference_mode == "tiled":
            wf_density, ff_density, tile_count = predict_tiled_image(
                model, inputs, device, args.tile_size,
                args.tile_overlap, args.tile_batch_size,
            )
        else:
            wf_density, ff_density, tile_count = predict_full_image(
                model, inputs, device)
        elapsed = time.time() - started
        total_inference_seconds += elapsed
        image_fps = 1.0 / elapsed if elapsed > 0 else float("inf")
        tile_fps = tile_count / elapsed if elapsed > 0 else float("inf")

        wf_boxes, wf_normalized = locate_points(
            wf_density, source.shape, args.threshold)
        ff_boxes, ff_normalized = locate_points(
            ff_density, source.shape, args.threshold)
        wf_peaks = locate_peaks(
            wf_density, source.shape, args.threshold, args.peak_min_distance)
        ff_peaks = locate_peaks(
            ff_density, source.shape, args.threshold, args.peak_min_distance)
        wf_count = float(wf_density.sum())
        ff_count = float(ff_density.sum())
        output_name = safe_output_name(image_path, input_root)
        visualization_path = os.path.join(
            directories["visualizations"], output_name + ".jpg")
        save_visualization(
            source, wf_normalized, ff_normalized, wf_peaks, ff_peaks,
            wf_boxes, ff_boxes, wf_count, ff_count, args, visualization_path)

        cv2.imwrite(
            os.path.join(directories["wf_density"], output_name + ".png"),
            cv2.applyColorMap(wf_normalized, cv2.COLORMAP_JET),
        )
        cv2.imwrite(
            os.path.join(directories["ff_density"], output_name + ".png"),
            cv2.applyColorMap(ff_normalized, cv2.COLORMAP_JET),
        )
        save_positions(
            os.path.join(directories["wf_points"], output_name + ".txt"),
            SPECIES["wf"]["class_id"], wf_boxes,
        )
        save_positions(
            os.path.join(directories["ff_points"], output_name + ".txt"),
            SPECIES["ff"]["class_id"], ff_boxes,
        )

        rows.append({
            "image": image_path,
            "whitefly_predicted": wf_count,
            "fruit_fly_predicted": ff_count,
            "whitefly_points": len(wf_peaks),
            "fruit_fly_points": len(ff_peaks),
            "whitefly_regions": len(wf_boxes),
            "fruit_fly_regions": len(ff_boxes),
            "inference_mode": args.inference_mode,
            "tile_count": tile_count,
            "inference_seconds": elapsed,
            "fps": image_fps,
            "tile_fps": tile_fps,
        })
        print(
            "[PREDICT] {}/{} {} | WF {:.2f} | FF {:.2f} | tiles {} | "
            "{:.2f}s | FPS {:.2f} | tile FPS {:.2f}".format(
                index, len(images), image_path, wf_count, ff_count,
                tile_count, elapsed, image_fps, tile_fps,
            ), flush=True,
        )
        print("[SAVE] visualization: {}".format(visualization_path), flush=True)

    summary_path = os.path.join(args.output_dir, "summary.csv")
    fieldnames = [
        "image", "whitefly_predicted", "fruit_fly_predicted",
        "whitefly_points", "fruit_fly_points",
        "whitefly_regions", "fruit_fly_regions", "inference_mode",
        "tile_count", "inference_seconds", "fps", "tile_fps",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("[DONE] processed {} images".format(len(rows)), flush=True)
    if rows and total_inference_seconds > 0:
        overall_fps = len(rows) / total_inference_seconds
        total_tiles = sum(row["tile_count"] for row in rows)
        overall_tile_fps = total_tiles / total_inference_seconds
        print("[FPS] images: {:.2f}; tiles: {:.2f}; inference time: {:.2f}s".format(
            overall_fps, overall_tile_fps, total_inference_seconds), flush=True)
    print("[DONE] summary: {}".format(os.path.abspath(summary_path)), flush=True)


if __name__ == "__main__":
    main()
