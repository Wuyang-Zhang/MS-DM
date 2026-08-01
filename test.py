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
    "wf": {"label": "whitefly", "class_id": 1, "half_box": 16, "color": (0, 0, 255)},
    "ff": {"label": "fruit-fly", "class_id": 2, "half_box": 32, "color": (0, 255, 255)},
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
        default=r"test_images-pre-result-full\data-used-by-train-val-test",
        help="directory containing test/train/val subset directories",
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
        "--device", default="cuda",
        help="torch device, for example cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--subsets", nargs="*", default=None,
        help="subset directories to process; default processes every directory",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="maximum images per subset; 0 processes all images",
    )
    parser.add_argument(
        "--threshold", type=int, default=10,
        help="0-255 density threshold used for point localization",
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


def locate_points(density, image_shape, threshold, half_box):
    normalized = normalize_density(density)
    mask = (normalized >= threshold).astype(np.uint8)
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, _, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

    image_height, image_width = image_shape[:2]
    density_height, density_width = normalized.shape
    scale_x = image_width / float(density_width)
    scale_y = image_height / float(density_height)
    boxes = []

    # Index 0 is the background component and must not become a prediction.
    for center_x, center_y in centroids[1:]:
        x = center_x * scale_x
        y = center_y * scale_y
        boxes.append((
            max(0, int(round(x - half_box))),
            max(0, int(round(y - half_box))),
            min(image_width - 1, int(round(x + half_box))),
            min(image_height - 1, int(round(y + half_box))),
        ))
    return boxes, normalized


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


def selected_subsets(data_path, requested):
    available = sorted(
        name for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))
    )
    if requested is None:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError("subset directories not found: {}".format(", ".join(missing)))
    return requested


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_dirs = ensure_output_dirs(args.output_dir)
    summary_path = os.path.join(args.output_dir, "summary.csv")

    print("[CONFIG] model: {}".format(os.path.abspath(args.model_path)), flush=True)
    print("[CONFIG] input: {}".format(os.path.abspath(args.data_path)), flush=True)
    print("[CONFIG] output: {}".format(os.path.abspath(args.output_dir)), flush=True)
    print("[CONFIG] mode: {}; device: {}".format(args.mode, device), flush=True)

    model = load_model(args.model_path, device)
    rows = []

    for subset_name in selected_subsets(args.data_path, args.subsets):
        subset_path = os.path.join(args.data_path, subset_name)
        dataset = Crowd_qnrf(subset_path, 512, 8, method="val")
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
            with torch.no_grad():
                wf_density, _, ff_density, _ = model(inputs.to(device, non_blocking=True))
            elapsed = time.time() - started

            wf_array = wf_density[0, 0].detach().cpu().numpy()
            ff_array = ff_density[0, 0].detach().cpu().numpy()
            wf_count = float(wf_array.sum())
            ff_count = float(ff_array.sum())
            wf_actual = float(wf_truth[0])
            ff_actual = float(ff_truth[0])

            print(
                "[PREDICT] {}/{} {}/{} {} | WF {:.2f}/{:.0f} | FF {:.2f}/{:.0f} | {:.2f}s".format(
                    subset_name, name, index, len(loader), name,
                    wf_count, wf_actual, ff_count, ff_actual, elapsed,
                ),
                flush=True,
            )

            wf_boxes = []
            ff_boxes = []
            if args.mode in ("points", "both"):
                wf_boxes, wf_normalized = locate_points(
                    wf_array, source.shape, args.threshold, SPECIES["wf"]["half_box"])
                ff_boxes, ff_normalized = locate_points(
                    ff_array, source.shape, args.threshold, SPECIES["ff"]["half_box"])

                wf_text = os.path.join(output_dirs["wf_points"], "{}_{}.txt".format(subset_name, name))
                ff_text = os.path.join(output_dirs["ff_points"], "{}_{}.txt".format(subset_name, name))
                save_positions(wf_text, SPECIES["wf"]["class_id"], wf_boxes)
                save_positions(ff_text, SPECIES["ff"]["class_id"], ff_boxes)
                print("[SAVE] whitefly positions: {}".format(os.path.abspath(wf_text)), flush=True)
                print("[SAVE] fruit-fly positions: {}".format(os.path.abspath(ff_text)), flush=True)

                overlay = source.copy()
                for box in wf_boxes:
                    cv2.rectangle(overlay, box[:2], box[2:], SPECIES["wf"]["color"], 2)
                for box in ff_boxes:
                    cv2.rectangle(overlay, box[:2], box[2:], SPECIES["ff"]["color"], 2)
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
                "whitefly_points": len(wf_boxes),
                "fruit_fly_actual": ff_actual,
                "fruit_fly_predicted": ff_count,
                "fruit_fly_points": len(ff_boxes),
                "inference_seconds": elapsed,
            })

    fieldnames = [
        "subset", "image", "whitefly_actual", "whitefly_predicted", "whitefly_points",
        "fruit_fly_actual", "fruit_fly_predicted", "fruit_fly_points", "inference_seconds",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[SAVE] summary: {}".format(os.path.abspath(summary_path)), flush=True)
    print("[DONE] processed {} images".format(len(rows)), flush=True)
    print("[DONE] output root: {}".format(os.path.abspath(args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
