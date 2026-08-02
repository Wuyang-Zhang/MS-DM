"""Benchmark MS-DM model-forward and overall inference FPS."""

import argparse
import csv
import os
import time

import cv2
import torch

from predict import find_images, image_tensor
from test import (load_model, predict_full_image, predict_tiled_image,
                  resolve_device, tile_starts)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MS-DM inference FPS")
    parser.add_argument("--input-path", default=r"data\predict")
    parser.add_argument(
        "--model-path", default=r"pretrained_models\msdm_final_v3_legacy.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend", choices=("pytorch", "tensorrt"), default="pytorch")
    parser.add_argument(
        "--engine-path", default=r"output\tensorrt\msdm_tiled_fp16.engine")
    parser.add_argument(
        "--disable-cuda-graph", action="store_true",
        help="disable TensorRT CUDA Graph capture for comparison/debugging",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("full", "tiled"), default=None,
        help="defaults to full+tiled for PyTorch and tiled for TensorRT",
    )
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--output-csv", default=r"output\benchmark\fps.csv")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_arguments(args):
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be positive")
    if args.tile_size <= 0 or args.tile_size % 8:
        raise ValueError("--tile-size must be positive and divisible by 8")
    if (args.tile_overlap < 0 or args.tile_overlap >= args.tile_size
            or args.tile_overlap % 8):
        raise ValueError(
            "--tile-overlap must be non-negative, smaller than tile size, "
            "and divisible by 8")
    if args.tile_batch_size <= 0:
        raise ValueError("--tile-batch-size must be positive")


def tiled_batches(inputs, device, tile_size, overlap, batch_size):
    """Prepare device-resident batches for forward-only measurement."""
    _, _, image_height, image_width = inputs.shape
    effective_height = min(tile_size, image_height)
    effective_width = min(tile_size, image_width)
    y_starts = tile_starts(image_height, effective_height, overlap)
    x_starts = tile_starts(image_width, effective_width, overlap)
    coordinates = [(y, x) for y in y_starts for x in x_starts]
    batches = []
    for offset in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[offset:offset + batch_size]
        batches.append(torch.cat([
            inputs[:, :, y:y + effective_height, x:x + effective_width]
            for y, x in batch_coordinates
        ], dim=0).to(device))
    return batches, len(coordinates)


def run_forward(model, batches):
    with torch.no_grad():
        for batch in batches:
            model(batch)


def benchmark_forward(model, batches, tile_count, device, warmups, runs):
    for _ in range(warmups):
        run_forward(model, batches)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(runs):
        run_forward(model, batches)
    synchronize(device)
    total_seconds = time.perf_counter() - started
    return {
        "forward_seconds": total_seconds / runs,
        "forward_fps": runs / total_seconds,
        "forward_tile_fps": tile_count * runs / total_seconds,
    }


def run_overall(model, inputs, mode, device, args):
    if mode == "tiled":
        return predict_tiled_image(
            model, inputs, device, args.tile_size,
            args.tile_overlap, args.tile_batch_size)
    return predict_full_image(model, inputs, device)


def benchmark_overall(model, inputs, mode, device, args, tile_count):
    for _ in range(args.warmup_runs):
        run_overall(model, inputs, mode, device, args)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(args.benchmark_runs):
        run_overall(model, inputs, mode, device, args)
    synchronize(device)
    total_seconds = time.perf_counter() - started
    return {
        "overall_seconds": total_seconds / args.benchmark_runs,
        "overall_fps": args.benchmark_runs / total_seconds,
        "overall_tile_fps": tile_count * args.benchmark_runs / total_seconds,
    }


def benchmark_image(model, image_path, mode, device, args):
    source = cv2.imread(image_path)
    if source is None:
        raise RuntimeError("OpenCV could not read {}".format(image_path))
    inputs = image_tensor(source)
    if mode == "tiled":
        batches, tile_count = tiled_batches(
            inputs, device, args.tile_size,
            args.tile_overlap, args.tile_batch_size)
    else:
        batches, tile_count = [inputs.to(device)], 1

    row = {
        "image": image_path,
        "width": source.shape[1],
        "height": source.shape[0],
        "mode": mode,
        "tile_size": args.tile_size if mode == "tiled" else "",
        "tile_overlap": args.tile_overlap if mode == "tiled" else "",
        "tile_batch_size": args.tile_batch_size if mode == "tiled" else 1,
        "tile_count": tile_count,
        "warmup_runs": args.warmup_runs,
        "benchmark_runs": args.benchmark_runs,
    }
    row.update(benchmark_forward(
        model, batches, tile_count, device,
        args.warmup_runs, args.benchmark_runs))
    row.update(benchmark_overall(
        model, inputs, mode, device, args, tile_count))
    return row


def print_result(row):
    print(
        "[FPS] {mode} | {width}x{height} | tiles {tile_count} | "
        "forward {forward_fps:.2f} image/s, {forward_tile_fps:.2f} tile/s | "
        "overall {overall_fps:.2f} image/s, {overall_tile_fps:.2f} tile/s".format(
            **row), flush=True)


def print_mode_averages(rows):
    """Print aggregate throughput across all benchmark images per mode."""
    for mode in ("full", "tiled"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        if not mode_rows:
            continue
        image_count = len(mode_rows)
        tile_count = sum(row["tile_count"] for row in mode_rows)
        forward_seconds = sum(row["forward_seconds"] for row in mode_rows)
        overall_seconds = sum(row["overall_seconds"] for row in mode_rows)
        print(
            "[AVERAGE FPS] {} | forward {:.2f} image/s, {:.2f} tile/s | "
            "overall {:.2f} image/s, {:.2f} tile/s".format(
                mode,
                image_count / forward_seconds,
                tile_count / forward_seconds,
                image_count / overall_seconds,
                tile_count / overall_seconds,
            ),
            flush=True,
        )


def main():
    args = parse_args()
    if args.modes is None:
        args.modes = ("tiled",) if args.backend == "tensorrt" else (
            "full", "tiled")
    validate_arguments(args)
    images, _ = find_images(args.input_path)
    device = resolve_device(args.device)
    if args.backend == "tensorrt":
        if device.type != "cuda":
            raise ValueError("TensorRT backend requires --device cuda")
        from tools.tensorrt.runtime import TensorRTRunner
        model = TensorRTRunner(
            args.engine_path, device,
            use_cuda_graph=not args.disable_cuda_graph)
    else:
        model = load_model(args.model_path, device)
    rows = []
    print("[CONFIG] backend: {}; device: {}; images: {}; modes: {}".format(
        args.backend, device, len(images), ", ".join(args.modes)), flush=True)
    print("[CONFIG] warmup: {}; runs: {}".format(
        args.warmup_runs, args.benchmark_runs), flush=True)
    if args.backend == "tensorrt":
        print("[CONFIG] CUDA Graph: {}".format(
            "disabled" if args.disable_cuda_graph else "enabled"), flush=True)
    for image_path in images:
        for mode in args.modes:
            row = benchmark_image(model, image_path, mode, device, args)
            rows.append(row)
            print_result(row)

    print_mode_averages(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    fieldnames = [
        "image", "width", "height", "mode", "tile_size", "tile_overlap",
        "tile_batch_size", "tile_count", "warmup_runs", "benchmark_runs",
        "forward_seconds", "forward_fps", "forward_tile_fps",
        "overall_seconds", "overall_fps", "overall_tile_fps",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("[SAVE] benchmark CSV: {}".format(
        os.path.abspath(args.output_csv)), flush=True)


if __name__ == "__main__":
    main()
