# MS-DM

PyTorch implementation of **A Multi-Species Pest Recognition and Counting Method Based on a Density Map in the Greenhouse**, published in *Computers and Electronics in Agriculture* (2024).

- Paper: [https://doi.org/10.1016/j.compag.2023.108554](https://doi.org/10.1016/j.compag.2023.108554)
- Android application: [MS_DM_Android](https://github.com/Wuyang-Zhang/MS_DM_Android)

## Project Structure

```text
MS-DM/
|-- data/                         # Training and validation datasets
|-- datasets/                     # Dataset loaders and augmentation
|-- losses/                       # OT, Sinkhorn, and related losses
|-- models/                       # DM-Count and MS-DM model definitions
|-- tools/                        # Data preprocessing utilities and legacy pipeline
|-- output/                       # Generated checkpoints, logs, and predictions
|-- pretrained_models/            # Downloaded pretrained weights
|-- train.py                      # Training entry point
|-- train_helper.py               # Training and validation routines
|-- test.py                       # Evaluation on the labeled test split
|-- predict.py                    # Prediction on unlabeled images
|-- benchmark_fps.py              # Full and tiled FPS benchmark
`-- profile_model.py              # Parameters, model size, MACs, and FLOPs
```

`models/msdm.py` contains the final MS-DM model used by the training and testing scripts. `models/dm_count.py` retains the baseline DM-Count-style model for comparison.

## Dataset Layout

See [`data/readme.md`](data/readme.md) for a file-by-file guide, normal usage,
and the historical dataset-preparation order.

The annotations for the two species are stored in parallel directory trees. Corresponding images and annotation files must have the same file names in both trees.

```text
data/
|-- whitefly/
|   |-- train/
|   |-- val/
|   `-- test/
`-- fruit_fly/
    |-- train/
    |-- val/
    `-- test/
```

Each image must have a matching NumPy annotation file containing point coordinates in `(x, y)` order:

```text
0001.jpg
0001.npy
```

- `whitefly` contains images and annotations for the whitefly counting branch.
- `fruit_fly` contains images and annotations for the fruit-fly counting branch.
- The default training crop size is `512 x 512` pixels.
- The predicted density maps have one eighth of the input image width and height.

Preprocessing utilities are available under `data/` and `tools/preprocess/`. The original test-data preparation scripts and their documentation are preserved in [`tools/preprocess/legacy_test_pipeline`](tools/preprocess/legacy_test_pipeline/README.md). Check their path arguments before running them because historical paths may not match your machine.

High-resolution test sources and their original point annotations are preserved locally under `data/raw_test/`. This directory is ignored by Git because it contains large dataset files.

## Pretrained Weights

Pretrained weights will be provided on the [GitHub Releases page](https://github.com/Wuyang-Zhang/MS-DM/releases).

## Environment

The original reproduction environment used:

- Python 3.7
- PyTorch 1.2.0
- TorchVision 0.4.0
- CUDA 10.0

To reproduce the legacy environment:

```powershell
conda activate msdm
```

For current NVIDIA GPUs, use a recent PyTorch/CUDA environment. Install a compatible CUDA-enabled PyTorch build first, then install the project dependencies:

```powershell
pip install -r requirements.txt
```

## Training

Start training with the default configuration:

```powershell
conda activate msdm
python train.py
```

Example with explicit arguments:

```powershell
python train.py `
  --whitefly-data-dir data\whitefly `
  --fruit-fly-data-dir data\fruit_fly `
  --batch-size 10 `
  --max-epoch 200 `
  --log-interval 1 `
  --device 0
```

The program automatically creates the runtime directories:

```text
output/
|-- ckpts/                        # Model checkpoints
|-- log/                          # Training metrics and run logs
|-- runs/                         # TensorBoard event files
`-- tmp/                          # Temporary files
```

Training progress is printed for every configured log interval. MS-DM computes an Optimal Transport loss for both species, with 100 Sinkhorn iterations by default, so a training batch can take considerable time.

## Testing on the Labeled Dataset

`test.py` evaluates the labeled `data/whitefly/test` and `data/fruit_fly/test`
splits. It produces predicted and actual counts, density maps, point locations,
combined annotated images, and a CSV summary.

```powershell
python test.py `
  --model-path path\to\model.pth `
  --data-path data\whitefly `
  --fruit-fly-data-path data\fruit_fly `
  --output-dir output\test `
  --mode both `
  --device cuda
```

Available modes:

```powershell
python test.py --mode count    # Counts and density maps
python test.py --mode points   # Point locations, visualizations, and density maps
python test.py --mode both     # All outputs (default)
```

For a quick check, limit the number of images processed in each subset:

```powershell
python test.py --max-images 1
```

Only the `test` subset is processed; `train` and `val` are never used by `test.py`. Use `--max-images 0` to process all test images. Prediction results are saved as:

```text
output/test/
|-- density/whitefly/
|-- density/fruit-fly/
|-- positions/whitefly/
|-- positions/fruit-fly/
|-- visualizations/
`-- summary.csv
```

Each image in `visualizations/` combines both species on the original image.
Whitefly markers are red and fruit-fly markers are green. Point markers are
semi-transparent by default so the source objects remain visible. The default `points`
style detects local maxima in the low-resolution density maps and maps their
centers back to the original image. This avoids the block-like appearance caused
by nearest-neighbor enlargement of the 1/8-resolution density map. The top-left
legend displays each species name and its density-sum predicted count.

Generate point-center visualization without boxes (default):

```powershell
python test.py --mode both --visualization-style points
```

Point radius and local-maximum suppression distance are configurable:

```powershell
python test.py `
  --visualization-style points `
  --point-radius 4 `
  --point-alpha 0.55 `
  --peak-min-distance 1
```

`--threshold` sets the minimum normalized peak strength. Peak extraction uses a
`3 x 3` Gaussian smoothing kernel. `--peak-min-distance 1` applies local-maximum
suppression over a one-pixel radius in the 1/8-resolution density map, which is
approximately eight pixels on the original image. `--point-radius` affects only
the displayed marker size, while `--point-alpha` controls marker opacity.

Generate a smooth density visualization instead:

```powershell
python test.py `
  --visualization-style density `
  --threshold 10 `
  --overlay-alpha 0.45
```

The density style uses bicubic interpolation and intensity-weighted opacity,
rather than enlarging density pixels into fixed `8 x 8` blocks.

Optionally draw bounding boxes:

```powershell
python test.py --mode both --show-boxes
```

When enabled, every box is the minimum axis-aligned rectangle enclosing a
connected density region, not a fixed-size box. Boxes can be combined with
either visualization style:

```powershell
python test.py `
  --mode both `
  --visualization-style density `
  --threshold 10 `
  --overlay-alpha 0.45 `
  --show-boxes
```

## Prediction on Unlabeled Images

`predict.py` accepts an ordinary image or a directory of images. It does not
require `.npy` annotations or the dataset directory structure.

The repository includes a small open-source prediction example under
`data/predict/`. It is intentionally tracked by Git. Run it directly with:

```powershell
python predict.py
```

Predict one image with full-image inference:

```powershell
python predict.py `
  --input-path path\to\image.jpg `
  --inference-mode full
```

Predict every supported image below a directory with tiled inference:

```powershell
python predict.py `
  --input-path path\to\images `
  --inference-mode tiled `
  --tile-size 512 `
  --tile-overlap 64 `
  --tile-batch-size 1
```

Prediction boxes are disabled by default. Add `--show-boxes` when connected-
component bounding boxes are desired:

```powershell
python predict.py --input-path path\to\images --show-boxes
```

Direct prediction also defaults to point-center visualization. Select smooth
density visualization explicitly when preferred:

```powershell
python predict.py `
  --input-path path\to\images `
  --visualization-style density `
  --threshold 10 `
  --overlay-alpha 0.45
```

The default output directory is `output/predict/`:

```text
output/predict/
|-- density/whitefly/
|-- density/fruit-fly/
|-- positions/whitefly/
|-- positions/fruit-fly/
|-- visualizations/
`-- summary.csv
```

`summary.csv` records both predicted counts, local-maximum point counts,
connected-region counts, inference mode, tile count, and inference time for
every image. It also records image FPS and tile FPS.

## FPS Measurement

Both `test.py` and `predict.py` report inference throughput automatically. No
extra flag is required:

```powershell
python test.py --inference-mode full
python test.py --inference-mode tiled --tile-size 512 --tile-overlap 64
python predict.py --inference-mode tiled --tile-size 512 --tile-overlap 64
```

Each prediction log contains:

- `FPS`: complete images processed per second (`1 / inference_seconds`).
- `tile FPS`: model tiles processed per second in tiled mode.

The final `[FPS]` line reports throughput over all processed images. FPS timing
includes model forward inference and tiled density stitching. Image loading,
visualization rendering, and file writing are excluded. For a stable comparison,
run multiple images and compare modes with the same device, tile batch size, and
input set.

For a dedicated repeatable benchmark, use `benchmark_fps.py`. It tests both
full-image and tiled inference without rendering or saving prediction outputs:

```powershell
python benchmark_fps.py `
  --input-path data\predict `
  --modes full tiled `
  --warmup-runs 2 `
  --benchmark-runs 10 `
  --tile-size 512 `
  --tile-overlap 64 `
  --tile-batch-size 1
```

The benchmark reports two levels of performance for each mode:

- `forward FPS`: model forward computation only. Input tensors are already on
  the selected device; CPU transfer, tiled stitching, and output processing are
  excluded.
- `overall FPS`: device input transfer, model forward computation, prediction
  transfer back to CPU, and tiled density stitching. Image decoding and output
  rendering are excluded.

For tiled mode, both levels also report tile FPS. Results are saved to
`output/benchmark/fps.csv` by default. Use `--modes full` or `--modes tiled` to
benchmark only one mode. When the input directory contains multiple images, an
`[AVERAGE FPS]` line reports aggregate throughput for each mode.

## Model Complexity

Use the standalone profiler to report parameter counts, model storage, MACs,
and approximate FLOPs for a specified input size:

```powershell
python profile_model.py --height 512 --width 512 --batch-size 1
```

Complexity values are device-independent, so the profiler uses CPU by default
and does not occupy GPU memory. Pass `--device cuda` only if desired.

It loads `pretrained_models\msdm_final_v3_legacy.pth` by default and also
reports the checkpoint file size. To profile only the model architecture, use:

```powershell
python profile_model.py --skip-checkpoint
```

The profiler uses the convention `1 MAC = 2 FLOPs`. It counts convolution,
linear, batch-normalization, and activation operations. Functional
interpolation, concatenation, pooling, indexing, and tensor additions are not
included, so the reported FLOPs are an architecture-level approximation and
depend on the input resolution and batch size. Because profiling does not
depend on random parameter values, the script skips the model's historical
repeated random-initialization pass; checkpoint loading remains strict.

## Image Size and Tiling

Test images may be as large as `1440 x 1920` pixels. Two inference modes are
available.

Full-image inference is the default:

```powershell
python test.py --inference-mode full
```

Tiled inference divides each image in memory, predicts every tile, and stitches
the density maps back together:

```powershell
python test.py `
  --inference-mode tiled `
  --tile-size 512 `
  --tile-overlap 64 `
  --tile-batch-size 1
```

- `--tile-size` sets the square tile size and must be divisible by `8`.
- `--tile-overlap` controls overlap between adjacent tiles, must be divisible by
  `8`, and must be smaller than the tile size.
- `--tile-batch-size` controls how many tiles are sent to the model together.
  Increase it for speed only when sufficient GPU memory is available.
- Overlapping density predictions are averaged in the stitched output.
- The final tile on each axis is aligned with the image boundary, so the full
  image is covered without saving temporary tile files.

Tiled inference is useful when full images require too much GPU memory or when
an input scale closer to the `512 x 512` training crops is preferred. Counts,
density maps, point files, CSV summaries, and combined visualizations remain
full-image outputs in both modes.

## Citation

If you use this project, please cite the MS-DM paper:

```bibtex
@article{zhang2024msdm,
  title={A multi-species pest recognition and counting method based on a density map in the greenhouse},
  author={Zhang, Zhiqin and Rong, Jiacheng and Qi, Zhongxian and Yang, Yan and Zheng, Xiajun and Gao, Jin and Li, Wei and Yuan, Ting},
  journal={Computers and Electronics in Agriculture},
  volume={217},
  pages={108554},
  year={2024},
  issn={0168-1699},
  doi={10.1016/j.compag.2023.108554}
}
```

MS-DM builds on the distribution-matching framework introduced by DM-Count:

```bibtex
@inproceedings{wang2020dmcount,
  title={Distribution Matching for Crowd Counting},
  author={Wang, Boyu and Liu, Huidong and Samaras, Dimitris and Nguyen, Minh Hoai},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## License

See [LICENSE](LICENSE) for license information.
