# Legacy Test Dataset Preprocessing Pipeline

This directory preserves the original scripts used to prepare the historical
MS-DM test dataset. They are not called by `train.py` or `test.py`.

The main preprocessing paths now use `data/raw_test/`. Some historical helper
and analysis scripts still refer to unavailable inputs, and some original
comments have damaged character encoding.

## Raw Test Data

```text
data/raw_test/
|-- whitefly/
|   |-- images/                   # High-resolution source images
|   |-- annotations/              # Point annotations in text format
|   |-- train.txt                 # Intentionally empty
|   |-- val.txt                   # Intentionally empty
|   `-- test.txt
`-- fruit_fly/
    |-- images/
    |-- annotations/
    |-- train.txt                 # Intentionally empty
    |-- val.txt                   # Intentionally empty
    `-- test.txt
```

Each species has an independent `test.txt`, keeping its image list consistent
with its annotations. The raw dataset is ignored by Git because the image files
are large.

## Current Model-Ready Data

```text
data/
|-- whitefly/test/
`-- fruit_fly/test/
```

Both paths are passed explicitly to the dataset loader.

## Scripts

### `generate_mat.py`

Converts both species' `annotations/*.txt` files into MATLAB files under their
respective `mats/` directories:

```powershell
python tools\preprocess\legacy_test_pipeline\generate_mat.py
```

### `preprocess_dataset_nwpu-test-04.py`

Reads the whitefly source images and MATLAB annotations, preserves aspect ratio
while resizing, transforms point coordinates, and writes `.jpg + .npy` files to
`data/whitefly/`.

### `preprocess_dataset_nwpu-test-05.py`

Performs the same conversion for fruit fly and writes to
`data/fruit_fly/`. It also copies missing fruit-fly
images into the first branch so that the first branch contains the union of test
images expected by the dataset loader.

Run the conversion in this order:

```powershell
python tools\preprocess\legacy_test_pipeline\generate_mat.py
python tools\preprocess\legacy_test_pipeline\preprocess_dataset_nwpu-test-04.py
python tools\preprocess\legacy_test_pipeline\preprocess_dataset_nwpu-test-05.py
```

### `Splitting_images _and_coordinates-01.py`

Optionally divides high-resolution images and point annotations into a `3 x 3`,
`4 x 4`, or `8 x 8` grid, depending on source resolution. Tile names end in a
row and column index:

```text
image_name_00.jpg
image_name_01.jpg
image_name_10.jpg
```

Generated tiles are written below each species' `tiles/` directory. The current
model-ready test dataset uses resized full images, not these optional tiles.

### `split-train-val-test-name-to-txt-03.py`

Historical fruit-fly split-list generator. Its current configuration assigns
all discovered images to `test.txt`. The supplied lists are already complete,
so it normally does not need to be run.

### `pre_saaign-00.py`

Historical category-assignment and file-copying code. Its original
`test-all-ori` input is unavailable, so this file is retained as a reference.

### `相关性分析-last.py`

Historical correlation-analysis code for old prediction logs. Those logs are
not part of the current pipeline, so this file is retained as a reference.

### `HISTORICAL_WORKFLOW.md`

The original Chinese processing notes. They are preserved unchanged and refer
to the former `test_images-pre-result-full/` layout.

## Output Contract

Every model-ready image uses a same-stem NumPy point file:

```text
sample.jpg
sample.npy
```

Each `.npy` file contains point coordinates in `(x, y)` order. File names in
the two species branches must remain aligned so the loader can associate them.

## Dependencies

The conversion scripts use `numpy`, `scipy`, `Pillow`, and `opencv-python`.
The historical correlation script additionally references `matplotlib` and
`PySide2`; these are not required for training or prediction.

## Missing Historical Step

The original JSON-to-point-text converter and its JSON inputs are not present.
The preserved pipeline therefore starts from the high-resolution images and TXT
point annotations in `data/raw_test/`.
